"""RAGAS end-to-end evaluation for the RAG QA subtask.

Usage:
    python -m scripts.evaluate_ragas --variant full
    python -m scripts.evaluate_ragas --variant all
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import json
import os
import sys
import types
import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)

from app.config import settings  # noqa: E402
from app.rag.context_builder import build_retrieval_context  # noqa: E402
from app.rag.postprocess import result_parent_id  # noqa: E402
from app.rag.qa_chain import RagVariant, answer_with_rag, retrieve_for_qa  # noqa: E402
from app.rag.ragas_embeddings import OpenAICompatibleTextEmbeddings  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "ragas_qa_golden.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"
LOW_SCORE_THRESHOLD = 0.85


def ensure_ragas_langchain_compat() -> None:
    """Patch optional LangChain VertexAI imports expected by RAGAS 0.4.x.

    RAGAS imports VertexAI classes to detect multiple-completion support even
    when the project uses OpenAI-compatible judges. Recent langchain-community
    versions no longer expose the old module path, so we provide harmless
    placeholders before importing RAGAS.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules or importlib.util.find_spec(module_name):
        return
    try:
        from langchain_core.language_models import BaseChatModel, BaseLLM
    except Exception:
        return

    module = types.ModuleType(module_name)
    module.ChatVertexAI = type("ChatVertexAI", (BaseChatModel,), {})
    module.VertexAI = type("VertexAI", (BaseLLM,), {})
    sys.modules[module_name] = module


ensure_ragas_langchain_compat()


def load_cases(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def build_dataset(
    cases: list[dict[str, Any]],
    variant: RagVariant,
    answer_source: str = "generated",
    top_k: int = 5,
) -> tuple[list[dict[str, Any]], list[dict]]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for case in cases:
        if answer_source == "reference":
            retrieved = await retrieve_for_qa(
                question=case["question"],
                tags=case.get("tags", []),
                variant=variant,
                top_k=top_k,
            )
            contexts = [_context_from_result_for_eval(item) for item in retrieved]
            answer = case["ground_truth"]
        else:
            result = await answer_with_rag(
                question=case["question"],
                tags=case.get("tags", []),
                variant=variant,
                top_k=top_k,
            )
            retrieved = result.retrieved
            contexts = result.contexts
            answer = result.answer

        retrieved_ids = [result_parent_id(item) for item in retrieved]
        rows.append({
            "user_input": case["question"],
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": case["ground_truth"],
        })
        traces.append({
            "id": case["id"],
            "category": case.get("category", ""),
            "difficulty": case.get("difficulty", ""),
            "question_type": case.get("question_type", ""),
            "tags": case.get("tags", []),
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "expected_source_ids": case.get("expected_source_ids", []),
            "retrieved_source_ids": retrieved_ids,
            "answer": answer,
            "contexts": contexts,
        })

    return rows, traces


def build_ragas_clients(args: argparse.Namespace) -> tuple[Any, Any]:
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    judge_model = args.judge_model or os.getenv("RAGAS_JUDGE_MODEL") or settings.llm_model
    judge_api_key = args.judge_api_key or os.getenv("RAGAS_JUDGE_API_KEY") or settings.llm_api_key
    judge_base_url = args.judge_base_url or os.getenv("RAGAS_JUDGE_BASE_URL") or settings.llm_base_url

    ragas_embedding_model = (
        args.embedding_model
        or os.getenv("RAGAS_EMBEDDING_MODEL")
        or settings.embedding_model
    )
    ragas_embedding_api_key = (
        args.embedding_api_key
        or os.getenv("RAGAS_EMBEDDING_API_KEY")
        or settings.effective_embedding_api_key
    )
    ragas_embedding_base_url = (
        args.embedding_base_url
        or os.getenv("RAGAS_EMBEDDING_BASE_URL")
        or settings.effective_embedding_base_url
    )
    if "vision" in ragas_embedding_model.lower():
        ragas_embedding_model = "text-embedding-v3"
    timeout = float(os.getenv("RAGAS_TIMEOUT_SECONDS", "180"))
    max_retries = int(os.getenv("RAGAS_MAX_RETRIES", "2"))

    llm = ChatOpenAI(
        model=judge_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
        temperature=0,
        max_retries=max_retries,
        timeout=timeout,
    )
    embeddings = OpenAICompatibleTextEmbeddings(
        model=ragas_embedding_model,
        api_key=ragas_embedding_api_key,
        base_url=ragas_embedding_base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return LangchainLLMWrapper(llm, bypass_n=True), LangchainEmbeddingsWrapper(embeddings)


def select_metrics(metric_set: str) -> list[Any]:
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    core4 = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    if metric_set in {"core", "core4", "all"}:
        return core4
    if metric_set == "faithfulness":
        return [faithfulness]
    if metric_set == "retrieval":
        return [context_precision, context_recall]
    if metric_set == "answer":
        return [answer_relevancy]
    raise ValueError(f"Unsupported metric set: {metric_set}")


def summarize_eval(result: Any) -> dict[str, float]:
    """Convert a RAGAS EvaluationResult into a plain metric summary."""
    try:
        items = dict(result).items()
    except Exception:
        items = []

    summary: dict[str, float] = {}
    for key, value in items:
        try:
            summary[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    if summary:
        return summary

    try:
        df = result.to_pandas()
    except Exception:
        return {}

    for column in df.columns:
        if column in {"user_input", "response", "retrieved_contexts", "reference"}:
            continue
        try:
            summary[column] = float(df[column].mean())
        except Exception:
            continue
    return summary


def per_case_scores(result: Any) -> list[dict[str, Any]]:
    try:
        df = result.to_pandas()
    except Exception:
        return []
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    return records


def metric_coverage(rows: list[dict[str, Any]], metrics: list[Any]) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    metric_names = [getattr(metric, "name", str(metric)) for metric in metrics]
    for metric_name in metric_names:
        valid = 0
        missing = 0
        for row in rows:
            value = row.get(metric_name)
            if value is None:
                missing += 1
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                missing += 1
                continue
            if number != number:
                missing += 1
                continue
            valid += 1
        coverage[metric_name] = {"valid": valid, "missing": missing}
    return coverage


async def evaluate_variant(
    variant: RagVariant,
    cases: list[dict[str, Any]],
    answer_source: str,
    metric_set: str,
    batch_size: int,
    top_k: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate

    dataset_rows, traces = await build_dataset(
        cases,
        variant,
        answer_source=answer_source,
        top_k=top_k,
    )
    dataset = Dataset.from_list(dataset_rows)
    ragas_llm, ragas_embeddings = build_ragas_clients(args)
    metrics = select_metrics(metric_set)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
        show_progress=True,
        batch_size=batch_size,
    )

    summary = summarize_eval(result)
    rows = per_case_scores(result)
    diagnostics = diagnose_cases(rows, traces)
    return {
        "variant": variant,
        "summary": summary,
        "metric_coverage": metric_coverage(rows, metrics),
        "traces": traces,
        "ragas_rows": rows,
        "case_diagnostics": diagnostics,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nRAGAS Summary")
    print("-" * 78)
    metrics = sorted({metric for result in results for metric in result["summary"]})
    print(f"{'variant':12s} " + " ".join(f"{metric:>16s}" for metric in metrics))
    for result in results:
        values = " ".join(
            f"{format_score(result['summary'].get(metric)):>16s}"
            for metric in metrics
        )
        print(f"{result['variant']:12s} {values}")
        coverage = result.get("metric_coverage", {})
        incomplete = [
            f"{name}: valid={counts['valid']}, missing={counts['missing']}"
            for name, counts in coverage.items()
            if counts.get("missing", 0)
        ]
        if incomplete:
            print(" " * 13 + "coverage warnings: " + "; ".join(incomplete))


def diagnose_cases(rows: list[dict[str, Any]], traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    trace_by_index = {index: trace for index, trace in enumerate(traces)}

    for index, row in enumerate(rows):
        trace = trace_by_index.get(index, {})
        metric_values = extract_metric_values(row)
        flags: list[str] = []
        suggestions: list[str] = []

        if metric_values.get("faithfulness", 1.0) < LOW_SCORE_THRESHOLD:
            flags.append("unsupported_claims")
            suggestions.append("Tighten grounded-answer prompt or inspect retrieved evidence.")
        if metric_values.get("answer_relevancy", 1.0) < LOW_SCORE_THRESHOLD:
            flags.append("answer_off_topic")
            suggestions.append("Check query rewriting and answer prompt alignment.")
        if metric_values.get("context_precision", 1.0) < LOW_SCORE_THRESHOLD:
            flags.append("context_noise")
            suggestions.append("Reduce top_k, improve reranking, or add metadata filters.")
        if metric_values.get("context_recall", 1.0) < LOW_SCORE_THRESHOLD:
            flags.append("insufficient_context")
            suggestions.append("Increase top_k, improve query variants, or enrich chunk context.")

        expected = set(trace.get("expected_source_ids") or [])
        retrieved = set(trace.get("retrieved_source_ids") or [])
        source_hit = bool(expected.intersection(retrieved)) if expected else None
        if source_hit is False:
            flags.append("retrieval_miss")
            suggestions.append("Inspect source ids, query terms, and retriever variant for this case.")

        diagnostics.append({
            "id": trace.get("id") or row.get("id") or f"case-{index + 1}",
            "category": trace.get("category", ""),
            "difficulty": trace.get("difficulty", ""),
            "question_type": trace.get("question_type", ""),
            "question": trace.get("question") or row.get("user_input", ""),
            "metrics": metric_values,
            "expected_source_ids": trace.get("expected_source_ids", []),
            "retrieved_source_ids": trace.get("retrieved_source_ids", []),
            "source_hit": source_hit,
            "flags": sorted(set(flags)),
            "suggestions": sorted(set(suggestions)),
        })

    return diagnostics


def extract_metric_values(row: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        if name not in row:
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if value == value:
            values[name] = value
    return values


def run_config(args: argparse.Namespace, cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": str(args.dataset),
        "case_count": len(cases),
        "variant": args.variant,
        "answer_source": args.answer_source,
        "metrics": args.metrics,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "run_name": args.run_name,
        "low_score_threshold": LOW_SCORE_THRESHOLD,
        "judge_model": args.judge_model or os.getenv("RAGAS_JUDGE_MODEL") or settings.llm_model,
        "judge_timeout_seconds": float(os.getenv("RAGAS_TIMEOUT_SECONDS", "180")),
        "judge_max_retries": int(os.getenv("RAGAS_MAX_RETRIES", "2")),
        "embedding_model": (
            args.embedding_model
            or os.getenv("RAGAS_EMBEDDING_MODEL")
            or settings.embedding_model
        ),
    }


def write_case_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "variant",
        "id",
        "category",
        "difficulty",
        "question_type",
        "source_hit",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "flags",
        "question",
        "expected_source_ids",
        "retrieved_source_ids",
        "suggestions",
    ]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for item in result.get("case_diagnostics", []):
                metrics = item.get("metrics", {})
                writer.writerow({
                    "variant": result["variant"],
                    "id": item.get("id", ""),
                    "category": item.get("category", ""),
                    "difficulty": item.get("difficulty", ""),
                    "question_type": item.get("question_type", ""),
                    "source_hit": item.get("source_hit", ""),
                    "faithfulness": metrics.get("faithfulness", ""),
                    "answer_relevancy": metrics.get("answer_relevancy", ""),
                    "context_precision": metrics.get("context_precision", ""),
                    "context_recall": metrics.get("context_recall", ""),
                    "flags": "; ".join(item.get("flags", [])),
                    "question": item.get("question", ""),
                    "expected_source_ids": "; ".join(item.get("expected_source_ids", [])),
                    "retrieved_source_ids": "; ".join(item.get("retrieved_source_ids", [])),
                    "suggestions": " | ".join(item.get("suggestions", [])),
                })


def write_summary_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    metrics = sorted({metric for result in results for metric in result["summary"]})
    fieldnames = ["variant", *metrics, "flagged_cases", "total_cases", "missing_metrics"]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            diagnostics = result.get("case_diagnostics", [])
            row = {
                "variant": result["variant"],
                "flagged_cases": sum(1 for item in diagnostics if item.get("flags")),
                "total_cases": len(diagnostics),
            }
            row.update(result["summary"])
            row["missing_metrics"] = "; ".join(missing_metric_labels(result))
            writer.writerow(row)


def write_markdown_report(
    results: list[dict[str, Any]],
    output_path: Path,
    config: dict[str, Any],
    json_path: Path,
) -> None:
    metrics = sorted({metric for result in results for metric in result["summary"]})
    lines = [
        "# RAGAS Experiment Report",
        "",
        "## Run Config",
        "",
        f"- Dataset: `{config['dataset']}`",
        f"- Cases: `{config['case_count']}`",
        f"- Variant: `{config['variant']}`",
        f"- Metrics: `{config['metrics']}`",
        f"- Answer source: `{config['answer_source']}`",
        f"- Top K: `{config['top_k']}`",
        f"- Batch size: `{config['batch_size']}`",
        f"- Low-score threshold: `{config['low_score_threshold']}`",
        f"- Judge timeout seconds: `{config.get('judge_timeout_seconds', '')}`",
        f"- Judge max retries: `{config.get('judge_max_retries', '')}`",
        f"- JSON result: `{json_path.name}`",
        "",
        "## Summary",
        "",
    ]

    lines.append("| Variant | " + " | ".join(metric_label(metric) for metric in metrics) + " | Flagged Cases |")
    lines.append("|---|" + "|".join("---:" for _ in metrics) + "|---:|")
    for result in results:
        diagnostics = result.get("case_diagnostics", [])
        values = [
            format_score(result['summary'].get(metric))
            for metric in metrics
        ]
        flagged = sum(1 for item in diagnostics if item.get("flags"))
        lines.append(f"| {result['variant']} | " + " | ".join(values) + f" | {flagged}/{len(diagnostics)} |")

    lines.extend(["", "## Metric Coverage", ""])
    lines.append("| Variant | Metric | Valid | Missing |")
    lines.append("|---|---|---:|---:|")
    for result in results:
        for name, counts in result.get("metric_coverage", {}).items():
            lines.append(
                f"| {result['variant']} | {metric_label(name)} | "
                f"{counts.get('valid', 0)} | {counts.get('missing', 0)} |"
            )

    missing_notes = [
        (result["variant"], label)
        for result in results
        for label in missing_metric_labels(result)
    ]
    if missing_notes:
        lines.extend([
            "",
            "Metrics shown as `n/a` were not scored by the judge for all requested cases.",
            "This usually indicates provider timeout, rate limit, or a transient judge-model failure.",
            "Aggregate metric means are computed only from valid scored rows; missing values are kept explicit instead of being imputed.",
            "",
            "### Missing Metric Notes",
            "",
            "| Variant | Missing Metric |",
            "|---|---|",
        ])
        for variant, label in missing_notes:
            lines.append(f"| {variant} | {label} |")

    category_rows = category_breakdown(results)
    if category_rows:
        lines.extend(["", "## Category Breakdown", ""])
        lines.append("| Variant | Category | Cases | Faithfulness | Answer Relevancy | Context Precision | Context Recall |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in category_rows:
            lines.append(
                f"| {row['variant']} | {row['category'] or 'uncategorized'} | {row['cases']} | "
                f"{format_optional(row.get('faithfulness'))} | "
                f"{format_optional(row.get('answer_relevancy'))} | "
                f"{format_optional(row.get('context_precision'))} | "
                f"{format_optional(row.get('context_recall'))} |"
            )

    lines.extend(["", "## Problem Cases", ""])
    problem_cases = [
        (result["variant"], item)
        for result in results
        for item in result.get("case_diagnostics", [])
        if item.get("flags")
    ]
    if not problem_cases:
        lines.append("No problem cases were detected by the configured threshold.")
    else:
        lines.append("| Variant | Case | Flags | Lowest Metric | Source Hit | Suggested Action |")
        lines.append("|---|---|---|---:|---|---|")
        for variant, item in sorted(problem_cases, key=lambda pair: lowest_metric(pair[1])[1]):
            metric_name, metric_value = lowest_metric(item)
            suggestions = item.get("suggestions", [])
            lines.append(
                f"| {variant} | {item.get('id', '')} | {', '.join(item.get('flags', []))} | "
                f"{metric_label(metric_name)}={metric_value:.3f} | {item.get('source_hit', '')} | "
                f"{suggestions[0] if suggestions else ''} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summary_chart(results: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    variants = [result["variant"] for result in results]
    if not variants:
        return

    width = 0.18
    x_positions = list(range(len(variants)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, metric in enumerate(metrics):
        offsets = [x + (index - 1.5) * width for x in x_positions]
        values = [result["summary"].get(metric, float("nan")) for result in results]
        if all(value != value for value in values):
            continue
        ax.bar(offsets, values, width=width, label=metric_label(metric))

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("RAGAS Metrics by Variant")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(variants)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def category_breakdown(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        buckets: dict[str, list[dict[str, float]]] = {}
        for item in result.get("case_diagnostics", []):
            buckets.setdefault(item.get("category", ""), []).append(item.get("metrics", {}))
        for category, metric_rows in sorted(buckets.items()):
            row: dict[str, Any] = {
                "variant": result["variant"],
                "category": category,
                "cases": len(metric_rows),
            }
            for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                values = [metrics[metric] for metrics in metric_rows if metric in metrics]
                row[metric] = mean(values) if values else None
            rows.append(row)
    return rows


def lowest_metric(item: dict[str, Any]) -> tuple[str, float]:
    metrics = item.get("metrics", {})
    if not metrics:
        return "", 0.0
    return min(metrics.items(), key=lambda pair: pair[1])


def metric_label(metric: str) -> str:
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }
    return labels.get(metric, metric)


def is_missing_score(value: Any) -> bool:
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return number != number


def format_score(value: Any) -> str:
    if is_missing_score(value):
        return "n/a"
    return f"{float(value):.3f}"


def format_optional(value: Any) -> str:
    if is_missing_score(value):
        return ""
    return f"{float(value):.3f}"


def missing_metric_labels(result: dict[str, Any]) -> list[str]:
    labels = []
    for name, counts in result.get("metric_coverage", {}).items():
        if counts.get("missing", 0):
            labels.append(
                f"{metric_label(name)} ({counts.get('missing', 0)} missing / "
                f"{counts.get('valid', 0)} valid)"
            )
    return labels


def safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return safe.strip("._-") or "ragas_run"


def save_results(results: list[dict[str, Any]], args: argparse.Namespace, cases: list[dict[str, Any]]) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variants = "-".join(result["variant"] for result in results)
    run_name = args.run_name or f"ragas_{variants}"
    output_path = output_dir / f"{safe_filename(run_name)}_{timestamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_config": run_config(args, cases),
        "corpus": {
            "vector_store_documents": get_vector_store().count,
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_case_csv(results, output_path.with_suffix(".cases.csv"))
    write_summary_csv(results, output_path.with_suffix(".summary.csv"))
    report_path = output_path.with_suffix(".md")
    write_markdown_report(results, report_path, payload["run_config"], output_path)
    chart_path = output_path.with_suffix(".png")
    save_summary_chart(results, chart_path)
    return output_path


def save_prepared_dataset(
    variant: str,
    dataset_rows: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ragas_prepared_{variant}"
    output_path = args.output_dir / f"{safe_filename(run_name)}_{timestamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variant": variant,
        "run_config": run_config(args, []),
        "dataset": dataset_rows,
        "traces": traces,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG QA with RAGAS.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=GOLDEN_PATH,
        help="Golden QA dataset path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for JSON, CSV, Markdown, and chart outputs.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Human-readable run name used as the output filename prefix.",
    )
    parser.add_argument(
        "--variant",
        choices=["vector", "hybrid", "multi", "full", "all"],
        default="full",
        help="RAG pipeline variant to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of golden cases to run for a quick smoke test.",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated golden case ids to evaluate, for targeted reruns.",
    )
    parser.add_argument(
        "--answer-source",
        choices=["generated", "reference"],
        default="generated",
        help=(
            "Use generated answers from the project LLM, or use ground-truth references "
            "as answers for a cheaper RAGAS smoke test."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only retrieve contexts and save the prepared RAGAS dataset; do not call RAGAS judges.",
    )
    parser.add_argument(
        "--metrics",
        choices=["core", "core4", "all", "faithfulness", "retrieval", "answer"],
        default="core",
        help=(
            "core/core4 = faithfulness/answer_relevancy/context_precision/context_recall. "
            "all is kept as an alias for the same four main metrics. "
            "faithfulness/retrieval/answer run targeted metric subsets."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Final number of hydrated contexts passed to generation and RAGAS.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="RAGAS evaluation batch size. Smaller values are slower but reduce provider timeouts.",
    )
    parser.add_argument("--judge-model", default="", help="RAGAS judge chat model override.")
    parser.add_argument("--judge-api-key", default="", help="RAGAS judge API key override.")
    parser.add_argument("--judge-base-url", default="", help="RAGAS judge base URL override.")
    parser.add_argument("--embedding-model", default="", help="RAGAS embedding model override.")
    parser.add_argument("--embedding-api-key", default="", help="RAGAS embedding API key override.")
    parser.add_argument("--embedding-base-url", default="", help="RAGAS embedding base URL override.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    if args.case_ids:
        wanted = {case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()}
        cases = [case for case in cases if case.get("id") in wanted]
        if not cases:
            print(f"No cases matched --case-ids={args.case_ids}")
            return

    if get_vector_store().count == 0:
        print("Vector store is empty. Run: python -m scripts.init_vector_store --reset")
        return

    variants: list[RagVariant]
    if args.variant == "all":
        variants = ["vector", "hybrid", "multi", "full"]
    else:
        variants = [args.variant]

    results = []
    for variant in variants:
        print(f"\nEvaluating variant: {variant} ({len(cases)} cases)")
        if args.prepare_only:
            dataset_rows, traces = await build_dataset(
                cases,
                variant,
                answer_source="reference",
                top_k=args.top_k,
            )
            output_path = save_prepared_dataset(variant, dataset_rows, traces, args)
            print(f"Prepared dataset saved to: {output_path}")
            continue
        try:
            results.append(
                await evaluate_variant(
                    variant,
                    cases,
                    args.answer_source,
                    args.metrics,
                    args.batch_size,
                    args.top_k,
                    args,
                )
            )
        except Exception as exc:
            print("\nRAGAS evaluation failed before metrics could be produced.")
            print(f"Reason: {exc}")
            print(
                "Common causes: unavailable chat-completion account, invalid model/base_url, "
                "or provider rate limits. You can still run "
                "`python -m scripts.evaluate_ragas --variant full --prepare-only` "
                "to verify retrieval contexts without calling RAGAS judges."
            )
            raise

    if results:
        print_summary(results)
        output_path = save_results(results, args, cases)
        print(f"\nSaved detailed results to: {output_path}")
        print(f"Saved case CSV to: {output_path.with_suffix('.cases.csv')}")
        print(f"Saved summary CSV to: {output_path.with_suffix('.summary.csv')}")
        print(f"Saved Markdown report to: {output_path.with_suffix('.md')}")
        if output_path.with_suffix(".png").exists():
            print(f"Saved metric chart to: {output_path.with_suffix('.png')}")


def _context_from_result_for_eval(item: Any) -> str:
    return build_retrieval_context(item)


if __name__ == "__main__":
    asyncio.run(main())
