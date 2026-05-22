"""RAGAS end-to-end evaluation for the RAG QA subtask.

Usage:
    python -m scripts.evaluate_ragas --variant full
    python -m scripts.evaluate_ragas --variant all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)

from datasets import Dataset  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings  # noqa: E402
from app.rag.context_builder import build_retrieval_context  # noqa: E402
from app.rag.postprocess import result_parent_id  # noqa: E402
from app.rag.qa_chain import RagVariant, answer_with_rag, retrieve_for_qa  # noqa: E402
from app.rag.ragas_embeddings import OpenAICompatibleTextEmbeddings  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

GOLDEN_PATH = PROJECT_ROOT / "data" / "eval" / "ragas_qa_golden.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"


def load_cases(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def build_dataset(
    cases: list[dict[str, Any]],
    variant: RagVariant,
    answer_source: str = "generated",
) -> tuple[Dataset, list[dict]]:
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for case in cases:
        if answer_source == "reference":
            retrieved = await retrieve_for_qa(
                question=case["question"],
                tags=case.get("tags", []),
                variant=variant,
                top_k=5,
            )
            contexts = [_context_from_result_for_eval(item) for item in retrieved]
            answer = case["ground_truth"]
        else:
            result = await answer_with_rag(
                question=case["question"],
                tags=case.get("tags", []),
                variant=variant,
                top_k=5,
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
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "expected_source_ids": case.get("expected_source_ids", []),
            "retrieved_source_ids": retrieved_ids,
            "answer": answer,
            "contexts": contexts,
        })

    return Dataset.from_list(rows), traces


def build_ragas_clients() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    ragas_embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL") or settings.embedding_model
    if "vision" in ragas_embedding_model.lower():
        ragas_embedding_model = "text-embedding-v3"
    timeout = float(os.getenv("RAGAS_TIMEOUT_SECONDS", "180"))
    max_retries = int(os.getenv("RAGAS_MAX_RETRIES", "2"))

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        max_retries=max_retries,
        timeout=timeout,
    )
    embeddings = OpenAICompatibleTextEmbeddings(
        model=ragas_embedding_model,
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    return LangchainLLMWrapper(llm, bypass_n=True), LangchainEmbeddingsWrapper(embeddings)


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
) -> dict[str, Any]:
    dataset, traces = await build_dataset(cases, variant, answer_source=answer_source)
    ragas_llm, ragas_embeddings = build_ragas_clients()
    metrics = [
        faithfulness,
        context_precision,
        context_recall,
    ]
    if metric_set == "all":
        metrics.insert(1, answer_relevancy)
    elif metric_set == "faithfulness":
        metrics = [faithfulness]
    elif metric_set == "retrieval":
        metrics = [context_precision, context_recall]
    elif metric_set == "answer":
        metrics = [answer_relevancy]

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
    return {
        "variant": variant,
        "summary": summary,
        "metric_coverage": metric_coverage(rows, metrics),
        "traces": traces,
        "ragas_rows": rows,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\nRAGAS Summary")
    print("-" * 78)
    metrics = sorted({metric for result in results for metric in result["summary"]})
    print(f"{'variant':12s} " + " ".join(f"{metric:>16s}" for metric in metrics))
    for result in results:
        values = " ".join(
            f"{result['summary'].get(metric, float('nan')):16.3f}"
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


def save_results(results: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variants = "-".join(result["variant"] for result in results)
    output_path = RESULTS_DIR / f"ragas_{variants}_{timestamp}.json"
    payload = {
        "created_at": timestamp,
        "corpus": {
            "vector_store_documents": get_vector_store().count,
        },
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def save_prepared_dataset(variant: str, dataset: Dataset, traces: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"ragas_prepared_{variant}_{timestamp}.json"
    payload = {
        "created_at": timestamp,
        "variant": variant,
        "dataset": dataset.to_list(),
        "traces": traces,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG QA with RAGAS.")
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
        choices=["core", "all", "faithfulness", "retrieval", "answer"],
        default="core",
        help=(
            "core = faithfulness/context_precision/context_recall. "
            "all also includes answer_relevancy. "
            "faithfulness/retrieval/answer run targeted metric subsets."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="RAGAS evaluation batch size. Smaller values are slower but reduce provider timeouts.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = load_cases()
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
            dataset, traces = await build_dataset(
                cases,
                variant,
                answer_source="reference",
            )
            output_path = save_prepared_dataset(variant, dataset, traces)
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
        output_path = save_results(results)
        print(f"\nSaved detailed results to: {output_path}")


def _context_from_result_for_eval(item: Any) -> str:
    return build_retrieval_context(item)


if __name__ == "__main__":
    asyncio.run(main())
