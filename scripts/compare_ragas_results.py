"""Compare multiple saved RAGAS result JSON files.

Example:
    python -m scripts.compare_ragas_results \
        data/eval/results/topk3.json data/eval/results/topk5.json \
        --labels top_k=3,top_k=5 --run-name topk_compare
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved RAGAS result files.")
    parser.add_argument("results", nargs="+", type=Path, help="RAGAS JSON result files.")
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels matching result files. Defaults to run names.",
    )
    parser.add_argument(
        "--run-name",
        default="ragas_compare",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for comparison CSV, Markdown, and PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = parse_labels(args)
    rows = []
    case_rows = []
    for label, path in zip(labels, args.results):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(summary_rows(label, path, data))
        case_rows.extend(problem_case_rows(label, data))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir / f"{safe_filename(args.run_name)}_{timestamp}"
    write_summary_csv(rows, output_base.with_suffix(".summary.csv"))
    write_cases_csv(case_rows, output_base.with_suffix(".cases.csv"))
    write_markdown(rows, case_rows, output_base.with_suffix(".md"))
    write_chart(rows, output_base.with_suffix(".png"))
    print(f"Saved comparison summary to: {output_base.with_suffix('.summary.csv')}")
    print(f"Saved comparison cases to: {output_base.with_suffix('.cases.csv')}")
    print(f"Saved comparison report to: {output_base.with_suffix('.md')}")
    if output_base.with_suffix(".png").exists():
        print(f"Saved comparison chart to: {output_base.with_suffix('.png')}")


def parse_labels(args: argparse.Namespace) -> list[str]:
    if args.labels:
        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        if len(labels) != len(args.results):
            raise SystemExit("--labels count must match result file count")
        return labels
    return [path.stem for path in args.results]


def summary_rows(label: str, path: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    config = data.get("run_config", {})
    rows = []
    for result in data.get("results", []):
        diagnostics = result.get("case_diagnostics", [])
        row = {
            "label": label,
            "file": path.name,
            "variant": result.get("variant", ""),
            "top_k": config.get("top_k", ""),
            "metrics": config.get("metrics", ""),
            "answer_source": config.get("answer_source", ""),
            "case_count": len(diagnostics) or config.get("case_count", ""),
            "flagged_cases": sum(1 for item in diagnostics if item.get("flags")),
        }
        row.update({metric: result.get("summary", {}).get(metric, "") for metric in METRICS})
        rows.append(row)
    return rows


def problem_case_rows(label: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in data.get("results", []):
        variant = result.get("variant", "")
        for item in result.get("case_diagnostics", []):
            if not item.get("flags"):
                continue
            metrics = item.get("metrics", {})
            rows.append({
                "label": label,
                "variant": variant,
                "id": item.get("id", ""),
                "category": item.get("category", ""),
                "source_hit": item.get("source_hit", ""),
                "lowest_metric": lowest_metric(metrics),
                "flags": "; ".join(item.get("flags", [])),
                "question": item.get("question", ""),
                "suggestions": " | ".join(item.get("suggestions", [])),
            })
    return rows


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "label",
        "file",
        "variant",
        "top_k",
        "metrics",
        "answer_source",
        "case_count",
        "flagged_cases",
        *METRICS,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cases_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "label",
        "variant",
        "id",
        "category",
        "source_hit",
        "lowest_metric",
        "flags",
        "question",
        "suggestions",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# RAGAS Comparison Report",
        "",
        "## Summary",
        "",
        "| Label | Variant | Top K | Cases | Flagged | Faithfulness | Answer Relevancy | Context Precision | Context Recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['variant']} | {row['top_k']} | "
            f"{row['case_count']} | {row['flagged_cases']} | "
            f"{fmt(row.get('faithfulness'))} | {fmt(row.get('answer_relevancy'))} | "
            f"{fmt(row.get('context_precision'))} | {fmt(row.get('context_recall'))} |"
        )

    lines.extend(["", "## Problem Cases", ""])
    if not case_rows:
        lines.append("No flagged cases.")
    else:
        lines.append("| Label | Case | Flags | Lowest Metric | Source Hit | Suggestion |")
        lines.append("|---|---|---|---|---|---|")
        for row in case_rows:
            suggestion = str(row.get("suggestions", "")).split(" | ")[0]
            lines.append(
                f"| {row['label']} | {row['id']} | {row['flags']} | "
                f"{row['lowest_metric']} | {row['source_hit']} | {suggestion} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chart(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [str(row["label"]) for row in rows]
    if not labels:
        return
    metrics = [metric for metric in METRICS if any(row.get(metric) != "" for row in rows)]
    x_positions = list(range(len(labels)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, metric in enumerate(metrics):
        values = [float(row.get(metric) or "nan") for row in rows]
        offsets = [x + (index - (len(metrics) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=metric.replace("_", " ").title())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("RAGAS Result Comparison")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def lowest_metric(metrics: dict[str, Any]) -> str:
    values = []
    for key, value in metrics.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        values.append((key, number))
    if not values:
        return ""
    key, value = min(values, key=lambda pair: pair[1])
    return f"{key}={value:.3f}"


def fmt(value: Any) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return safe.strip("._-") or "ragas_compare"


if __name__ == "__main__":
    main()
