"""Fast local variant comparison for RAG retrieval.

This script does not call RAGAS judges. It compares whether each RAG variant
retrieves the expected parent sources from the golden QA dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.postprocess import result_parent_id  # noqa: E402
from app.rag.qa_chain import RagVariant, retrieve_for_qa  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "ragas_qa_golden_v2.json"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"
VARIANTS: list[RagVariant] = ["vector", "hybrid", "multi", "full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare RAG variants without judge calls.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run-name", default="variant_source_hit")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    if get_vector_store().count == 0:
        print("Vector store is empty. Run: python -m scripts.init_vector_store --reset")
        return

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for case in cases:
            retrieved = await retrieve_for_qa(
                question=case["question"],
                tags=case.get("tags", []),
                variant=variant,
                top_k=args.top_k,
            )
            retrieved_ids = unique_ids(result_parent_id(item) for item in retrieved)
            expected_ids = [str(item) for item in case.get("expected_source_ids", [])]
            expected = set(expected_ids)
            retrieved_set = set(retrieved_ids)
            overlap = sorted(expected.intersection(retrieved_set))
            rows.append({
                "variant": variant,
                "id": case["id"],
                "category": case.get("category", ""),
                "question": case["question"],
                "expected_source_ids": expected_ids,
                "retrieved_source_ids": retrieved_ids,
                "source_hit": bool(overlap) if expected else None,
                "expected_recall": len(overlap) / max(len(expected), 1),
                "retrieved_count": len(retrieved_ids),
            })

    summary = summarize(rows)
    output_base = save_outputs(args, summary, rows)
    print_summary(summary)
    print(f"\nSaved variant summary to: {output_base.with_suffix('.summary.csv')}")
    print(f"Saved variant cases to: {output_base.with_suffix('.cases.csv')}")
    print(f"Saved variant report to: {output_base.with_suffix('.md')}")


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_ids(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = str(value)
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            continue
        summary.append({
            "variant": variant,
            "cases": len(variant_rows),
            "hit_rate": mean(1.0 if row["source_hit"] else 0.0 for row in variant_rows),
            "expected_recall": mean(float(row["expected_recall"]) for row in variant_rows),
            "avg_retrieved_count": mean(float(row["retrieved_count"]) for row in variant_rows),
            "missed_cases": sum(1 for row in variant_rows if not row["source_hit"]),
        })
    return summary


def save_outputs(
    args: argparse.Namespace,
    summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir / f"{safe_filename(args.run_name)}_{timestamp}"
    write_summary_csv(summary, output_base.with_suffix(".summary.csv"))
    write_cases_csv(rows, output_base.with_suffix(".cases.csv"))
    write_markdown(args, summary, rows, output_base.with_suffix(".md"))
    return output_base


def write_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["variant", "cases", "hit_rate", "expected_recall", "avg_retrieved_count", "missed_cases"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def write_cases_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "variant",
        "id",
        "category",
        "source_hit",
        "expected_recall",
        "retrieved_count",
        "expected_source_ids",
        "retrieved_source_ids",
        "question",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "expected_source_ids": "; ".join(row["expected_source_ids"]),
                "retrieved_source_ids": "; ".join(row["retrieved_source_ids"]),
            })


def write_markdown(
    args: argparse.Namespace,
    summary: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# RAG Variant Source-Hit Report",
        "",
        "## Config",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Top K: `{args.top_k}`",
        f"- Cases per variant: `{summary[0]['cases'] if summary else 0}`",
        "",
        "## Summary",
        "",
        "| Variant | Cases | Hit Rate | Expected Recall | Avg Retrieved Parents | Missed Cases |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['cases']} | {row['hit_rate']:.3f} | "
            f"{row['expected_recall']:.3f} | {row['avg_retrieved_count']:.2f} | "
            f"{row['missed_cases']} |"
        )

    missed = [row for row in rows if not row["source_hit"]]
    lines.extend(["", "## Missed Cases", ""])
    if not missed:
        lines.append("No missed cases.")
    else:
        lines.append("| Variant | Case | Expected | Retrieved |")
        lines.append("|---|---|---|---|")
        for row in missed:
            lines.append(
                f"| {row['variant']} | {row['id']} | "
                f"{'; '.join(row['expected_source_ids'])} | "
                f"{'; '.join(row['retrieved_source_ids'])} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: list[dict[str, Any]]) -> None:
    print("\nVariant Source-Hit Summary")
    print("-" * 76)
    print(f"{'variant':10s} {'hit_rate':>10s} {'expected_recall':>16s} {'avg_parents':>12s} {'missed':>8s}")
    for row in summary:
        print(
            f"{row['variant']:10s} {row['hit_rate']:10.3f} "
            f"{row['expected_recall']:16.3f} {row['avg_retrieved_count']:12.2f} "
            f"{row['missed_cases']:8d}"
        )


def safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return safe.strip("._-") or "variant_source_hit"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_async())
