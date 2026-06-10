"""Run the standard local RAGAS experiment suite.

This wrapper keeps experiment commands reproducible. It does not import RAGAS
directly; each experiment is delegated to scripts.evaluate_ragas.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "eval" / "ragas_qa_golden_v2.json"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard RAGAS experiments.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Golden QA dataset path.",
    )
    parser.add_argument(
        "--suite",
        choices=["smoke", "baseline", "variant", "topk", "all"],
        default="smoke",
        help="Experiment suite to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Case limit for smoke runs. Ignored by full suites.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="RAGAS batch size for judge calls.",
    )
    parser.add_argument(
        "--answer-source",
        choices=["generated", "reference"],
        default="generated",
        help="Answer source passed to evaluate_ragas.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare datasets without calling RAGAS judges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    for command in commands:
        print("\n$ " + " ".join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    suites = ["smoke", "baseline", "variant", "topk"] if args.suite == "all" else [args.suite]
    commands: list[list[str]] = []
    for suite in suites:
        if suite == "smoke":
            commands.append(evaluate_command(
                args,
                variant="full",
                run_name="smoke_full_v2",
                limit=args.limit,
            ))
        elif suite == "baseline":
            commands.append(evaluate_command(
                args,
                variant="full",
                run_name="full_v2_baseline",
            ))
        elif suite == "variant":
            commands.append(evaluate_command(
                args,
                variant="all",
                run_name="variant_compare_v2",
            ))
        elif suite == "topk":
            for top_k in (3, 5, 8, 10):
                commands.append(evaluate_command(
                    args,
                    variant="full",
                    run_name=f"full_topk{top_k}",
                    top_k=top_k,
                ))
    return commands


def evaluate_command(
    args: argparse.Namespace,
    *,
    variant: str,
    run_name: str,
    top_k: int = 5,
    limit: int = 0,
) -> list[str]:
    python_executable = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))
    command = [
        python_executable,
        "-m",
        "scripts.evaluate_ragas",
        "--dataset",
        str(args.dataset),
        "--variant",
        variant,
        "--metrics",
        "core",
        "--answer-source",
        args.answer_source,
        "--batch-size",
        str(args.batch_size),
        "--top-k",
        str(top_k),
        "--run-name",
        run_name,
    ]
    if limit:
        command.extend(["--limit", str(limit)])
    if args.prepare_only:
        command.append("--prepare-only")
    return command


if __name__ == "__main__":
    main()
