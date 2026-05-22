"""Evaluation script — tests scoring consistency of the Evaluator Agent.

Runs a set of sample Q&A pairs through the assessment pipeline multiple times,
then reports score mean/std to verify stability.

Usage:
    python -m scripts.evaluate
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.client import get_llm_client
from app.llm.prompts import ANSWER_ASSESSOR_SYSTEM
from app.models.interview import AnswerAssessment

# Sample test cases: (question, reference_points, candidate_answer, expected_range)
TEST_CASES = [
    {
        "id": "eval-001",
        "question": "请解释 Python 中的 GIL 是什么？它对多线程程序有什么影响？",
        "reference_points": [
            "GIL 是 CPython 中的互斥锁",
            "同一时刻只允许一个线程执行 Python 字节码",
            "对 CPU 密集型任务影响大",
            "IO 密集型任务受影响较小",
        ],
        "answer": (
            "GIL 是全局解释器锁，它保证同一时刻只有一个线程执行 Python 字节码。"
            "这对 CPU 密集型任务影响很大，因为多线程无法利用多核。"
            "但对 IO 密集型任务影响较小，因为线程在等待 IO 时会释放 GIL。"
            "可以用 multiprocessing 来绕过。"
        ),
        "expected_min": 7,
        "expected_max": 10,
        "description": "Good answer covering all points",
    },
    {
        "id": "eval-002",
        "question": "请解释 Python 中的 GIL 是什么？它对多线程程序有什么影响？",
        "reference_points": [
            "GIL 是 CPython 中的互斥锁",
            "同一时刻只允许一个线程执行 Python 字节码",
            "对 CPU 密集型任务影响大",
            "IO 密集型任务受影响较小",
        ],
        "answer": "GIL 是一种锁机制，和线程有关。",
        "expected_min": 2,
        "expected_max": 5,
        "description": "Vague answer, minimal content",
    },
    {
        "id": "eval-003",
        "question": "请解释 RAG 的原理和完整流程。",
        "reference_points": [
            "LLM 的知识有截止日期",
            "RAG 流程：检索 → 拼接 → 生成",
            "文档处理：分块、向量化、存入向量数据库",
            "混合检索",
        ],
        "answer": (
            "RAG 就是检索增强生成。先把文档分块，用 embedding 模型向量化后存到向量数据库。"
            "查询时先检索相关文档，把检索到的内容和用户问题拼接到 prompt 里给 LLM 生成回答。"
            "这样可以解决 LLM 知识过时和幻觉的问题。可以用混合检索提高召回率。"
        ),
        "expected_min": 7,
        "expected_max": 10,
        "description": "Comprehensive RAG answer",
    },
    {
        "id": "eval-004",
        "question": "请解释 RAG 的原理和完整流程。",
        "reference_points": [
            "LLM 的知识有截止日期",
            "RAG 流程：检索 → 拼接 → 生成",
            "文档处理：分块、向量化、存入向量数据库",
            "混合检索",
        ],
        "answer": "RAG 是一种让 AI 回答更准确的技术，用到了数据库。",
        "expected_min": 2,
        "expected_max": 5,
        "description": "Superficial answer",
    },
]

RUNS_PER_CASE = 3


async def assess_once(case: dict) -> AnswerAssessment:
    """Run a single assessment."""
    llm = get_llm_client()
    system_prompt = ANSWER_ASSESSOR_SYSTEM.format(max_follow_ups=2)

    user_content = (
        f"Question:\n{case['question']}\n\n"
        f"Reference answer key points:\n"
        + "\n".join(f"- {p}" for p in case["reference_points"])
        + f"\n\nCandidate's answer:\n{case['answer']}\n\n"
        f"Current follow-up count for this question: 0/2\n\n"
        f"Produce a structured assessment."
    )

    return await llm.chat_structured(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_model=AnswerAssessment,
        temperature=0.2,
    )


async def main():
    print(f"\n{'=' * 65}")
    print("  Scoring Consistency Evaluation")
    print(f"  {len(TEST_CASES)} test cases × {RUNS_PER_CASE} runs each")
    print(f"{'=' * 65}\n")

    all_passed = True

    for case in TEST_CASES:
        scores: list[int] = []
        print(f"[{case['id']}] {case['description']}")
        print(f"  Expected range: {case['expected_min']}-{case['expected_max']}")

        for run in range(RUNS_PER_CASE):
            try:
                result = await assess_once(case)
                scores.append(result.score)
                print(f"  Run {run+1}: score={result.score}, "
                      f"follow_up={result.should_follow_up}, "
                      f"covered={len(result.covered_points)}, "
                      f"missed={len(result.missed_points)}")
            except Exception as e:
                print(f"  Run {run+1}: FAILED — {e}")

        if len(scores) >= 2:
            mean = statistics.mean(scores)
            stdev = statistics.stdev(scores) if len(scores) > 1 else 0
            in_range = all(case["expected_min"] <= s <= case["expected_max"] for s in scores)
            status = "PASS ✓" if in_range and stdev <= 2.0 else "WARN ⚠"
            if not in_range or stdev > 2.0:
                all_passed = False

            print(f"  → Mean={mean:.1f}, Std={stdev:.1f}, "
                  f"InRange={'Yes' if in_range else 'NO'} | {status}")
        elif scores:
            s = scores[0]
            in_range = case["expected_min"] <= s <= case["expected_max"]
            print(f"  → Score={s}, InRange={'Yes' if in_range else 'NO'}")
        else:
            print("  → No successful runs (check API key)")
            all_passed = False

        print()

    print(f"{'=' * 65}")
    print(f"  Overall: {'ALL PASSED ✓' if all_passed else 'SOME WARNINGS ⚠'}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    asyncio.run(main())
