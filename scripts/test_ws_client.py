"""WebSocket test client — validates the full interview flow end-to-end.

Usage:
    1. Start the server:  uvicorn app.main:app --port 8000
    2. Run this script:   python -m scripts.test_ws_client

You can also run it interactively:
    python -m scripts.test_ws_client --interactive
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

BASE_URL = "http://127.0.0.1:8000"
WS_BASE = "ws://127.0.0.1:8000"

SAMPLE_JD = (PROJECT_ROOT / "data" / "sample_jds" / "ai_engineer.txt").read_text(encoding="utf-8")

AUTO_ANSWERS = [
    "Python的GIL是全局解释器锁，它保证同一时刻只有一个线程执行Python字节码。这对CPU密集型任务影响很大，因为多线程无法利用多核。但对IO密集型任务影响较小，因为线程在等待IO时会释放GIL。可以用multiprocessing模块来绕过GIL限制。",
    "Transformer的核心是Self-Attention机制。输入通过Q、K、V三个线性变换，然后计算注意力分数，公式是softmax(QK^T/√dk)乘以V。Multi-Head Attention允许模型关注不同位置。还有位置编码来补充位置信息。",
    "RAG是检索增强生成，流程是先将用户查询用来检索相关文档，然后把文档和查询拼接到prompt中给LLM生成回答。需要RAG是因为LLM有知识截止日期，而且容易产生幻觉。",
    "我了解LangGraph，它和LangChain的区别在于LangGraph支持有环图，可以处理循环场景比如多轮对话和重试。LangChain的Chain是DAG不支持循环。LangGraph通过StateGraph和条件边来实现状态管理。",
    "系统设计中缓存常见问题有缓存穿透、缓存击穿和缓存雪崩。穿透可以用布隆过滤器解决，击穿可以用互斥锁，雪崩可以用随机化过期时间。一致性方面通常用Cache Aside模式。",
    "Agent的核心组件包括LLM、工具调用、记忆和规划能力。ReAct是一种经典范式，交替进行推理和行动。Function Calling让LLM能调用外部工具。",
    "Docker容器和虚拟机的区别是容器共享宿主内核更轻量，虚拟机模拟完整硬件更重。Docker用namespace做隔离，cgroup做资源限制，镜像分层用UnionFS实现。",
    "Prompt Engineering核心技巧包括：明确角色设定、Few-shot示例、Chain-of-Thought引导逐步推理、要求Structured Output输出JSON格式。",
]


async def run_auto_mode():
    """Automated test: create session → connect WS → auto-answer all questions."""
    print(f"\n{'=' * 60}")
    print("  WebSocket E2E Test (Auto Mode)")
    print(f"{'=' * 60}\n")

    # Step 1: Create session via REST
    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        resp = await http.post(
            "/api/interview/start",
            json={"jd_text": SAMPLE_JD, "max_follow_ups": 1},
        )
        resp.raise_for_status()
        data = resp.json()
        session_id = data["session_id"]
        ws_path = data["websocket_url"]
        print(f"Session created: {session_id}")
        print(f"WebSocket URL: {ws_path}\n")

    # Step 2: Connect via WebSocket
    import websockets

    ws_url = f"{WS_BASE}{ws_path}"
    answer_idx = 0

    async with websockets.connect(ws_url) as ws:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "status":
                print(f"  [STATUS] {msg.get('stage')}: {msg.get('message')}")

            elif msg_type == "question":
                q_idx = msg.get("question_index", "?")
                total = msg.get("total_questions", "?")
                diff = msg.get("difficulty", "")
                print(f"\n  [Q{q_idx}/{total}] ({diff}) {msg['content'][:120]}...")

                answer = AUTO_ANSWERS[answer_idx % len(AUTO_ANSWERS)]
                answer_idx += 1
                print(f"  [ANSWER] {answer[:80]}...")
                await ws.send(json.dumps({"type": "answer", "content": answer}))

            elif msg_type == "follow_up":
                fu_num = msg.get("follow_up_number", "?")
                print(f"\n  [FOLLOW-UP #{fu_num}] {msg['content'][:120]}...")

                answer = "补充一下，" + AUTO_ANSWERS[answer_idx % len(AUTO_ANSWERS)]
                answer_idx += 1
                print(f"  [ANSWER] {answer[:80]}...")
                await ws.send(json.dumps({"type": "answer", "content": answer}))

            elif msg_type == "interview_end":
                print(f"\n  [END] {msg.get('message')}")

            elif msg_type == "report":
                report = msg.get("data", {})
                print(f"\n{'=' * 60}")
                print("  INTERVIEW REPORT")
                print(f"{'=' * 60}")
                print(f"  Overall: {report.get('overall_score', '?')}/10  "
                      f"Grade: {report.get('grade', '?')}")
                for ss in report.get("skill_scores", []):
                    score = ss.get("score", 0)
                    bar = "#" * score + "." * (10 - score)
                    print(f"  {ss.get('skill_name', '?'):20s} [{bar}] {score}/10")
                print(f"\n  Strengths:")
                for s in report.get("strengths", []):
                    print(f"    + {s.get('point', '')}")
                print(f"  Improvements:")
                for imp in report.get("improvements", []):
                    print(f"    - {imp.get('point', '')}")
                print(f"\n  Assessment: {report.get('overall_assessment', '')}")
                break

            elif msg_type == "error":
                print(f"  [ERROR] {msg.get('message')}")
                break

    print(f"\n{'=' * 60}")
    print("  Test complete!")
    print(f"{'=' * 60}\n")


async def run_interactive_mode():
    """Interactive mode: you type answers manually."""
    print(f"\n{'=' * 60}")
    print("  WebSocket Interactive Interview")
    print(f"{'=' * 60}\n")

    async with httpx.AsyncClient(base_url=BASE_URL) as http:
        resp = await http.post(
            "/api/interview/start",
            json={"jd_text": SAMPLE_JD, "max_follow_ups": 2},
        )
        resp.raise_for_status()
        data = resp.json()
        session_id = data["session_id"]
        ws_path = data["websocket_url"]
        print(f"Session: {session_id}\n")

    import websockets

    ws_url = f"{WS_BASE}{ws_path}"

    async with websockets.connect(ws_url) as ws:
        # Read messages in background, prompt user for answers
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=180)
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "status":
                print(f"[System] {msg.get('message')}")

            elif msg_type in ("question", "follow_up"):
                print(f"\n[Interviewer] {msg.get('content', '')}\n")
                answer = input("[You]: ").strip()
                if answer.lower() in ("quit", "exit"):
                    await ws.send(json.dumps({"type": "end_interview"}))
                    break
                await ws.send(json.dumps({"type": "answer", "content": answer}))

            elif msg_type == "interview_end":
                print(f"\n[System] {msg.get('message')}")

            elif msg_type == "report":
                report = msg.get("data", {})
                print(f"\n{'=' * 40} REPORT {'=' * 40}")
                print(f"Score: {report.get('overall_score')}/10  Grade: {report.get('grade')}")
                print(f"Assessment: {report.get('overall_assessment')}")
                break

            elif msg_type == "error":
                print(f"[Error] {msg.get('message')}")


if __name__ == "__main__":
    interactive = "--interactive" in sys.argv or "-i" in sys.argv
    if interactive:
        asyncio.run(run_interactive_mode())
    else:
        asyncio.run(run_auto_mode())
