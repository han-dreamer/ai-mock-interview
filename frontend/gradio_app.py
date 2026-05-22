"""Gradio frontend for the AI Mock Interview system.

Usage:
    python -m frontend.gradio_app

Supports two modes:
  - Practice:     Quick practice with reference answers for learning
  - Professional: Full interview simulation with resume analysis and detailed evaluation

Calls the FastAPI REST API. Start the backend with:
    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import gradio as gr
import httpx

from app.models.interview import ChatMessage
from frontend.charts import generate_radar_chart
from frontend.export import export_interview_markdown, save_export

logger = logging.getLogger("gradio_app")
API_BASE_URL = os.getenv("INTERVIEW_API_BASE_URL", "http://127.0.0.1:8000/api")

SAMPLE_JD = (PROJECT_ROOT / "data" / "sample_jds" / "ai_engineer.txt").read_text(encoding="utf-8")

# ── Custom CSS ──────────────────────────────────────────────────────

CUSTOM_CSS = """
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.main-header h1 {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: #666;
    font-size: 0.95rem;
}
.report-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 16px;
    padding: 24px;
    margin: 8px 0;
}
.score-bar {
    height: 12px;
    border-radius: 6px;
    background: #e0e0e0;
    overflow: hidden;
    margin: 4px 0 8px;
}
.score-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
}
.grade-badge {
    display: inline-block;
    font-size: 2.2rem;
    font-weight: 800;
    width: 70px;
    height: 70px;
    line-height: 70px;
    text-align: center;
    border-radius: 50%;
    color: white;
}
.grade-A { background: linear-gradient(135deg, #43e97b, #38f9d7); }
.grade-B { background: linear-gradient(135deg, #667eea, #764ba2); }
.grade-C { background: linear-gradient(135deg, #f093fb, #f5576c); }
.grade-D { background: linear-gradient(135deg, #ff6b6b, #ee5a24); }
.missed-card {
    background: #fff;
    border-left: 4px solid #667eea;
    border-radius: 8px;
    padding: 16px;
    margin: 10px 0;
}
.missed-card h4 { margin: 0 0 8px; color: #333; }
.missed-card .ref-answer {
    background: #f8f9ff;
    padding: 12px;
    border-radius: 6px;
    margin-top: 8px;
    line-height: 1.6;
    color: #444;
}
.study-tip {
    padding: 8px 16px;
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    margin: 6px 0;
}
.resume-info {
    background: #e8f5e9;
    border-left: 4px solid #4caf50;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 12px 0;
}
.feature-card {
    background: white;
    border-radius: 10px;
    padding: 14px;
    border: 1px solid #e8e8e8;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.feature-card .icon { font-size: 1.6rem; margin-bottom: 6px; }
.feature-card .title { font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; }
.feature-card .desc { font-size: 0.78rem; color: #888; }
.export-success {
    background: #e8f5e9;
    border: 1px solid #4caf50;
    border-radius: 8px;
    padding: 10px 16px;
    color: #2e7d32;
    font-size: 0.9rem;
}
"""


# ── Report rendering helpers ────────────────────────────────────────


def _bar_color(score: int) -> str:
    if score >= 8:
        return "linear-gradient(90deg, #43e97b, #38f9d7)"
    if score >= 6:
        return "linear-gradient(90deg, #667eea, #764ba2)"
    if score >= 4:
        return "linear-gradient(90deg, #f093fb, #f5576c)"
    return "linear-gradient(90deg, #ff6b6b, #ee5a24)"


def _render_skill_scores(skill_scores: list[dict]) -> str:
    html = ""
    for ss in skill_scores:
        score = ss.get("score", 0)
        name = ss.get("skill_name", "?")
        evidence = ss.get("evidence", "")
        pct = score * 10
        html += f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; font-weight:600;">
                <span>{name}</span><span>{score}/10</span>
            </div>
            <div class="score-bar">
                <div class="score-fill" style="width:{pct}%; background:{_bar_color(score)};"></div>
            </div>
            <div style="font-size:0.82rem; color:#777;">{evidence[:150]}</div>
        </div>
        """
    return html


def _render_radar_from_skills(skill_scores: list[dict], title: str = "", color: str = "#667eea") -> str:
    """Build a radar chart from skill_scores list."""
    if len(skill_scores) < 3:
        return ""
    labels = [ss.get("skill_name", "?") for ss in skill_scores]
    scores = [ss.get("score", 0) for ss in skill_scores]
    return generate_radar_chart(labels, scores, title=title, color=color)


def _get_last_interviewer_message(state: dict) -> str | None:
    history = state.get("conversation_history", [])
    for msg in reversed(history):
        if isinstance(msg, ChatMessage) and msg.role == "interviewer":
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "interviewer":
            return msg.get("content", "")
    return None


def _format_professional_report(report) -> str:
    """Fallback renderer for single-round InterviewReport."""
    data = report.model_dump() if hasattr(report, "model_dump") else report

    grade = data.get("grade", "?")
    overall = data.get("overall_score", 0)
    grade_class = f"grade-{grade}" if grade in "ABCD" else "grade-D"
    skill_scores = data.get("skill_scores", [])
    radar_html = _render_radar_from_skills(skill_scores)

    html = f"""
    <div class="report-card">
        <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
            <div class="grade-badge {grade_class}">{grade}</div>
            <div>
                <div style="font-size:1.4rem; font-weight:700;">综合评分: {overall:.1f}/10</div>
                <div style="color:#666; margin-top:4px;">{data.get('overall_assessment', '')}</div>
            </div>
        </div>
    """
    if radar_html:
        html += f'<div style="text-align:center; margin:16px 0;">{radar_html}</div>'

    html += f"""
        <h3 style="margin:16px 0 8px;">📊 各项技能评分</h3>
        {_render_skill_scores(skill_scores)}
        <h3 style="margin:20px 0 8px;">✅ 优势</h3><ul>
    """
    for s in data.get("strengths", []):
        html += f'<li><strong>{s.get("point", "")}</strong><br/><span style="color:#666;font-size:0.85rem;">{s.get("evidence", "")[:120]}</span></li>'
    html += '</ul><h3 style="margin:20px 0 8px;">🎯 改进建议</h3><ul>'
    for imp in data.get("improvements", []):
        html += f'<li><strong>{imp.get("point", "")}</strong><br/><span style="color:#666;font-size:0.85rem;">{imp.get("evidence", "")[:120]}</span></li>'
    html += "</ul></div>"
    return html


def _format_dual_round_report(report) -> str:
    """Render the dual-round ProfessionalReport with per-round breakdown."""
    data = report.model_dump() if hasattr(report, "model_dump") else report

    grade = data.get("grade", "?")
    overall = data.get("overall_score", 0)
    depth = data.get("technical_depth_score", 0)
    breadth = data.get("technical_breadth_score", 0)
    rec = data.get("hiring_recommendation", "")
    grade_class = f"grade-{grade}" if grade in "ABCD" else "grade-D"

    html = f"""
    <div class="report-card">
        <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
            <div class="grade-badge {grade_class}">{grade}</div>
            <div>
                <div style="font-size:1.4rem; font-weight:700;">综合评分: {overall:.1f}/10</div>
                <div style="color:#555; margin-top:4px; font-weight:600;">{rec}</div>
                <div style="color:#666; margin-top:4px;">{data.get('overall_assessment', '')}</div>
            </div>
        </div>

        <h3 style="margin:16px 0 8px;">📋 双轮得分</h3>
        <div style="display:flex; gap:16px; margin-bottom:16px;">
    """

    round_scores = data.get("round_scores", [])
    for rs in round_scores:
        rs_score = rs.get("score", 0)
        rs_grade = rs.get("grade", "?")
        rs_name = rs.get("round_name", "?")
        rs_summary = rs.get("summary", "")
        rs_gc = f"grade-{rs_grade}" if rs_grade in "ABCD" else "grade-D"
        html += f"""
            <div style="flex:1; background:white; border-radius:12px; padding:16px; text-align:center;">
                <div style="font-weight:700; margin-bottom:8px;">{rs_name}</div>
                <div class="grade-badge {rs_gc}" style="font-size:1.5rem; width:50px; height:50px; line-height:50px; margin:0 auto 8px;">{rs_grade}</div>
                <div style="font-size:1.2rem; font-weight:600;">{rs_score:.1f}/10</div>
                <div style="font-size:0.85rem; color:#666; margin-top:6px;">{rs_summary}</div>
            </div>
        """

    html += f"""
        </div>

        <div style="display:flex; gap:16px; margin-bottom:16px;">
            <div style="flex:1; background:white; border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:0.85rem; color:#888;">技术深度</div>
                <div style="font-size:1.3rem; font-weight:700; color:#667eea;">{depth:.1f}</div>
            </div>
            <div style="flex:1; background:white; border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:0.85rem; color:#888;">技术广度</div>
                <div style="font-size:1.3rem; font-weight:700; color:#764ba2;">{breadth:.1f}</div>
            </div>
            <div style="flex:1; background:white; border-radius:10px; padding:12px; text-align:center;">
                <div style="font-size:0.85rem; color:#888;">综合评分</div>
                <div style="font-size:1.3rem; font-weight:700; color:#43e97b;">{overall:.1f}</div>
            </div>
        </div>

        <h3 style="margin:16px 0 8px;">📊 各项技能评分</h3>
    """

    skill_scores = data.get("skill_scores", [])
    radar_html = _render_radar_from_skills(skill_scores, color="#764ba2")
    if radar_html:
        html += f'<div style="text-align:center; margin:12px 0;">{radar_html}</div>'

    html += f"""
        {_render_skill_scores(skill_scores)}
        <h3 style="margin:20px 0 8px;">✅ 优势</h3><ul>
    """
    for s in data.get("strengths", []):
        html += f'<li><strong>{s.get("point", "")}</strong><br/><span style="color:#666;font-size:0.85rem;">{s.get("evidence", "")[:150]}</span></li>'
    html += '</ul><h3 style="margin:20px 0 8px;">🎯 改进建议</h3><ul>'
    for imp in data.get("improvements", []):
        html += f'<li><strong>{imp.get("point", "")}</strong><br/><span style="color:#666;font-size:0.85rem;">{imp.get("evidence", "")[:150]}</span></li>'
    html += "</ul></div>"
    return html


def _format_practice_report(report) -> str:
    data = report.model_dump() if hasattr(report, "model_dump") else report

    grade = data.get("grade", "?")
    overall = data.get("overall_score", 0)
    total_q = data.get("total_questions", 0)
    grade_class = f"grade-{grade}" if grade in "ABCD" else "grade-D"

    html = f"""
    <div class="report-card">
        <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
            <div class="grade-badge {grade_class}">{grade}</div>
            <div>
                <div style="font-size:1.4rem; font-weight:700;">练习评分: {overall:.1f}/10</div>
                <div style="color:#666; margin-top:4px;">共回答 {total_q} 题 — {data.get('overall_assessment', '')}</div>
            </div>
        </div>

        <h3 style="margin:16px 0 8px;">📊 各项技能评分</h3>
    """
    skill_scores = data.get("skill_scores", [])
    radar_html = _render_radar_from_skills(skill_scores, color="#43e97b")
    if radar_html:
        html += f'<div style="text-align:center; margin:12px 0;">{radar_html}</div>'

    html += f"""
        {_render_skill_scores(skill_scores)}
    """

    missed = data.get("missed_knowledge", [])
    if missed:
        html += '<h3 style="margin:20px 0 8px;">📚 知识点查漏补缺</h3>'
        for i, mk in enumerate(missed, 1):
            score = mk.get("score", 0)
            question = mk.get("question", "")
            missed_pts = mk.get("missed_points", [])
            ref_answer = mk.get("reference_answer", "")

            missed_tags = " ".join(
                f'<span style="background:#ffe0e0;padding:2px 8px;border-radius:10px;'
                f'font-size:0.8rem;margin-right:4px;">❌ {p}</span>'
                for p in missed_pts[:5]
            )

            html += f"""
            <div class="missed-card">
                <h4>Q{i}: {question} <span style="color:{_bar_color(score).split(',')[0].split('(')[1]};
                    font-size:0.9rem;">({score}/10)</span></h4>
                <div style="margin:6px 0;">{missed_tags}</div>
                <div class="ref-answer">
                    <strong>📖 参考答案：</strong><br/>{ref_answer}
                </div>
            </div>
            """

    suggestions = data.get("study_suggestions", [])
    if suggestions:
        html += '<h3 style="margin:20px 0 8px;">💡 学习建议</h3>'
        for tip in suggestions:
            html += f'<div class="study-tip">📌 {tip}</div>'

    html += "</div>"
    return html


# ── Core interview logic ────────────────────────────────────────────


class InterviewApiClient:
    """Thin async client for the FastAPI interview API."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        timeout = httpx.Timeout(180.0, connect=10.0)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
            response = await client.request(method, path, **kwargs)
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise RuntimeError(f"API {response.status_code}: {detail}")
        return response.json()

    async def create_session(
        self,
        jd_text: str,
        max_follow_ups: int,
        mode: str,
        user_id: str = "local-user",
    ) -> dict:
        return await self._request(
            "POST",
            "/interview/start",
            json={
                "jd_text": jd_text,
                "max_follow_ups": max_follow_ups,
                "mode": mode,
                "user_id": user_id,
            },
        )

    async def create_session_with_resume(
        self,
        jd_text: str,
        max_follow_ups: int,
        mode: str,
        resume_file: str,
        user_id: str = "local-user",
    ) -> dict:
        path = Path(resume_file)
        data = {
            "jd_text": jd_text,
            "max_follow_ups": str(max_follow_ups),
            "mode": mode,
            "user_id": user_id,
        }
        with path.open("rb") as file_obj:
            files = {
                "resume_file": (
                    path.name,
                    file_obj,
                    "application/octet-stream",
                )
            }
            return await self._request(
                "POST",
                "/interview/start-with-resume",
                data=data,
                files=files,
            )

    async def start_session(self, session_id: str) -> dict:
        return await self._request("POST", f"/interview/session/{session_id}/start")

    async def submit_answer(self, session_id: str, answer: str) -> dict:
        return await self._request(
            "POST",
            f"/interview/session/{session_id}/answer",
            json={"content": answer},
        )

    async def stop_session(self, session_id: str) -> dict:
        return await self._request("POST", f"/interview/session/{session_id}/stop")


class InterviewApp:
    """Wraps the FastAPI API to provide a Gradio-friendly interface."""

    def __init__(self):
        self.api = InterviewApiClient()
        self.session_id: str | None = None
        self.is_running = False
        self.mode: str = "practice"
        self.last_report: object | None = None
        self.jd_text: str = ""
        self.resume_attached = False

    def _state_from_payload(self, payload: dict) -> dict:
        """Convert REST payload into the legacy UI state shape."""
        state = dict(payload.get("state") or {})
        next_turn = payload.get("next")
        if next_turn:
            total = next_turn.get("total_questions") or state.get("question_count") or 0
            state["question_plan"] = [{} for _ in range(int(total or 0))]
            state["current_question_index"] = max(
                int(next_turn.get("question_index", 1) or 1) - 1,
                0,
            )
            state["follow_up_count"] = int(next_turn.get("follow_up_number", 0) or 0)
            state["conversation_history"] = [
                {"role": "interviewer", "content": next_turn.get("content", "")}
            ]

        report = payload.get("report")
        if report:
            if self.mode == "practice":
                state["practice_report"] = report
            else:
                state["professional_report"] = report
        return state

    def _next_turn_from_payload(self, payload: dict) -> dict | None:
        return payload.get("next")

    async def start_interview(
        self,
        jd_text: str,
        max_follow_ups: int,
        mode: str,
        resume_file: str | None,
    ):
        """Generator-based start that yields progress updates to the UI."""
        if not jd_text or len(jd_text.strip()) < 10:
            yield [], "⚠️ 请输入至少10个字符的职位描述", "", gr.update(interactive=False)
            return

        if mode == "professional":
            if not resume_file:
                yield [], "⚠️ 专业面试模式需要上传简历文件 (PDF/PNG)", "", gr.update(interactive=False)
                return

            yield [], "📄 [1/4] 正在通过 FastAPI 上传并解析简历...", "", gr.update(interactive=False)
        else:
            yield [], "🔍 [1/3] 正在通过 FastAPI 创建面试会话...", "", gr.update(interactive=False)

        self.mode = mode
        self.jd_text = jd_text.strip()
        self.last_report = None
        self.resume_attached = False
        if mode == "professional":
            created = await self.api.create_session_with_resume(
                jd_text=jd_text.strip(),
                max_follow_ups=int(max_follow_ups),
                mode=mode,
                resume_file=resume_file,
            )
            self.resume_attached = True
            yield [], "🔍 [2/4] 简历已接入，正在启动 Agent 工作流...", "", gr.update(interactive=False)
        else:
            created = await self.api.create_session(
                jd_text=jd_text.strip(),
                max_follow_ups=int(max_follow_ups),
                mode=mode,
            )
        self.session_id = created["session_id"]

        step = "[3/4]" if mode == "professional" else "[2/3]"
        yield [], f"📝 {step} 正在生成面试题目（RAG检索 + LLM规划）...", "", gr.update(interactive=False)

        payload = await self.api.start_session(self.session_id)
        state = self._state_from_payload(payload)
        self.is_running = True

        question = _get_last_interviewer_message(state)
        next_turn = self._next_turn_from_payload(payload)
        question_count = (
            (next_turn or {}).get("total_questions")
            or state.get("question_count")
            or len(state.get("question_plan", []))
        )
        mode_label = "练习" if mode == "practice" else "专业面试"

        resume_info = "  |  简历已接入" if self.resume_attached else ""

        round_info = ""
        if mode == "professional":
            round_info = "  |  当前: 一面（技术深度）"

        status = f"✅ [{mode_label}模式] 已生成 {question_count} 道面试题{resume_info}{round_info}，面试开始！"

        history = []
        if question:
            history.append({"role": "assistant", "content": question})

        yield history, status, "", gr.update(interactive=True)

    async def submit_answer(self, answer: str, history: list):
        if not self.session_id or not self.is_running:
            return history, "⚠️ 面试尚未开始", "", gr.update(), ""

        if not answer.strip():
            return history, "⚠️ 请输入你的回答", answer, gr.update(), ""

        history = history + [{"role": "user", "content": answer}]
        payload = await self.api.submit_answer(self.session_id, answer.strip())
        state = self._state_from_payload(payload)

        if state.get("interview_complete"):
            self.is_running = False
            return self._handle_complete(state, history)

        next_turn = self._next_turn_from_payload(payload)
        interviewer_msg = (next_turn or {}).get("content") or _get_last_interviewer_message(state)
        idx = state.get("current_question_index", 0)
        plan = state.get("question_plan", [])
        fu = state.get("follow_up_count", 0)
        current_round = state.get("current_round", 1)

        round_prefix = ""
        if self.mode == "professional":
            round_label = "一面" if current_round == 1 else "二面"
            round_prefix = f"[{round_label}] "

        if fu > 0:
            status = f"🔄 {round_prefix}追问 #{fu} — 第 {idx+1}/{len(plan)} 题"
        else:
            status = f"📝 {round_prefix}第 {idx+1}/{len(plan)} 题"

        if interviewer_msg:
            history.append({"role": "assistant", "content": interviewer_msg})

        return history, status, "", gr.update(interactive=True), ""

    def _handle_complete(self, state: dict, history: list):
        if self.mode == "practice":
            report = state.get("practice_report")
            if report:
                self.last_report = report
                rendered = _format_practice_report(report)
                history.append({
                    "role": "assistant",
                    "content": "练习结束！请查看下方的评估报告和知识点总结。点击「导出记录」可保存为 Markdown 文件。",
                })
                return history, "🎉 练习完成！", "", gr.update(interactive=False), rendered
        else:
            pro_report = state.get("professional_report")
            if pro_report:
                self.last_report = pro_report
                rendered = _format_dual_round_report(pro_report)
                history.append({
                    "role": "assistant",
                    "content": "面试结束！一面（技术深度）和二面（技术广度）均已完成。综合评估报告已生成，请查看下方。点击「导出记录」可保存。",
                })
                return history, "🎉 面试完成（一面 + 二面）！", "", gr.update(interactive=False), rendered

            report = state.get("final_report")
            if report:
                self.last_report = report
                rendered = _format_professional_report(report)
                history.append({
                    "role": "assistant",
                    "content": "面试结束！评估报告已生成，请查看下方的详细报告。点击「导出记录」可保存。",
                })
                return history, "🎉 面试完成！", "", gr.update(interactive=False), rendered

        history.append({"role": "assistant", "content": "面试结束！"})
        return (
            history, "✅ 完成", "", gr.update(interactive=False),
            '<div class="report-card"><p>⚠️ 评估报告未能生成，请重试。</p></div>',
        )


# ── Build the Gradio UI ─────────────────────────────────────────────


def build_app():
    interview_app = InterviewApp()

    with gr.Blocks(title="AI Mock Interview") as app:

        gr.HTML("""
        <div class="main-header">
            <h1>🎯 AI Mock Interview Agent</h1>
            <p>Multi-Agent 智能面试模拟系统 — 练习巩固 · 简历深挖 · 双轮考察 · 可视化报告</p>
        </div>
        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">🤖</div>
                <div class="title">6+ Agent 协作</div>
                <div class="desc">JD分析 · 简历解析 · 出题 · 面试 · 评估</div>
            </div>
            <div class="feature-card">
                <div class="icon">📊</div>
                <div class="title">雷达图可视化</div>
                <div class="desc">多维技能评分直观展示</div>
            </div>
            <div class="feature-card">
                <div class="icon">🔄</div>
                <div class="title">双轮面试</div>
                <div class="desc">一面技术深度 + 二面技术广度</div>
            </div>
            <div class="feature-card">
                <div class="icon">📥</div>
                <div class="title">一键导出</div>
                <div class="desc">面试记录 + 报告 → Markdown</div>
            </div>
        </div>
        """)

        with gr.Row():
            # ── Left panel ─────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📋 面试设置")

                mode_selector = gr.Radio(
                    choices=[
                        ("🎓 练习模式（八股巩固 + 答案总结）", "practice"),
                        ("💼 专业面试模式（简历解析 + 项目深挖）", "professional"),
                    ],
                    value="practice",
                    label="面试模式",
                )

                jd_input = gr.Textbox(
                    label="JD 文本",
                    placeholder="粘贴完整的职位描述...",
                    lines=8,
                    value=SAMPLE_JD,
                )

                resume_upload = gr.File(
                    label="📄 上传简历（PDF / PNG）",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    visible=False,
                )
                resume_hint = gr.Markdown(
                    value="",
                    visible=False,
                )

                with gr.Row():
                    max_fu = gr.Slider(
                        minimum=0, maximum=3, value=2, step=1,
                        label="每题最大追问次数",
                    )
                with gr.Row():
                    start_btn = gr.Button(
                        "🚀 开始面试", variant="primary", size="lg", scale=3,
                    )
                    stop_btn = gr.Button(
                        "⏹️ 结束", variant="stop", size="lg", scale=1,
                    )
                status_display = gr.Textbox(
                    label="状态",
                    value="等待开始...",
                    interactive=False,
                    max_lines=2,
                )

            # ── Right panel ────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 💬 面试对话")
                chatbot = gr.Chatbot(
                    label="Interview",
                    height=480,
                    render_markdown=True,
                )
                with gr.Row():
                    answer_input = gr.Textbox(
                        label="你的回答",
                        placeholder="输入你的回答，按 Enter 或点击发送...",
                        lines=3,
                        scale=5,
                        interactive=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

        gr.Markdown("### 📊 评估报告")
        report_html = gr.HTML(value="")

        with gr.Row():
            export_btn = gr.Button("📥 导出面试记录 (Markdown)", variant="secondary", size="sm")
            export_status = gr.HTML(value="")

        # ── Mode toggle: show/hide resume upload ───────────────────

        def on_mode_change(mode):
            if mode == "professional":
                return (
                    gr.update(visible=True),
                    gr.update(
                        value="*专业面试模式需要上传简历 (PDF/PNG)，系统将基于你的简历进行项目深挖和技术考察。*",
                        visible=True,
                    ),
                )
            return gr.update(visible=False), gr.update(value="", visible=False)

        mode_selector.change(
            fn=on_mode_change,
            inputs=[mode_selector],
            outputs=[resume_upload, resume_hint],
        )

        # ── Event handlers ──────────────────────────────────────────

        async def on_start(jd, fu, mode, resume_file):
            try:
                file_path = None
                if resume_file is not None:
                    file_path = resume_file.name if hasattr(resume_file, "name") else resume_file
                async for h, s, c, u in interview_app.start_interview(jd, fu, mode, file_path):
                    yield h, s, c, u, ""
            except Exception as e:
                logger.exception("Start interview failed")
                yield [], f"❌ 启动失败: {e}", "", gr.update(interactive=False), ""

        start_btn.click(
            fn=on_start,
            inputs=[jd_input, max_fu, mode_selector, resume_upload],
            outputs=[chatbot, status_display, answer_input, answer_input, report_html],
        )

        async def on_send(answer, history):
            try:
                return await interview_app.submit_answer(answer, history)
            except Exception as e:
                logger.exception("Submit answer failed")
                history = history + [
                    {"role": "user", "content": answer},
                    {"role": "assistant", "content": f"⚠️ 处理出错: {e}"},
                ]
                return history, f"❌ 错误: {e}", "", gr.update(), ""

        send_btn.click(
            fn=on_send,
            inputs=[answer_input, chatbot],
            outputs=[chatbot, status_display, answer_input, answer_input, report_html],
        )
        answer_input.submit(
            fn=on_send,
            inputs=[answer_input, chatbot],
            outputs=[chatbot, status_display, answer_input, answer_input, report_html],
        )

        async def on_stop(history):
            try:
                if not interview_app.is_running:
                    return history, "⚠️ 没有正在进行的面试", ""

                interview_app.is_running = False
                history = history + [
                    {"role": "assistant", "content": "⏹️ 面试已提前结束，正在生成评估报告..."}
                ]

                payload = await interview_app.api.stop_session(interview_app.session_id)
                state = interview_app._state_from_payload(payload)
                result = interview_app._handle_complete(state, history)
                return result[0], result[1], result[4]
            except Exception as e:
                logger.exception("Stop interview failed")
                return history, f"❌ 停止失败: {e}", ""

        stop_btn.click(
            fn=on_stop,
            inputs=[chatbot],
            outputs=[chatbot, status_display, report_html],
        )

        # ── Export handler ─────────────────────────────────────────

        def on_export(history):
            try:
                if not history:
                    return '<span style="color:#e53e3e;">⚠️ 没有面试记录可以导出</span>'

                md_content = export_interview_markdown(
                    history=history,
                    report=interview_app.last_report,
                    mode=interview_app.mode,
                    jd_text=interview_app.jd_text,
                )
                export_dir = PROJECT_ROOT / "exports"
                filepath = save_export(md_content, directory=export_dir)
                return (
                    f'<div class="export-success">✅ 面试记录已导出到: '
                    f'<code>{filepath}</code></div>'
                )
            except Exception as e:
                logger.exception("Export failed")
                return f'<span style="color:#e53e3e;">❌ 导出失败: {e}</span>'

        export_btn.click(
            fn=on_export,
            inputs=[chatbot],
            outputs=[export_status],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )
