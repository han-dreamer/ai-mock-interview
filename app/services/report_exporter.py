"""Export interview reports to portable text formats."""

from __future__ import annotations

from html import escape
from typing import Any


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    return value


def _line(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value)


def _mode_name(mode: str) -> str:
    return "专业面试模式" if mode == "professional" else "练习模式"


def _grade(report: dict[str, Any]) -> str:
    return _line(report.get("grade"), "?")


def _score(report: dict[str, Any]) -> str:
    value = report.get("overall_score")
    if isinstance(value, int | float):
        return f"{value:.1f}"
    return _line(value, "暂无")


def _bullet_lines(items: list[Any], *, evidence: bool = True) -> list[str]:
    lines: list[str] = []
    for item in items:
        if isinstance(item, dict):
            title = item.get("point") or item.get("skill_name") or item.get("question") or "条目"
            body = item.get("evidence") or item.get("summary") or item.get("reference_answer") or ""
            score = item.get("score")
            prefix = f"- {title}"
            if score is not None:
                prefix += f"：{score}/10"
            if evidence and body:
                prefix += f" - {body}"
            lines.append(prefix)
        else:
            lines.append(f"- {item}")
    return lines


def report_to_markdown(report: Any, *, mode: str, session_id: str) -> str:
    """Render a report as Markdown."""
    data = _dump(report) or {}
    lines = [
        "# AI 模拟面试报告",
        "",
        f"- 会话 ID：{session_id}",
        f"- 模式：{_mode_name(mode)}",
        f"- 总分：{_score(data)}/10",
        f"- 等级：{_grade(data)}",
        "",
        "## 总体评价",
        _line(data.get("overall_assessment"), "暂无总体评价。"),
        "",
    ]

    if data.get("hiring_recommendation"):
        lines.extend(["## 面试结论", _line(data.get("hiring_recommendation")), ""])

    if data.get("round_scores"):
        lines.extend(["## 轮次表现", *_bullet_lines(data["round_scores"]), ""])

    if data.get("skill_scores"):
        lines.extend(["## 技能评分", *_bullet_lines(data["skill_scores"]), ""])

    if data.get("strengths"):
        lines.extend(["## 优势表现", *_bullet_lines(data["strengths"]), ""])

    if data.get("improvements"):
        lines.extend(["## 改进建议", *_bullet_lines(data["improvements"]), ""])

    if data.get("missed_knowledge"):
        lines.append("## 遗漏知识点")
        for item in data["missed_knowledge"]:
            if not isinstance(item, dict):
                lines.append(f"- {item}")
                continue
            lines.append(f"### {_line(item.get('question'), '问题')}")
            lines.append(f"- 得分：{_line(item.get('score'), '?')}/10")
            missed = item.get("missed_points") or []
            if missed:
                lines.append("- 遗漏点：" + "；".join(str(point) for point in missed))
            if item.get("reference_answer"):
                lines.append(f"- 参考答案：{item['reference_answer']}")
            lines.append("")

    if data.get("study_suggestions"):
        lines.extend(["## 学习建议", *_bullet_lines(data["study_suggestions"], evidence=False), ""])

    return "\n".join(lines).strip() + "\n"


def markdown_to_html(markdown_text: str, *, title: str = "AI 模拟面试报告") -> str:
    """Render the limited Markdown produced by report_to_markdown as printable HTML."""
    body: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            body.append("</ul>")
            in_list = False

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            close_list()
            continue
        if line.startswith("### "):
            close_list()
            body.append(f"<h3>{escape(line[4:])}</h3>")
        elif line.startswith("## "):
            close_list()
            body.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("# "):
            close_list()
            body.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{escape(line[2:])}</li>")
        else:
            close_list()
            body.append(f"<p>{escape(line)}</p>")
    close_list()

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      color: #172027;
      font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    }}
    body {{
      max-width: 860px;
      margin: 0 auto;
      padding: 42px 28px;
      line-height: 1.72;
      background: #fff;
    }}
    h1 {{
      margin: 0 0 22px;
      font-size: 34px;
      color: #253036;
    }}
    h2 {{
      margin: 30px 0 10px;
      padding-top: 16px;
      border-top: 1px solid #dfe4ea;
      font-size: 21px;
      color: #12615e;
    }}
    h3 {{
      margin: 20px 0 8px;
      font-size: 17px;
      color: #253036;
    }}
    p, li {{
      font-size: 14px;
    }}
    ul {{
      padding-left: 22px;
    }}
    li {{
      margin: 6px 0;
    }}
    @media print {{
      body {{
        padding: 0;
      }}
      h2 {{
        break-after: avoid;
      }}
    }}
  </style>
</head>
<body>
  {"".join(body)}
</body>
</html>
"""
