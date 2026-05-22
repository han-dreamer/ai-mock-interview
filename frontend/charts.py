"""Radar chart generation for interview reports using matplotlib.

Generates SVG strings that can be embedded directly in Gradio HTML components.
"""

from __future__ import annotations

import base64
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


def _setup_chinese_font():
    """Try to find a Chinese-capable font; fall back to sans-serif."""
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei", "Noto Sans CJK"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "sans-serif"


_CN_FONT = _setup_chinese_font()


def generate_radar_chart(
    labels: list[str],
    scores: list[float],
    max_score: float = 10.0,
    title: str = "",
    color: str = "#667eea",
    fill_alpha: float = 0.25,
    size: tuple[int, int] = (380, 380),
) -> str:
    """Generate a radar chart as a base64-encoded PNG <img> tag.

    Returns an HTML <img> tag with the chart embedded as a data URI.
    """
    n = len(labels)
    if n < 3:
        return ""

    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    values = [min(s / max_score, 1.0) for s in scores]

    angles += angles[:1]
    values += values[:1]

    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(angles, values, "o-", linewidth=2, color=color, markersize=5)
    ax.fill(angles, values, alpha=fill_alpha, color=color)

    ax.set_xticks(angles[:-1])
    truncated = [l[:6] + ".." if len(l) > 8 else l for l in labels]
    ax.set_xticklabels(truncated, fontsize=9, fontfamily=_CN_FONT)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color="#999")
    ax.spines["polar"].set_color("#ddd")
    ax.grid(color="#e0e0e0", linewidth=0.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=18, fontfamily=_CN_FONT)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;" />'


def generate_dual_radar_chart(
    labels: list[str],
    scores1: list[float],
    scores2: list[float],
    legend1: str = "一面（深度）",
    legend2: str = "二面（广度）",
    max_score: float = 10.0,
    color1: str = "#667eea",
    color2: str = "#f5576c",
) -> str:
    """Generate an overlaid dual-radar chart for comparing two rounds."""
    n = len(labels)
    if n < 3:
        return ""

    angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
    v1 = [min(s / max_score, 1.0) for s in scores1]
    v2 = [min(s / max_score, 1.0) for s in scores2]

    angles += angles[:1]
    v1 += v1[:1]
    v2 += v2[:1]

    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(angles, v1, "o-", linewidth=2, color=color1, markersize=4, label=legend1)
    ax.fill(angles, v1, alpha=0.15, color=color1)
    ax.plot(angles, v2, "s-", linewidth=2, color=color2, markersize=4, label=legend2)
    ax.fill(angles, v2, alpha=0.15, color=color2)

    ax.set_xticks(angles[:-1])
    truncated = [l[:6] + ".." if len(l) > 8 else l for l in labels]
    ax.set_xticklabels(truncated, fontsize=9, fontfamily=_CN_FONT)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=7, color="#999")
    ax.spines["polar"].set_color("#ddd")
    ax.grid(color="#e0e0e0", linewidth=0.5)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9, prop={"family": _CN_FONT})

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;" />'
