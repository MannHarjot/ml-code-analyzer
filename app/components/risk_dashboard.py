"""Risk score dashboard component: gauge, badge, and recommendations."""

from typing import Any

import streamlit as st


RISK_CONFIG = {
    "Low": {
        "color": "#10B981",
        "bg": "#D1FAE5",
        "emoji": "✅",
        "description": "This code looks healthy. Minor improvements may still apply.",
    },
    "Medium": {
        "color": "#F59E0B",
        "bg": "#FEF3C7",
        "emoji": "⚠️",
        "description": "Some quality issues detected. Review flagged features.",
    },
    "High": {
        "color": "#EF4444",
        "bg": "#FEE2E2",
        "emoji": "🔴",
        "description": "Significant risk factors found. Refactoring recommended.",
    },
    "Critical": {
        "color": "#7F1D1D",
        "bg": "#FCA5A5",
        "emoji": "🚨",
        "description": "Critical quality issues. This code needs immediate attention.",
    },
}

FEATURE_RECOMMENDATIONS = {
    "cyclomatic_complexity": lambda v: (
        f"Reduce cyclomatic complexity (currently {v:.0f}) — break this module into smaller functions."
        if v > 15 else None
    ),
    "max_nesting_depth": lambda v: (
        f"Reduce nesting depth (currently {v:.0f} levels) — extract inner blocks into helper functions."
        if v > 4 else None
    ),
    "docstring_coverage": lambda v: (
        f"Improve docstring coverage (currently {v:.0%}) — add Google-style docstrings to undocumented functions."
        if v < 0.5 else None
    ),
    "comment_ratio": lambda v: (
        f"Add inline comments (current comment ratio: {v:.1%}) — aim for 10–20%."
        if v < 0.08 else None
    ),
    "type_hint_coverage": lambda v: (
        f"Add type hints (currently {v:.0%} coverage) — improves IDE support and readability."
        if v < 0.4 else None
    ),
    "uses_star_import": lambda v: (
        "Remove star imports (`from x import *`) — they pollute the namespace and hide dependencies."
        if v >= 1 else None
    ),
    "avg_function_length": lambda v: (
        f"Shorten functions (average {v:.0f} lines) — target under 20 lines per function."
        if v > 30 else None
    ),
    "num_global_variables": lambda v: (
        f"Reduce global variables (currently {v:.0f}) — use dependency injection or class attributes instead."
        if v > 4 else None
    ),
    "num_nested_functions": lambda v: (
        f"Consider extracting nested functions ({v:.0f} found) — makes code easier to test."
        if v > 3 else None
    ),
}


def render_risk_gauge(score: float, level: str) -> None:
    """Render an HTML/CSS animated risk score gauge.

    Args:
        score: Risk score between 0.0 and 1.0.
        level: Risk level string ('Low', 'Medium', 'High', 'Critical').
    """
    cfg = RISK_CONFIG.get(level, RISK_CONFIG["Medium"])
    pct = int(score * 100)
    bar_width = max(4, pct)

    st.markdown(
        f"""
        <div style="
            background:{cfg['bg']};
            border-left: 6px solid {cfg['color']};
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 20px;
        ">
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="font-size: 3.5rem;">{cfg['emoji']}</div>
                <div>
                    <div style="font-size: 2.8rem; font-weight: 900; color: {cfg['color']}; line-height:1;">
                        {pct}%
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: {cfg['color']};">
                        {level} Risk
                    </div>
                </div>
            </div>

            <div style="margin-top:16px;">
                <div style="
                    background: #e5e7eb;
                    border-radius: 9999px;
                    height: 14px;
                    overflow: hidden;
                ">
                    <div style="
                        width: {bar_width}%;
                        height: 100%;
                        background: {cfg['color']};
                        border-radius: 9999px;
                        transition: width 0.5s;
                    "></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#6b7280; margin-top:4px;">
                    <span>0% — Clean</span>
                    <span>100% — Critical</span>
                </div>
            </div>

            <p style="margin-top:14px; color:#374151; font-size:0.95rem;">
                {cfg['description']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations(features: dict[str, Any]) -> None:
    """Render actionable improvement recommendations based on feature values.

    Args:
        features: Extracted feature dictionary from the analyzer.
    """
    recs = []
    for feat, fn in FEATURE_RECOMMENDATIONS.items():
        value = features.get(feat, 0)
        msg = fn(value)
        if msg:
            recs.append(msg)

    if not recs:
        st.success("No major recommendations — code quality looks good overall!")
        return

    st.markdown("#### 💡 Recommendations")
    for rec in recs:
        st.markdown(
            f"""
            <div style="
                background:#FFF7ED;
                border-left:4px solid #F59E0B;
                border-radius:6px;
                padding:10px 14px;
                margin-bottom:8px;
                font-size:0.9rem;
                color:#1F2937;
            ">
                ▶ {rec}
            </div>
            """,
            unsafe_allow_html=True,
        )
