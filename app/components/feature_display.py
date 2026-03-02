"""Feature breakdown table and bar chart component for Streamlit."""

from typing import Any

import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THRESHOLD_STATUS = {
    # feature: (warning_threshold, danger_threshold, higher_is_worse)
    "cyclomatic_complexity": (10, 20, True),
    "max_nesting_depth": (3, 5, True),
    "avg_nesting_depth": (2, 4, True),
    "comment_ratio": (0.1, 0.05, False),        # lower = worse
    "docstring_coverage": (0.7, 0.3, False),    # lower = worse
    "type_hint_coverage": (0.7, 0.3, False),    # lower = worse
    "avg_function_length": (20, 50, True),
    "max_function_length": (40, 100, True),
    "num_global_variables": (2, 5, True),
    "num_nested_functions": (2, 5, True),
    "uses_star_import": (0.5, 0.5, True),
    "num_branches": (10, 25, True),
    "num_loops": (6, 15, True),
    "total_lines": (200, 500, True),
}

FEATURE_LABELS = {
    "total_lines": "Total Lines",
    "code_lines": "Code Lines",
    "blank_lines": "Blank Lines",
    "comment_lines": "Comment Lines",
    "comment_ratio": "Comment Ratio",
    "num_functions": "Num Functions",
    "num_classes": "Num Classes",
    "num_methods": "Num Methods",
    "avg_function_length": "Avg Fn Length",
    "max_function_length": "Max Fn Length",
    "num_imports": "Num Imports",
    "num_unique_imports": "Unique Imports",
    "cyclomatic_complexity": "Cyclomatic Complexity",
    "max_nesting_depth": "Max Nesting Depth",
    "avg_nesting_depth": "Avg Nesting Depth",
    "num_branches": "Num Branches",
    "num_loops": "Num Loops",
    "num_try_except": "Try/Except Blocks",
    "num_assertions": "Assertions",
    "num_lambda_functions": "Lambda Functions",
    "has_docstrings": "Has Docstrings",
    "docstring_coverage": "Docstring Coverage",
    "has_type_hints": "Has Type Hints",
    "type_hint_coverage": "Type Hint Coverage",
    "avg_identifier_length": "Avg Identifier Length",
    "num_global_variables": "Global Variables",
    "num_nested_functions": "Nested Functions",
    "num_return_statements": "Return Statements",
    "uses_star_import": "Uses Star Import",
    "syntax_error": "Syntax Error",
}


def _status_badge(feature: str, value: float) -> str:
    """Return an emoji status badge for a feature value.

    Args:
        feature: Feature name key.
        value: Numeric feature value.

    Returns:
        Status string: '✅ Healthy', '⚠️ Warning', or '🔴 Danger'.
    """
    config = THRESHOLD_STATUS.get(feature)
    if config is None:
        return "—"
    warn, danger, higher_is_worse = config
    if higher_is_worse:
        if value <= warn:
            return "✅ Healthy"
        if value <= danger:
            return "⚠️ Warning"
        return "🔴 Danger"
    else:
        if value >= warn:
            return "✅ Healthy"
        if value >= danger:
            return "⚠️ Warning"
        return "🔴 Danger"


def render_feature_table(features: dict[str, Any]) -> None:
    """Render a color-coded feature breakdown table.

    Args:
        features: Extracted feature dictionary from the analyzer.
    """
    rows = []
    skip = {"syntax_error", "file_path", "repo"}
    for feat, raw_val in features.items():
        if feat in skip:
            continue
        label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
        if isinstance(raw_val, float):
            display_val = f"{raw_val:.4f}" if raw_val < 1 else f"{raw_val:.2f}"
        else:
            display_val = str(raw_val)
        status = _status_badge(feat, float(raw_val))
        rows.append({"Feature": label, "Value": display_val, "Status": status})

    df = pd.DataFrame(rows)

    def _color_status(val: str) -> str:
        if "Healthy" in val:
            return "color: #10B981; font-weight: 600;"
        if "Warning" in val:
            return "color: #F59E0B; font-weight: 600;"
        if "Danger" in val:
            return "color: #EF4444; font-weight: 600;"
        return ""

    styled = df.style.applymap(_color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, height=420)


def render_top_features_chart(model: Any, feature_names: list[str], top_n: int = 12) -> None:
    """Render a horizontal bar chart of top model feature importances.

    Args:
        model: Trained model with .feature_importances_ attribute.
        feature_names: Feature names aligned with model inputs.
        top_n: Number of top features to display.
    """
    if not hasattr(model, "feature_importances_"):
        st.info("Feature importance not available for this model type.")
        return

    import numpy as np
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names = [FEATURE_LABELS.get(feature_names[i], feature_names[i]) for i in indices][::-1]
    values = importances[indices][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    colors = ["#2563EB" if v > values.mean() else "#93C5FD" for v in values]
    ax.barh(range(len(names)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title(f"Top {top_n} Risk Predictors", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
