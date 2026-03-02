"""Publication-quality visualization functions for model and code analysis.

All functions return matplotlib Figure objects compatible with Streamlit's
st.pyplot() and standard notebook display.
"""

from typing import Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Consistent color palette
PALETTE = {
    "primary": "#2563EB",
    "secondary": "#7C3AED",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "neutral": "#6B7280",
    "buggy": "#EF4444",
    "clean": "#10B981",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 100,
})


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 15,
    title: str = "Feature Importance",
) -> Figure:
    """Plot a horizontal bar chart of top feature importances.

    Args:
        model: Trained model with .feature_importances_ attribute.
        feature_names: List of feature names aligned with model input.
        top_n: Number of top features to display.
        title: Chart title string.

    Returns:
        Matplotlib Figure object.
    """
    importances: np.ndarray = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    # Reverse for horizontal bar (highest at top)
    top_names = top_names[::-1]
    top_values = top_values[::-1]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.45)))
    colors = [PALETTE["primary"]] * len(top_values)
    colors[-1] = PALETTE["secondary"]  # Highlight top feature

    bars = ax.barh(range(len(top_names)), top_values, color=colors, height=0.65)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(
        [n.replace("_", " ").title() for n in top_names],
        fontsize=10,
    )
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    for bar, val in zip(bars, top_values):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=8.5,
        )

    plt.tight_layout()
    return fig


def plot_risk_distribution(
    scores: list[float] | np.ndarray,
    title: str = "Risk Score Distribution",
) -> Figure:
    """Plot a histogram of predicted risk scores across a dataset.

    Args:
        scores: List or array of risk scores (0.0 to 1.0).
        title: Chart title string.

    Returns:
        Matplotlib Figure object.
    """
    scores = np.asarray(scores)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    n, bins, patches = ax.hist(scores, bins=30, edgecolor="white", linewidth=0.5)
    for patch, left in zip(patches, bins[:-1]):
        if left < 0.3:
            patch.set_facecolor(PALETTE["clean"])
        elif left < 0.6:
            patch.set_facecolor(PALETTE["warning"])
        elif left < 0.8:
            patch.set_facecolor(PALETTE["danger"])
        else:
            patch.set_facecolor("#7F0000")

    ax.axvline(0.3, color=PALETTE["clean"], linestyle="--", linewidth=1.2, label="Low/Medium (0.3)")
    ax.axvline(0.6, color=PALETTE["warning"], linestyle="--", linewidth=1.2, label="Medium/High (0.6)")
    ax.axvline(0.8, color=PALETTE["danger"], linestyle="--", linewidth=1.2, label="High/Critical (0.8)")

    ax.set_xlabel("Risk Score", fontsize=11)
    ax.set_ylabel("Number of Files", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
) -> Figure:
    """Plot a styled confusion matrix heatmap.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        labels: Class label names. Defaults to ['Clean', 'Buggy'].
        title: Chart title string.

    Returns:
        Matplotlib Figure object.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    if labels is None:
        labels = ["Clean", "Buggy"]

    cm = sk_cm(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_pct[i, j] > thresh else "black"
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1%})",
                ha="center", va="center",
                color=color, fontsize=12, fontweight="bold",
            )

    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
) -> Figure:
    """Plot the Receiver Operating Characteristic curve with AUC.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        title: Chart title string.

    Returns:
        Matplotlib Figure object.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        fpr, tpr,
        color=PALETTE["primary"], lw=2,
        label=f"ROC Curve (AUC = {roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["primary"])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    return fig


def plot_feature_comparison(
    features: dict[str, float],
    thresholds: dict[str, dict],
    title: str = "Code Quality Radar",
) -> Figure:
    """Plot a radar/spider chart comparing file features to healthy thresholds.

    Normalizes each feature to a 0–1 danger scale using the provided
    threshold ranges, where 1 = worst (danger zone).

    Args:
        features: Dict of feature_name -> value for the analyzed file.
        thresholds: Nested dict from config: {feature: {healthy, warning, danger}}.
        title: Chart title string.

    Returns:
        Matplotlib Figure object.
    """
    # Select the most meaningful features for the radar
    radar_features = [
        "cyclomatic_complexity", "max_nesting_depth", "comment_ratio",
        "docstring_coverage", "type_hint_coverage", "avg_function_length",
        "num_global_variables", "uses_star_import",
    ]
    radar_features = [f for f in radar_features if f in features]
    N = len(radar_features)
    if N < 3:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Not enough features for radar chart",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Normalize: low comment_ratio, docstring, type_hint = bad; high complexity = bad
    INVERT = {"comment_ratio", "docstring_coverage", "type_hint_coverage"}

    def _normalize(feat: str, val: float) -> float:
        """Map feature value to 0 (good) – 1 (bad) scale."""
        t = thresholds.get(feat)
        if t is None:
            return 0.5
        danger_max = t["danger"][1] if t["danger"][1] < 99999 else max(val * 2, 1)
        normalized = float(np.clip(val / max(danger_max, 1e-9), 0, 1))
        if feat in INVERT:
            normalized = 1.0 - normalized
        return normalized

    values = [_normalize(f, features.get(f, 0.0)) for f in radar_features]
    values += values[:1]  # Close the loop

    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    labels = [f.replace("_", "\n").title() for f in radar_features]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, values, "o-", linewidth=2, color=PALETTE["primary"])
    ax.fill(angles, values, alpha=0.25, color=PALETTE["primary"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Low", "Med", "High", "Critical"], size=7, color="grey")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


def plot_metric_bars(
    model_results: dict[str, dict[str, float]],
    title: str = "Model Comparison",
) -> Figure:
    """Bar chart comparing multiple models across key metrics.

    Args:
        model_results: Dict of model_name -> {metric_name: value}.
        title: Chart title.

    Returns:
        Matplotlib Figure object.
    """
    metrics_to_show = ["accuracy", "f1_weighted", "roc_auc"]
    model_names = list(model_results.keys())
    x = np.arange(len(metrics_to_show))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"]]

    for i, name in enumerate(model_names):
        vals = [model_results[name].get(m, 0) for m in metrics_to_show]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=name.replace("_", " ").title(),
                      color=colors[i % len(colors)])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "F1 (Weighted)", "ROC-AUC"], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    return fig
