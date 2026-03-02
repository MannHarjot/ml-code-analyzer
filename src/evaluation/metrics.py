"""Evaluation metrics and reporting for trained classifiers.

Computes precision, recall, F1, ROC-AUC, confusion matrix, and formats
results for both human-readable display and machine-readable JSON output.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import BaseEstimator

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """Compute a full suite of classification metrics.

    Args:
        y_true: Ground-truth binary labels (0=clean, 1=buggy).
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for the positive class (buggy).

    Returns:
        Dictionary containing accuracy, per-class and macro metrics, ROC-AUC,
        confusion matrix, and a formatted classification report string.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_buggy": float(f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)),
        "f1_clean": float(f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["clean", "buggy"],
            zero_division=0,
        ),
    }
    return metrics


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    top_n_features: int = 10,
) -> dict[str, Any]:
    """Evaluate a trained model on a held-out test set.

    Args:
        model: Trained scikit-learn classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        feature_names: Feature name list (must align with X_test columns).
        top_n_features: How many top features to include in the report.

    Returns:
        Dictionary with all metrics plus feature importance ranking.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        ranked = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        metrics["top_features"] = [
            {"feature": name, "importance": float(score)}
            for name, score in ranked[:top_n_features]
        ]

    logger.info(
        "Evaluation — Accuracy=%.3f  F1=%.3f  ROC-AUC=%.3f",
        metrics["accuracy"], metrics["f1_weighted"], metrics["roc_auc"],
    )
    return metrics


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """Format evaluation metrics as a readable text table.

    Args:
        metrics: Metrics dictionary from compute_metrics or evaluate_model.

    Returns:
        Formatted multi-line string suitable for CLI or logging output.
    """
    lines = [
        "=" * 50,
        "MODEL EVALUATION RESULTS",
        "=" * 50,
        f"  Accuracy          : {metrics.get('accuracy', 0):.4f}",
        f"  F1 (weighted)     : {metrics.get('f1_weighted', 0):.4f}",
        f"  F1 (macro)        : {metrics.get('f1_macro', 0):.4f}",
        f"  F1 (buggy class)  : {metrics.get('f1_buggy', 0):.4f}",
        f"  Precision (wtd)   : {metrics.get('precision_weighted', 0):.4f}",
        f"  Recall (wtd)      : {metrics.get('recall_weighted', 0):.4f}",
        f"  ROC-AUC           : {metrics.get('roc_auc', 0):.4f}",
        "-" * 50,
        "CONFUSION MATRIX (rows=actual, cols=predicted)",
        "  [TN  FP]",
        f"  [{metrics.get('true_negatives', 0):<4} {metrics.get('false_positives', 0):<4}]",
        "  [FN  TP]",
        f"  [{metrics.get('false_negatives', 0):<4} {metrics.get('true_positives', 0):<4}]",
        "=" * 50,
    ]

    top_features = metrics.get("top_features", [])
    if top_features:
        lines.append("TOP FEATURES BY IMPORTANCE")
        lines.append("-" * 50)
        for i, feat in enumerate(top_features, start=1):
            lines.append(f"  {i:>2}. {feat['feature']:<30} {feat['importance']:.4f}")
        lines.append("=" * 50)

    return "\n".join(lines)
