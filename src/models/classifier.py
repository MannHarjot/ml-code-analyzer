"""ML classifier definitions and serialization utilities.

Provides factory functions for Random Forest and Gradient Boosting classifiers
tuned for the code quality prediction task, plus model save/load helpers.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator

from src.utils.logger import get_logger

logger = get_logger(__name__)

ModelType = RandomForestClassifier | GradientBoostingClassifier


def get_model(model_name: str, **kwargs: Any) -> ModelType:
    """Factory function that returns a configured classifier instance.

    Args:
        model_name: One of 'random_forest' or 'gradient_boosting'.
        **kwargs: Override default hyperparameters for the chosen model.

    Returns:
        Configured scikit-learn classifier instance.

    Raises:
        ValueError: If model_name is not recognized.
    """
    defaults: dict[str, dict[str, Any]] = {
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        },
    }

    if model_name not in defaults:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(defaults.keys())}"
        )

    params = {**defaults[model_name], **kwargs}

    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    return GradientBoostingClassifier(**params)


def get_feature_importance(
    model: ModelType,
    feature_names: list[str],
    top_n: int = 15,
) -> list[tuple[str, float]]:
    """Extract and rank feature importances from a trained model.

    Args:
        model: A trained scikit-learn classifier with feature_importances_.
        feature_names: List of feature names corresponding to model input columns.
        top_n: Number of top features to return.

    Returns:
        List of (feature_name, importance_score) tuples sorted descending.
    """
    importances: np.ndarray = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


def save_model(
    model: BaseEstimator,
    model_path: Path,
    feature_names: list[str],
    metrics: dict[str, Any],
) -> None:
    """Persist a trained model and its associated metadata.

    Saves the model as a joblib file and writes feature names and metrics
    as JSON files alongside it.

    Args:
        model: Trained scikit-learn estimator to save.
        model_path: Target path for the .joblib file.
        feature_names: Ordered list of feature names the model was trained on.
        metrics: Dictionary of training/evaluation metrics to persist.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    feature_path = model_path.parent / "feature_names.json"
    feature_path.write_text(json.dumps(feature_names, indent=2))
    logger.info("Saved feature names to %s", feature_path)

    metrics_path = model_path.parent / "metrics.json"
    # Convert numpy types to native Python for JSON serialization
    clean_metrics = _serialize_metrics(metrics)
    metrics_path.write_text(json.dumps(clean_metrics, indent=2))
    logger.info("Saved metrics to %s", metrics_path)


def load_model(model_dir: Path) -> tuple[BaseEstimator, list[str], dict[str, Any]]:
    """Load a saved model, feature names, and metrics from disk.

    Args:
        model_dir: Directory containing best_model.joblib, feature_names.json,
                   and metrics.json.

    Returns:
        Tuple of (model, feature_names, metrics).

    Raises:
        FileNotFoundError: If required model files are missing.
    """
    model_dir = Path(model_dir)
    model_path = model_dir / "best_model.joblib"
    feature_path = model_dir / "feature_names.json"
    metrics_path = model_dir / "metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No model file found at {model_path}")

    model = joblib.load(model_path)
    feature_names: list[str] = json.loads(feature_path.read_text()) if feature_path.exists() else []
    metrics: dict[str, Any] = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    logger.info("Loaded model from %s", model_path)
    return model, feature_names, metrics


def _serialize_metrics(obj: Any) -> Any:
    """Recursively convert numpy types to Python-native types for JSON.

    Args:
        obj: Any object (dict, list, numpy scalar, etc.).

    Returns:
        JSON-serializable version of the object.
    """
    if isinstance(obj, dict):
        return {k: _serialize_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_metrics(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
