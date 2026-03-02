"""Training pipeline with cross-validation, hyperparameter search, and model selection.

Runs stratified k-fold CV, optional grid search, and automatically selects
the best classifier by weighted F1 score. Supports class balancing via
class_weight or SMOTE.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.models.classifier import get_model, save_model, ModelType
from src.data.synthetic_dataset import FEATURE_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_DIR = Path("models/pretrained")
DEFAULT_DATA_PATH = Path("data/dataset.csv")


def load_dataset(data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the training CSV and return feature matrix, labels, and feature names.

    Args:
        data_path: Path to the dataset CSV file.

    Returns:
        Tuple of (X, y, feature_names) where X is (n_samples, n_features),
        y is (n_samples,), and feature_names is the list of column names.

    Raises:
        FileNotFoundError: If the dataset CSV does not exist.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        logger.warning("Missing feature columns in dataset: %s", missing)

    X = df[feature_cols].fillna(0).values.astype(np.float64)
    y = df["label"].values.astype(int)

    logger.info(
        "Loaded dataset: %d samples, %d features — buggy=%.1f%%",
        len(y), X.shape[1], 100 * y.mean()
    )
    return X, y, feature_cols


def _evaluate_fold(
    model: ModelType,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Compute evaluation metrics on a validation fold.

    Args:
        model: A fitted classifier.
        X_val: Validation feature matrix.
        y_val: True labels for the validation fold.

    Returns:
        Dictionary of metric name -> float value.
    """
    from sklearn.metrics import confusion_matrix as sk_cm

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    cm = sk_cm(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_val, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "f1_buggy": float(f1_score(y_val, y_pred, pos_label=1, average="binary", zero_division=0)),
        "precision_weighted": float(precision_score(y_val, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_val, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def cross_validate_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run stratified k-fold cross-validation for a model.

    Args:
        model_name: 'random_forest' or 'gradient_boosting'.
        X: Feature matrix.
        y: Label vector.
        cv_folds: Number of cross-validation folds.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with per-fold and aggregate metrics.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = get_model(model_name)
        model.fit(X_train, y_train)
        metrics = _evaluate_fold(model, X_val, y_val)
        fold_metrics.append(metrics)
        logger.info(
            "Fold %d/%d — acc=%.3f f1=%.3f roc_auc=%.3f",
            fold_idx, cv_folds,
            metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        )

    # Aggregate across folds
    agg: dict[str, Any] = {"folds": fold_metrics}
    for metric in fold_metrics[0]:
        values = [f[metric] for f in fold_metrics]
        agg[f"mean_{metric}"] = float(np.mean(values))
        agg[f"std_{metric}"] = float(np.std(values))

    logger.info(
        "%s CV results — F1=%.3f±%.3f  AUC=%.3f±%.3f",
        model_name,
        agg["mean_f1"], agg["std_f1"],
        agg["mean_roc_auc"], agg["std_roc_auc"],
    )
    return agg


def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
    use_grid_search: bool = False,
    cv_folds: int = 3,
) -> ModelType:
    """Train a model on the full training split, optionally with grid search.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        model_name: 'random_forest' or 'gradient_boosting'.
        use_grid_search: Whether to run GridSearchCV for hyperparameter tuning.
        cv_folds: Number of CV folds for grid search.

    Returns:
        Fitted best estimator.
    """
    model = get_model(model_name)

    if use_grid_search:
        param_grids: dict[str, dict[str, list]] = {
            "random_forest": {
                "n_estimators": [100, 200],
                "max_depth": [10, 15],
                "min_samples_split": [5, 10],
            },
            "gradient_boosting": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
            },
        }
        param_grid = param_grids.get(model_name, {})
        if param_grid:
            gs = GridSearchCV(
                model, param_grid,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_train, y_train)
            logger.info("Best params for %s: %s", model_name, gs.best_params_)
            return gs.best_estimator_

    model.fit(X_train, y_train)
    return model


def run_training_pipeline(
    data_path: Path = DEFAULT_DATA_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    model_names: Optional[list[str]] = None,
    cv_folds: int = 5,
    test_size: float = 0.2,
    use_grid_search: bool = False,
    random_state: int = 42,
) -> dict[str, Any]:
    """End-to-end training pipeline: load data, CV, train, evaluate, save.

    Compares all specified models by F1 score and saves the best one.

    Args:
        data_path: Path to the training dataset CSV.
        model_dir: Directory where the best model will be saved.
        model_names: Models to train and compare. Defaults to both classifiers.
        cv_folds: Number of folds for cross-validation.
        test_size: Fraction of data held out for final evaluation.
        use_grid_search: Enable GridSearchCV for hyperparameter tuning.
        random_state: Global random seed.

    Returns:
        Dictionary with training results for all models plus the best model info.
    """
    if model_names is None:
        model_names = ["random_forest", "gradient_boosting"]

    X, y, feature_names = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train: %d samples, Test: %d samples", len(y_train), len(y_test))

    results: dict[str, Any] = {}
    best_model_name: Optional[str] = None
    best_f1 = -1.0
    best_model: Optional[ModelType] = None

    for model_name in model_names:
        logger.info("=" * 40)
        logger.info("Training: %s", model_name)

        cv_metrics = cross_validate_model(
            model_name, X_train, y_train,
            cv_folds=cv_folds, random_state=random_state,
        )

        trained_model = train_best_model(
            X_train, y_train, model_name,
            use_grid_search=use_grid_search,
        )

        test_metrics = _evaluate_fold(trained_model, X_test, y_test)
        logger.info(
            "%s test metrics — F1=%.3f  AUC=%.3f  Acc=%.3f",
            model_name,
            test_metrics["f1"], test_metrics["roc_auc"], test_metrics["accuracy"],
        )

        results[model_name] = {
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
        }

        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_model_name = model_name
            best_model = trained_model

    # Save the best model
    if best_model is not None and best_model_name is not None:
        best_results = results[best_model_name]
        all_metrics = {
            "model_name": best_model_name,
            "feature_names": feature_names,
            "test_metrics": best_results["test_metrics"],
            "cv_metrics": best_results["cv_metrics"],
            "all_model_results": {
                name: res["test_metrics"] for name, res in results.items()
            },
        }
        save_model(
            best_model,
            model_dir / "best_model.joblib",
            feature_names,
            all_metrics,
        )
        results["best_model"] = best_model_name
        results["best_model_instance"] = best_model
        results["feature_names"] = feature_names
        logger.info("Best model: %s (F1=%.3f)", best_model_name, best_f1)

    return results
