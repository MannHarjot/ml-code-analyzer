"""Tests for model factory, training, and prediction pipeline."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classifier import get_model, save_model, load_model, get_feature_importance
from src.data.synthetic_dataset import generate_synthetic_dataset, FEATURE_NAMES


@pytest.fixture(scope="module")
def small_dataset():
    """Generate a small synthetic dataset for fast tests."""
    df = generate_synthetic_dataset(n_samples=400, buggy_ratio=0.4)
    X = df[FEATURE_NAMES].values.astype(np.float64)
    y = df["label"].values.astype(int)
    return X, y


@pytest.fixture(scope="module")
def trained_rf(small_dataset):
    """Return a trained Random Forest on the small dataset."""
    X, y = small_dataset
    model = get_model("random_forest", n_estimators=50, random_state=42)
    model.fit(X, y)
    return model


class TestGetModel:
    def test_random_forest_returns_correct_type(self):
        from sklearn.ensemble import RandomForestClassifier
        model = get_model("random_forest")
        assert isinstance(model, RandomForestClassifier)

    def test_gradient_boosting_returns_correct_type(self):
        from sklearn.ensemble import GradientBoostingClassifier
        model = get_model("gradient_boosting")
        assert isinstance(model, GradientBoostingClassifier)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("neural_network")

    def test_kwargs_override_defaults(self):
        model = get_model("random_forest", n_estimators=42)
        assert model.n_estimators == 42


class TestModelTraining:
    def test_model_fits_without_error(self, small_dataset):
        X, y = small_dataset
        model = get_model("random_forest", n_estimators=20)
        model.fit(X, y)
        assert hasattr(model, "feature_importances_")

    def test_predict_returns_binary_labels(self, trained_rf, small_dataset):
        X, _ = small_dataset
        preds = trained_rf.predict(X[:10])
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, trained_rf, small_dataset):
        X, _ = small_dataset
        proba = trained_rf.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(10), atol=1e-6)

    def test_proba_between_zero_and_one(self, trained_rf, small_dataset):
        X, _ = small_dataset
        proba = trained_rf.predict_proba(X[:20])
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_model_accuracy_above_chance(self, trained_rf, small_dataset):
        X, y = small_dataset
        preds = trained_rf.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy > 0.55, f"Model accuracy too low: {accuracy:.3f}"


class TestFeatureImportance:
    def test_returns_correct_length(self, trained_rf):
        importance = get_feature_importance(trained_rf, FEATURE_NAMES, top_n=10)
        assert len(importance) == 10

    def test_sorted_descending(self, trained_rf):
        importance = get_feature_importance(trained_rf, FEATURE_NAMES, top_n=15)
        scores = [score for _, score in importance]
        assert scores == sorted(scores, reverse=True)

    def test_all_names_from_feature_list(self, trained_rf):
        importance = get_feature_importance(trained_rf, FEATURE_NAMES)
        names = [name for name, _ in importance]
        for name in names:
            assert name in FEATURE_NAMES


class TestModelSaveLoad:
    def test_save_and_load_roundtrip(self, trained_rf):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "best_model.joblib"
            metrics = {"accuracy": 0.87, "f1": 0.85}
            save_model(trained_rf, model_path, FEATURE_NAMES, metrics)

            assert model_path.exists()
            assert (Path(tmpdir) / "feature_names.json").exists()
            assert (Path(tmpdir) / "metrics.json").exists()

            loaded_model, loaded_features, loaded_metrics = load_model(Path(tmpdir))
            assert loaded_features == FEATURE_NAMES
            assert loaded_metrics["accuracy"] == pytest.approx(0.87, abs=0.001)

    def test_predictions_identical_after_reload(self, trained_rf, small_dataset):
        X, _ = small_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "best_model.joblib"
            save_model(trained_rf, model_path, FEATURE_NAMES, {})
            loaded_model, _, _ = load_model(Path(tmpdir))

            original_proba = trained_rf.predict_proba(X[:20])
            loaded_proba = loaded_model.predict_proba(X[:20])
            np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-10)
