"""CLI training script: build dataset, train models, save best.

Usage:
    python scripts/train.py
    python scripts/train.py --synthetic --samples 3000
    python scripts/train.py --grid-search
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.build_dataset import build_dataset
from src.models.trainer import run_training_pipeline
from src.evaluation.metrics import format_metrics_table
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_PATH = Path("data/dataset.csv")
DEFAULT_MODEL_DIR = Path("models/pretrained")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train ML models for code quality prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--synthetic", action="store_true", default=True,
        help="Use synthetic dataset (no GitHub cloning required)",
    )
    parser.add_argument(
        "--real-repos", action="store_true",
        help="Mine real GitHub repositories (overrides --synthetic)",
    )
    parser.add_argument(
        "--samples", type=int, default=2500,
        help="Number of synthetic samples to generate",
    )
    parser.add_argument(
        "--buggy-ratio", type=float, default=0.35,
        help="Fraction of samples labeled as buggy",
    )
    parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_DATA_PATH,
        help="Path to save/load the training dataset CSV",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
        help="Directory to save trained model artifacts",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["random_forest", "gradient_boosting"],
        choices=["random_forest", "gradient_boosting"],
        help="Which models to train and compare",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Enable GridSearchCV for hyperparameter tuning (slower)",
    )
    parser.add_argument(
        "--skip-dataset", action="store_true",
        help="Skip dataset generation and use existing CSV at --data-path",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the training pipeline CLI."""
    args = parse_args()
    use_synthetic = not args.real_repos

    logger.info("Starting training pipeline")
    logger.info("  Mode        : %s", "Synthetic" if use_synthetic else "Real repos")
    logger.info("  Models      : %s", args.models)
    logger.info("  CV folds    : %d", args.cv_folds)
    logger.info("  Grid search : %s", args.grid_search)

    # Step 1: Build or reuse dataset
    if not args.skip_dataset:
        logger.info("Building dataset ...")
        build_dataset(
            use_synthetic=use_synthetic,
            n_synthetic_samples=args.samples,
            buggy_ratio=args.buggy_ratio,
            output_path=args.data_path,
        )
    else:
        logger.info("Skipping dataset build — using existing: %s", args.data_path)

    if not args.data_path.exists():
        logger.error("Dataset not found at %s. Run without --skip-dataset first.", args.data_path)
        sys.exit(1)

    # Step 2: Train and compare models
    results = run_training_pipeline(
        data_path=args.data_path,
        model_dir=args.model_dir,
        model_names=args.models,
        cv_folds=args.cv_folds,
        use_grid_search=args.grid_search,
    )

    # Step 3: Print final report
    best_name = results.get("best_model", "unknown")
    logger.info("Best model selected: %s", best_name)

    if best_name in results:
        test_metrics = results[best_name].get("test_metrics", {})
        print("\n" + format_metrics_table(test_metrics))

    print(f"\nModel saved to: {args.model_dir / 'best_model.joblib'}")
    print("Training complete!")


if __name__ == "__main__":
    main()
