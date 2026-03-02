"""Orchestrate the full dataset collection and assembly pipeline.

This module coordinates the end-to-end dataset building process:
either mining real GitHub repos or falling back to synthetic data generation.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.synthetic_dataset import generate_synthetic_dataset, FEATURE_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_OUTPUT_PATH = Path("data/dataset.csv")


def build_dataset(
    use_synthetic: bool = True,
    repo_urls: Optional[list[str]] = None,
    n_synthetic_samples: int = 2500,
    buggy_ratio: float = 0.35,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    max_files_per_repo: int = 300,
) -> pd.DataFrame:
    """Build the training dataset using real repos or synthetic data.

    Tries to use real GitHub repositories first if use_synthetic=False.
    Falls back to synthetic generation if repo mining fails or is empty.

    Args:
        use_synthetic: If True, skip repo mining and use synthetic data.
        repo_urls: List of GitHub repo URLs to mine. Uses defaults if None.
        n_synthetic_samples: Number of synthetic samples to generate.
        buggy_ratio: Fraction of samples to label as buggy.
        output_path: Where to save the resulting CSV.
        max_files_per_repo: Max files to process per cloned repository.

    Returns:
        DataFrame with all features and a 'label' column (1=buggy, 0=clean).
    """
    dataset: Optional[pd.DataFrame] = None

    if not use_synthetic and repo_urls:
        logger.info("Attempting to build dataset from %d GitHub repos", len(repo_urls))
        try:
            from src.data.repo_miner import mine_repositories
            dataset = mine_repositories(
                repo_urls,
                max_files_per_repo=max_files_per_repo,
            )
        except Exception as exc:
            logger.warning("Repo mining failed (%s), falling back to synthetic data", exc)
            dataset = None

    if dataset is None or dataset.empty:
        if not use_synthetic:
            logger.info("Repo mining returned empty dataset. Using synthetic fallback.")
        else:
            logger.info("Using synthetic dataset generator.")

        dataset = generate_synthetic_dataset(
            n_samples=n_synthetic_samples,
            buggy_ratio=buggy_ratio,
            save_path=output_path,
        )
    else:
        # Clean and align columns
        feature_cols = FEATURE_NAMES + ["label"]
        for col in feature_cols:
            if col not in dataset.columns:
                dataset[col] = 0
        dataset = dataset[feature_cols].copy()
        dataset = dataset.fillna(0)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
        logger.info("Saved real dataset to %s", output_path)

    _log_dataset_stats(dataset)
    return dataset


def _log_dataset_stats(df: pd.DataFrame) -> None:
    """Log descriptive statistics about the assembled dataset.

    Args:
        df: The complete labeled dataset.
    """
    total = len(df)
    n_buggy = int(df["label"].sum())
    n_clean = total - n_buggy

    logger.info("=" * 50)
    logger.info("DATASET STATISTICS")
    logger.info("  Total samples : %d", total)
    logger.info("  Buggy files   : %d (%.1f%%)", n_buggy, 100 * n_buggy / max(total, 1))
    logger.info("  Clean files   : %d (%.1f%%)", n_clean, 100 * n_clean / max(total, 1))
    logger.info("  Features      : %d", len(df.columns) - 1)

    logger.info("Feature ranges (min / mean / max):")
    numeric_cols = [c for c in df.columns if c != "label"]
    for col in numeric_cols[:10]:
        col_data = df[col]
        logger.info(
            "  %-30s  %.2f / %.2f / %.2f",
            col,
            col_data.min(),
            col_data.mean(),
            col_data.max(),
        )
    logger.info("=" * 50)
