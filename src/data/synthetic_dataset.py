"""Synthetic dataset generator for training without internet access.

Generates a realistic training dataset by sampling feature vectors from
distributions derived from real-world code analysis studies. Buggy code
patterns are modeled with higher complexity, lower docstring coverage,
and deeper nesting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Random state for reproducibility
RNG = np.random.default_rng(42)

# Feature names in order
FEATURE_NAMES = [
    "total_lines", "code_lines", "blank_lines", "comment_lines",
    "comment_ratio", "num_functions", "num_classes", "num_methods",
    "avg_function_length", "max_function_length", "num_imports",
    "num_unique_imports", "cyclomatic_complexity", "max_nesting_depth",
    "avg_nesting_depth", "num_branches", "num_loops", "num_try_except",
    "num_assertions", "num_lambda_functions", "has_docstrings",
    "docstring_coverage", "has_type_hints", "type_hint_coverage",
    "avg_identifier_length", "num_global_variables", "num_nested_functions",
    "num_return_statements", "uses_star_import", "syntax_error",
]


def _sample_clean(n: int) -> pd.DataFrame:
    """Sample feature vectors for clean (low-risk) code files.

    Args:
        n: Number of samples to generate.

    Returns:
        DataFrame of clean code feature vectors.
    """
    data: dict[str, np.ndarray] = {}

    data["total_lines"] = RNG.integers(50, 300, n).astype(float)
    data["code_lines"] = (data["total_lines"] * RNG.uniform(0.55, 0.75, n)).astype(float)
    data["blank_lines"] = (data["total_lines"] * RNG.uniform(0.1, 0.2, n)).astype(float)
    data["comment_lines"] = (data["total_lines"] * RNG.uniform(0.1, 0.25, n)).astype(float)
    data["comment_ratio"] = data["comment_lines"] / data["total_lines"]
    data["num_functions"] = RNG.integers(2, 15, n).astype(float)
    data["num_classes"] = RNG.integers(0, 4, n).astype(float)
    data["num_methods"] = (data["num_classes"] * RNG.uniform(2, 8, n)).astype(float)
    data["avg_function_length"] = RNG.uniform(5, 25, n)
    data["max_function_length"] = data["avg_function_length"] * RNG.uniform(1.5, 3.0, n)
    data["num_imports"] = RNG.integers(2, 12, n).astype(float)
    data["num_unique_imports"] = (data["num_imports"] * RNG.uniform(0.7, 1.0, n)).astype(float)
    data["cyclomatic_complexity"] = RNG.integers(1, 12, n).astype(float)
    data["max_nesting_depth"] = RNG.integers(1, 4, n).astype(float)
    data["avg_nesting_depth"] = data["max_nesting_depth"] * RNG.uniform(0.4, 0.8, n)
    data["num_branches"] = RNG.integers(0, 10, n).astype(float)
    data["num_loops"] = RNG.integers(0, 6, n).astype(float)
    data["num_try_except"] = RNG.integers(0, 4, n).astype(float)
    data["num_assertions"] = RNG.integers(0, 5, n).astype(float)
    data["num_lambda_functions"] = RNG.integers(0, 3, n).astype(float)
    data["has_docstrings"] = RNG.choice([0, 1], n, p=[0.2, 0.8]).astype(float)
    data["docstring_coverage"] = np.clip(RNG.normal(0.72, 0.18, n), 0.0, 1.0)
    data["has_type_hints"] = RNG.choice([0, 1], n, p=[0.35, 0.65]).astype(float)
    data["type_hint_coverage"] = np.clip(RNG.normal(0.60, 0.22, n), 0.0, 1.0)
    data["avg_identifier_length"] = np.clip(RNG.normal(9.5, 2.5, n), 3, 20)
    data["num_global_variables"] = RNG.integers(0, 3, n).astype(float)
    data["num_nested_functions"] = RNG.integers(0, 2, n).astype(float)
    data["num_return_statements"] = RNG.integers(1, 12, n).astype(float)
    data["uses_star_import"] = RNG.choice([0, 1], n, p=[0.95, 0.05]).astype(float)
    data["syntax_error"] = np.zeros(n, dtype=float)

    return pd.DataFrame(data, columns=FEATURE_NAMES)


def _sample_buggy(n: int) -> pd.DataFrame:
    """Sample feature vectors for buggy (high-risk) code files.

    Models patterns common in bug-prone code: higher complexity, deeper
    nesting, poor documentation, and more global state.

    Args:
        n: Number of samples to generate.

    Returns:
        DataFrame of buggy code feature vectors.
    """
    data: dict[str, np.ndarray] = {}

    data["total_lines"] = RNG.integers(80, 600, n).astype(float)
    data["code_lines"] = (data["total_lines"] * RNG.uniform(0.7, 0.9, n)).astype(float)
    data["blank_lines"] = (data["total_lines"] * RNG.uniform(0.03, 0.1, n)).astype(float)
    data["comment_lines"] = (data["total_lines"] * RNG.uniform(0.0, 0.08, n)).astype(float)
    data["comment_ratio"] = data["comment_lines"] / data["total_lines"]
    data["num_functions"] = RNG.integers(1, 25, n).astype(float)
    data["num_classes"] = RNG.integers(0, 6, n).astype(float)
    data["num_methods"] = (data["num_classes"] * RNG.uniform(3, 12, n)).astype(float)
    data["avg_function_length"] = RNG.uniform(20, 80, n)
    data["max_function_length"] = data["avg_function_length"] * RNG.uniform(2.0, 5.0, n)
    data["num_imports"] = RNG.integers(5, 25, n).astype(float)
    data["num_unique_imports"] = (data["num_imports"] * RNG.uniform(0.5, 0.9, n)).astype(float)
    data["cyclomatic_complexity"] = RNG.integers(10, 50, n).astype(float)
    data["max_nesting_depth"] = RNG.integers(4, 10, n).astype(float)
    data["avg_nesting_depth"] = data["max_nesting_depth"] * RNG.uniform(0.5, 0.9, n)
    data["num_branches"] = RNG.integers(8, 40, n).astype(float)
    data["num_loops"] = RNG.integers(3, 20, n).astype(float)
    data["num_try_except"] = RNG.integers(0, 8, n).astype(float)
    data["num_assertions"] = RNG.integers(0, 3, n).astype(float)
    data["num_lambda_functions"] = RNG.integers(0, 8, n).astype(float)
    data["has_docstrings"] = RNG.choice([0, 1], n, p=[0.65, 0.35]).astype(float)
    data["docstring_coverage"] = np.clip(RNG.normal(0.18, 0.18, n), 0.0, 1.0)
    data["has_type_hints"] = RNG.choice([0, 1], n, p=[0.75, 0.25]).astype(float)
    data["type_hint_coverage"] = np.clip(RNG.normal(0.12, 0.15, n), 0.0, 1.0)
    data["avg_identifier_length"] = np.clip(RNG.normal(5.5, 2.5, n), 1, 15)
    data["num_global_variables"] = RNG.integers(2, 15, n).astype(float)
    data["num_nested_functions"] = RNG.integers(1, 10, n).astype(float)
    data["num_return_statements"] = RNG.integers(0, 20, n).astype(float)
    data["uses_star_import"] = RNG.choice([0, 1], n, p=[0.6, 0.4]).astype(float)
    data["syntax_error"] = RNG.choice([0, 1], n, p=[0.85, 0.15]).astype(float)

    return pd.DataFrame(data, columns=FEATURE_NAMES)


def generate_synthetic_dataset(
    n_samples: int = 2500,
    buggy_ratio: float = 0.35,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate a complete synthetic training dataset.

    Creates a balanced dataset with clean and buggy code samples drawn
    from realistic feature distributions.

    Args:
        n_samples: Total number of samples to generate.
        buggy_ratio: Fraction of samples labeled as buggy (0 < buggy_ratio < 1).
        save_path: If provided, save the CSV to this path.

    Returns:
        DataFrame with feature columns plus 'label' (1=buggy, 0=clean).
    """
    n_buggy = int(n_samples * buggy_ratio)
    n_clean = n_samples - n_buggy

    logger.info("Generating synthetic dataset: %d clean, %d buggy samples", n_clean, n_buggy)

    clean_df = _sample_clean(n_clean)
    clean_df["label"] = 0

    buggy_df = _sample_buggy(n_buggy)
    buggy_df["label"] = 1

    dataset = pd.concat([clean_df, buggy_df], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "Dataset stats: total=%d, buggy=%d (%.1f%%), clean=%d (%.1f%%)",
        len(dataset),
        dataset["label"].sum(),
        dataset["label"].mean() * 100,
        (dataset["label"] == 0).sum(),
        (1 - dataset["label"].mean()) * 100,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(save_path, index=False)
        logger.info("Saved synthetic dataset to %s", save_path)

    return dataset
