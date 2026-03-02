"""CLI entry point: run the full dataset collection pipeline.

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --real-repos --output data/real_dataset.csv
    python scripts/build_dataset.py --samples 5000 --buggy-ratio 0.4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.build_dataset import build_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset building."""
    parser = argparse.ArgumentParser(
        description="Build training dataset for the ML code analyzer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--real-repos", action="store_true",
        help="Mine real GitHub repositories (requires internet + git)",
    )
    parser.add_argument(
        "--samples", type=int, default=2500,
        help="Number of synthetic samples (used when --real-repos is not set)",
    )
    parser.add_argument(
        "--buggy-ratio", type=float, default=0.35,
        help="Fraction of samples to label as buggy",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/dataset.csv"),
        help="Output path for the dataset CSV",
    )
    parser.add_argument(
        "--max-files", type=int, default=300,
        help="Max Python files to process per repository",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the dataset builder CLI."""
    args = parse_args()
    use_synthetic = not args.real_repos

    logger.info("Building dataset: mode=%s", "synthetic" if use_synthetic else "real repos")

    dataset = build_dataset(
        use_synthetic=use_synthetic,
        n_synthetic_samples=args.samples,
        buggy_ratio=args.buggy_ratio,
        output_path=args.output,
        max_files_per_repo=args.max_files,
    )

    print(f"\nDataset saved to: {args.output}")
    print(f"  Total samples : {len(dataset)}")
    print(f"  Buggy         : {int(dataset['label'].sum())} ({dataset['label'].mean():.1%})")
    print(f"  Clean         : {int((dataset['label'] == 0).sum())} ({1 - dataset['label'].mean():.1%})")
    print(f"  Features      : {len(dataset.columns) - 1}")


if __name__ == "__main__":
    main()
