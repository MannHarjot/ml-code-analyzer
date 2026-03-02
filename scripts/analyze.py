"""CLI tool: analyze a single Python file or an entire directory.

Usage:
    python scripts/analyze.py --file path/to/file.py
    python scripts/analyze.py --dir path/to/project/
    python scripts/analyze.py --file my.py --output json
    python scripts/analyze.py --dir src/ --threshold 0.5 --output csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.ast_extractor import extract_all_features
from src.models.classifier import load_model
from src.data.synthetic_dataset import FEATURE_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_DIR = Path("models/pretrained")
RISK_LABELS = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "critical": "Critical",
}


def risk_level(score: float, thresholds: dict) -> str:
    """Map a continuous risk score to a categorical risk level.

    Args:
        score: Float risk score between 0.0 and 1.0.
        thresholds: Dict with 'low', 'medium', 'high' float thresholds.

    Returns:
        One of: 'Low', 'Medium', 'High', 'Critical'.
    """
    if score < thresholds.get("low", 0.3):
        return "Low"
    if score < thresholds.get("medium", 0.6):
        return "Medium"
    if score < thresholds.get("high", 0.8):
        return "High"
    return "Critical"


def analyze_file(
    file_path: Path,
    model,
    feature_names: list[str],
    threshold: float = 0.6,
) -> dict:
    """Extract features and predict risk for a single Python file.

    Args:
        file_path: Path to the Python source file.
        model: Loaded scikit-learn classifier.
        feature_names: Ordered list of feature names the model expects.
        threshold: Risk score threshold for flagging a file (informational).

    Returns:
        Dictionary with file_path, risk_score, risk_level, top_factors, features.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        logger.error("Cannot read file %s: %s", file_path, exc)
        return {"file_path": str(file_path), "error": str(exc)}

    features = extract_all_features(source)

    # Build feature vector aligned to model's expected order
    x = np.array(
        [float(features.get(name, 0.0)) for name in feature_names],
        dtype=np.float64,
    ).reshape(1, -1)

    score = float(model.predict_proba(x)[0][1])
    level = risk_level(score, {"low": 0.3, "medium": 0.6, "high": 0.8})

    # Top risk factors: features with highest importance that are above typical values
    importances = model.feature_importances_ if hasattr(model, "feature_importances_") else []
    top_factors = []
    if len(importances) == len(feature_names):
        ranked = sorted(
            zip(feature_names, importances, [features.get(n, 0) for n in feature_names]),
            key=lambda t: t[1],
            reverse=True,
        )
        top_factors = [
            {"feature": name, "value": val, "importance": float(imp)}
            for name, imp, val in ranked[:3]
        ]

    return {
        "file_path": str(file_path),
        "risk_score": round(score, 4),
        "risk_level": level,
        "flagged": score >= threshold,
        "top_factors": top_factors,
        "features": {k: v for k, v in features.items() if k != "syntax_error"},
    }


def print_table(results: list[dict], threshold: float) -> None:
    """Print analysis results as a formatted ASCII table.

    Args:
        results: List of result dicts from analyze_file.
        threshold: Threshold used for flagging.
    """
    RISK_COLOR = {
        "Low": "\033[92m",
        "Medium": "\033[93m",
        "High": "\033[91m",
        "Critical": "\033[31m",
    }
    RESET = "\033[0m"

    print("\n" + "=" * 80)
    print(f"{'FILE':<40} {'SCORE':>7}  {'LEVEL':<10} {'FACTORS'}")
    print("=" * 80)

    for r in sorted(results, key=lambda x: x.get("risk_score", 0), reverse=True):
        if "error" in r:
            print(f"{str(Path(r['file_path']).name):<40}  ERROR: {r['error']}")
            continue
        name = Path(r["file_path"]).name[:39]
        score = r["risk_score"]
        level = r["risk_level"]
        color = RISK_COLOR.get(level, "")
        factors = ", ".join(f["feature"] for f in r.get("top_factors", []))
        print(f"{name:<40} {score:>7.4f}  {color}{level:<10}{RESET} {factors}")

    flagged = sum(1 for r in results if r.get("flagged"))
    print("=" * 80)
    print(f"Analyzed: {len(results)} files  |  Flagged (>{threshold:.0%}): {flagged}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ML-powered Python code quality analyzer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Path to a single .py file")
    group.add_argument("--dir", type=Path, help="Directory to analyze recursively")

    parser.add_argument(
        "--model-path", type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Path to model directory (containing best_model.joblib)",
    )
    parser.add_argument(
        "--output", choices=["table", "json", "csv"], default="table",
        help="Output format",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.6,
        help="Risk score threshold for flagging files (0.0–1.0)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the CLI analysis tool."""
    args = parse_args()

    # Load model
    try:
        model, feature_names, metrics = load_model(args.model_path)
    except FileNotFoundError:
        logger.error(
            "No trained model found at '%s'. Run: python scripts/train.py first.",
            args.model_path,
        )
        sys.exit(1)

    if not feature_names:
        feature_names = FEATURE_NAMES

    # Collect files to analyze
    files: list[Path] = []
    if args.file:
        if not args.file.exists():
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        files = [args.file]
    elif args.dir:
        if not args.dir.is_dir():
            logger.error("Directory not found: %s", args.dir)
            sys.exit(1)
        files = sorted(args.dir.rglob("*.py"))
        if not files:
            logger.error("No Python files found in %s", args.dir)
            sys.exit(1)

    logger.info("Analyzing %d file(s) ...", len(files))

    results = [
        analyze_file(f, model, feature_names, threshold=args.threshold)
        for f in files
    ]

    # Output
    if args.output == "table":
        print_table(results, args.threshold)
    elif args.output == "json":
        print(json.dumps(results, indent=2))
    elif args.output == "csv":
        import csv
        import io
        output = io.StringIO()
        if results:
            flat_rows = []
            for r in results:
                row = {
                    "file_path": r.get("file_path", ""),
                    "risk_score": r.get("risk_score", ""),
                    "risk_level": r.get("risk_level", ""),
                    "flagged": r.get("flagged", ""),
                }
                row.update(r.get("features", {}))
                flat_rows.append(row)
            writer = csv.DictWriter(output, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)
            print(output.getvalue())


if __name__ == "__main__":
    main()
