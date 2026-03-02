"""GitHub repository mining pipeline for dataset collection.

Clones repositories, extracts AST features from all Python files, and
combines labeled data into a dataset ready for model training.
"""

import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features.ast_extractor import extract_all_features
from src.data.labeling import label_repository_files
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_REPOS = [
    "https://github.com/pallets/flask",
    "https://github.com/psf/requests",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/fastapi/fastapi",
    "https://github.com/pandas-dev/pandas",
]


def clone_repository(repo_url: str, target_dir: Path) -> Optional[Path]:
    """Clone a GitHub repository to a local directory.

    Uses a shallow clone (depth=1 for history, full for log analysis).
    Falls back to full clone if shallow fails.

    Args:
        repo_url: HTTPS URL of the GitHub repository.
        target_dir: Directory where the repo will be cloned.

    Returns:
        Path to the cloned repository, or None if cloning failed.
    """
    repo_name = repo_url.rstrip("/").split("/")[-1]
    clone_path = target_dir / repo_name

    if clone_path.exists():
        logger.info("Repository already exists, skipping clone: %s", repo_name)
        return clone_path

    logger.info("Cloning repository: %s", repo_url)
    try:
        subprocess.run(
            ["git", "clone", "--depth=100", repo_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        logger.info("Successfully cloned: %s", repo_name)
        return clone_path
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to clone %s: %s", repo_url, exc.stderr[:200])
        return None
    except subprocess.TimeoutExpired:
        logger.error("Clone timed out for: %s", repo_url)
        if clone_path.exists():
            shutil.rmtree(clone_path, ignore_errors=True)
        return None


def extract_features_from_file(file_path: Path) -> Optional[dict]:
    """Read and extract features from a single Python file.

    Args:
        file_path: Path to the Python file to analyze.

    Returns:
        Feature dictionary, or None if the file cannot be read.
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        features = extract_all_features(source)
        features["file_path"] = str(file_path)
        return features
    except Exception as exc:
        logger.debug("Could not process file %s: %s", file_path, exc)
        return None


def mine_repository(
    repo_url: str,
    work_dir: Path,
    max_files: int = 300,
) -> pd.DataFrame:
    """Clone a repository, extract features, and label files.

    Args:
        repo_url: HTTPS URL of the GitHub repository.
        work_dir: Working directory for cloning (temp storage).
        max_files: Maximum files to process per repository.

    Returns:
        DataFrame with features and labels for all processed files.
    """
    repo_path = clone_repository(repo_url, work_dir)
    if repo_path is None:
        return pd.DataFrame()

    labeled_files = label_repository_files(repo_path, max_files=max_files)
    if not labeled_files:
        logger.warning("No labeled files found in %s", repo_path.name)
        return pd.DataFrame()

    rows: list[dict] = []
    for item in labeled_files:
        file_path = Path(item["file_path"])
        features = extract_features_from_file(file_path)
        if features is not None:
            features["label"] = item["label"]
            features["repo"] = item["repo"]
            rows.append(features)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(
        "Mined %d samples from %s (buggy=%d, clean=%d)",
        len(df),
        repo_path.name,
        int(df["label"].sum()),
        int((df["label"] == 0).sum()),
    )
    return df


def mine_repositories(
    repo_urls: list[str],
    work_dir: Optional[Path] = None,
    cleanup: bool = True,
    max_files_per_repo: int = 300,
) -> pd.DataFrame:
    """Mine multiple repositories and combine into one dataset.

    Args:
        repo_urls: List of GitHub repository HTTPS URLs.
        work_dir: Directory for temp storage. Uses system temp if None.
        cleanup: Whether to delete cloned repos after processing.
        max_files_per_repo: File limit per repository.

    Returns:
        Combined DataFrame from all repositories.
    """
    tmp_dir: Optional[Path] = None
    if work_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="ml_code_analyzer_repos_"))
        work_dir = tmp_dir

    work_dir.mkdir(parents=True, exist_ok=True)
    all_frames: list[pd.DataFrame] = []

    try:
        for url in repo_urls:
            df = mine_repository(url, work_dir, max_files=max_files_per_repo)
            if not df.empty:
                all_frames.append(df)
    finally:
        if cleanup and tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info("Cleaned up temp repos directory")

    if not all_frames:
        logger.warning("No data collected from any repository")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(
        "Combined dataset: %d total samples from %d repos",
        len(combined), len(all_frames)
    )
    return combined
