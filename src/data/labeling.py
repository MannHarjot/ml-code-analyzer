"""Label Python files as buggy or clean based on git commit history.

A file is labeled 'buggy' if it was modified in a bug-fix commit within the
repo's history. Bug-fix commits are identified by keywords in commit messages.
Files untouched by bug fixes for 30+ days are labeled 'clean'.
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Patterns that indicate a bug-fix commit
BUG_FIX_PATTERNS = [
    "fix", "bug", "patch", "error", "issue", "crash",
    "resolve", "hotfix", "repair", "workaround", "incorrect",
]

STABLE_DAYS_THRESHOLD = 30


def get_bug_fix_commits(repo_path: Path) -> list[str]:
    """Retrieve all commit hashes that look like bug-fix commits.

    Searches commit messages for keywords associated with bug fixes.

    Args:
        repo_path: Path to the local git repository.

    Returns:
        List of commit hashes (short form) for bug-fix commits.
    """
    pattern = "|".join(BUG_FIX_PATTERNS)
    try:
        result = subprocess.run(
            [
                "git", "log", "--oneline",
                f"--grep={pattern}",
                "--regexp-ignore-case",
                "--format=%H",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        logger.info("Found %d bug-fix commits in %s", len(commits), repo_path.name)
        return commits
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
        logger.warning("Failed to get bug-fix commits: %s", exc)
        return []


def get_files_changed_in_commit(repo_path: Path, commit_hash: str) -> list[str]:
    """Get the list of Python files changed in a specific commit.

    Args:
        repo_path: Path to the local git repository.
        commit_hash: The commit hash to inspect.

    Returns:
        List of Python file paths (relative to repo root) changed in the commit.
    """
    try:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        files = [
            f.strip() for f in result.stdout.splitlines()
            if f.strip().endswith(".py")
        ]
        return files
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as exc:
        logger.warning("Failed to get files for commit %s: %s", commit_hash[:8], exc)
        return []


def get_last_modification_date(repo_path: Path, file_path: str) -> Optional[datetime]:
    """Get the date of the most recent commit that modified a file.

    Args:
        repo_path: Path to the local git repository.
        file_path: Relative path to the file within the repo.

    Returns:
        datetime of the latest modification, or None on error.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", "--", file_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        date_str = result.stdout.strip()
        if date_str:
            return datetime.fromisoformat(date_str)
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
        pass
    return None


def label_repository_files(
    repo_path: Path,
    max_files: int = 500,
) -> list[dict]:
    """Label all Python files in a repository as buggy or clean.

    Files changed in bug-fix commits are labeled buggy. Files stable for
    STABLE_DAYS_THRESHOLD or more days are labeled clean. Files that don't
    clearly fit either category are skipped.

    Args:
        repo_path: Path to the local git repository root.
        max_files: Maximum number of files to process (to limit memory usage).

    Returns:
        List of dicts with keys: 'file_path', 'label', 'repo'.
    """
    logger.info("Labeling files in repo: %s", repo_path.name)

    bug_fix_commits = get_bug_fix_commits(repo_path)

    # Collect all files touched by bug-fix commits
    buggy_files: set[str] = set()
    for commit in bug_fix_commits[:200]:  # Limit to first 200 to avoid timeout
        changed = get_files_changed_in_commit(repo_path, commit)
        buggy_files.update(changed)

    logger.info("Found %d unique files touched in bug-fix commits", len(buggy_files))

    # Walk all Python files
    all_py_files = list(repo_path.rglob("*.py"))[:max_files]
    cutoff_date = datetime.now().astimezone() - timedelta(days=STABLE_DAYS_THRESHOLD)

    labeled: list[dict] = []
    for py_file in all_py_files:
        rel_path = str(py_file.relative_to(repo_path))

        if rel_path in buggy_files:
            labeled.append({
                "file_path": str(py_file),
                "label": 1,
                "repo": repo_path.name,
            })
        else:
            # Check if file is stable (not recently modified)
            last_mod = get_last_modification_date(repo_path, rel_path)
            if last_mod is not None and last_mod < cutoff_date:
                labeled.append({
                    "file_path": str(py_file),
                    "label": 0,
                    "repo": repo_path.name,
                })

    logger.info(
        "Labeled %d files: %d buggy, %d clean",
        len(labeled),
        sum(1 for l in labeled if l["label"] == 1),
        sum(1 for l in labeled if l["label"] == 0),
    )
    return labeled
