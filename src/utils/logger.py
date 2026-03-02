"""Centralized logging configuration for ml-code-analyzer."""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with consistent formatting.

    Args:
        name: The logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_file_logger(name: str, log_path: Path, level: int = logging.DEBUG) -> logging.Logger:
    """Create a logger that writes to both stdout and a file.

    Args:
        name: The logger name.
        log_path: Path to the log file (created if it doesn't exist).
        level: Logging level (default: DEBUG).

    Returns:
        Configured Logger instance with file and stream handlers.
    """
    logger = get_logger(name, level)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
