"""Core AST parsing and feature extraction for Python source code.

This module provides the primary interface for extracting structured features
from Python source code using the built-in ast module. Features cover structural
properties, complexity measures, and code quality indicators.
"""

import ast
import tokenize
import io
from typing import Any

from src.features.complexity import extract_complexity_features
from src.features.code_metrics import extract_code_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_structural_features(source_code: str) -> dict[str, Any]:
    """Extract structural AST-level features from Python source code.

    Parses the source code into an AST and walks it to count structural
    elements such as functions, classes, imports, and compute line-level
    statistics.

    Args:
        source_code: Raw Python source code as a string.

    Returns:
        Dictionary mapping feature names to numeric values. Includes
        'syntax_error' flag (1 if parsing failed, 0 otherwise).
    """
    features: dict[str, Any] = {
        "num_functions": 0,
        "num_classes": 0,
        "num_methods": 0,
        "avg_function_length": 0.0,
        "max_function_length": 0,
        "num_imports": 0,
        "num_unique_imports": 0,
        "syntax_error": 0,
    }

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        logger.warning("Syntax error during AST parse: %s", exc)
        features["syntax_error"] = 1
        return features

    import_names: set[str] = set()
    function_lengths: list[int] = []
    class_stack: list[bool] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            features["num_classes"] += 1

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_method = _is_method(node, tree)
            if is_method:
                features["num_methods"] += 1
            else:
                features["num_functions"] += 1
            length = _function_length(node)
            function_lengths.append(length)

        elif isinstance(node, ast.Import):
            features["num_imports"] += len(node.names)
            for alias in node.names:
                import_names.add(alias.name.split(".")[0])

        elif isinstance(node, ast.ImportFrom):
            features["num_imports"] += 1
            if node.module:
                import_names.add(node.module.split(".")[0])

    features["num_unique_imports"] = len(import_names)

    if function_lengths:
        features["avg_function_length"] = round(
            sum(function_lengths) / len(function_lengths), 2
        )
        features["max_function_length"] = max(function_lengths)

    return features


def _is_method(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    tree: ast.AST,
) -> bool:
    """Check if a function node is a method inside a class definition.

    Args:
        func_node: The function node to check.
        tree: The full module AST.

    Returns:
        True if the function is defined directly inside a class body.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in ast.walk(node):
                if item is func_node:
                    return True
    return False


def _function_length(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Compute the line length of a function definition.

    Args:
        func_node: The function AST node.

    Returns:
        Number of lines occupied by the function body.
    """
    if not (hasattr(func_node, "lineno") and hasattr(func_node, "end_lineno")):
        return 0
    return (func_node.end_lineno or func_node.lineno) - func_node.lineno


def extract_line_features(source_code: str) -> dict[str, Any]:
    """Extract line-level features from source code using tokenization.

    Counts total lines, code lines, blank lines, and comment lines. Comment
    ratio is computed as comment_lines / max(total_lines, 1).

    Args:
        source_code: Raw Python source code as a string.

    Returns:
        Dictionary with line-level feature values.
    """
    lines = source_code.splitlines()
    total_lines = len(lines)
    blank_lines = sum(1 for l in lines if l.strip() == "")

    comment_lines = 0
    try:
        tokens = list(
            tokenize.generate_tokens(io.StringIO(source_code).readline)
        )
        comment_lines = sum(1 for tok in tokens if tok.type == tokenize.COMMENT)
    except tokenize.TokenError:
        # Fallback: count lines starting with #
        comment_lines = sum(
            1 for l in lines if l.strip().startswith("#")
        )

    code_lines = total_lines - blank_lines - comment_lines
    comment_ratio = round(comment_lines / max(total_lines, 1), 4)

    return {
        "total_lines": total_lines,
        "code_lines": max(code_lines, 0),
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "comment_ratio": comment_ratio,
    }


def extract_all_features(source_code: str) -> dict[str, Any]:
    """Extract all features from Python source code in a single flat dictionary.

    Combines structural features, complexity metrics, and code quality
    indicators into one flat dictionary suitable for ML model input.

    Args:
        source_code: Raw Python source code as a string.

    Returns:
        Flat dictionary of feature_name -> numeric_value for all 30+ features.
        Always returns a complete dictionary even on syntax errors, with
        'syntax_error' flag set to 1.

    Example:
        >>> features = extract_all_features("def foo(): pass")
        >>> features["num_functions"]
        1
    """
    line_features = extract_line_features(source_code)
    structural_features = extract_structural_features(source_code)
    complexity_features = extract_complexity_features(source_code)
    code_quality_features = extract_code_metrics(source_code)

    all_features: dict[str, Any] = {}
    all_features.update(line_features)
    all_features.update(structural_features)
    all_features.update(complexity_features)
    all_features.update(code_quality_features)

    # Resolve syntax_error: take max across modules that independently detect it
    all_features["syntax_error"] = max(
        structural_features.get("syntax_error", 0),
        complexity_features.get("syntax_error", 0),
        code_quality_features.get("syntax_error", 0),
    )

    return all_features
