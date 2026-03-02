"""Cyclomatic complexity, nesting depth, and Halstead-style metrics.

Provides functions to compute control-flow complexity measures from Python
ASTs. These features are strong predictors of bug-prone code.
"""

import ast
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# AST node types that contribute to cyclomatic complexity
_COMPLEXITY_NODE_TYPES = (
    ast.If,
    ast.For,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.Assert,
    ast.comprehension,
)

# Boolean operators that increase decision paths
_BOOL_OPS = (ast.And, ast.Or)


def _count_cyclomatic_complexity(tree: ast.AST) -> int:
    """Count cyclomatic complexity for an AST.

    Complexity = 1 + number of decision points (if/elif/for/while/and/or/
    try-except/with/assert).

    Args:
        tree: Parsed AST module or function node.

    Returns:
        Integer cyclomatic complexity score.
    """
    complexity = 1
    for node in ast.walk(tree):
        if isinstance(node, _COMPLEXITY_NODE_TYPES):
            complexity += 1
        elif isinstance(node, ast.BoolOp) and isinstance(node.op, _BOOL_OPS):
            # Each additional operand beyond the first adds one path
            complexity += len(node.values) - 1
        elif isinstance(node, ast.IfExp):
            complexity += 1
    return complexity


def _max_nesting_depth(tree: ast.AST) -> int:
    """Calculate the maximum nesting depth of block statements.

    Traverses the AST and tracks the depth at which block-creating nodes
    appear (if/for/while/with/try/function/class definitions).

    Args:
        tree: Parsed AST module.

    Returns:
        Maximum nesting depth as an integer.
    """
    _BLOCK_NODES = (
        ast.If, ast.For, ast.While, ast.With, ast.Try,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.AsyncFor, ast.AsyncWith,
    )

    max_depth = [0]

    def _walk(node: ast.AST, depth: int) -> None:
        max_depth[0] = max(max_depth[0], depth)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _BLOCK_NODES):
                _walk(child, depth + 1)
            else:
                _walk(child, depth)

    _walk(tree, 0)
    return max_depth[0]


def _avg_nesting_depth(tree: ast.AST) -> float:
    """Compute the average nesting depth of all block statements.

    Args:
        tree: Parsed AST module.

    Returns:
        Average nesting depth rounded to 2 decimal places.
    """
    _BLOCK_NODES = (
        ast.If, ast.For, ast.While, ast.With, ast.Try,
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    )
    depths: list[int] = []

    def _walk(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _BLOCK_NODES):
                depths.append(depth + 1)
                _walk(child, depth + 1)
            else:
                _walk(child, depth)

    _walk(tree, 0)
    return round(sum(depths) / len(depths), 2) if depths else 0.0


def extract_complexity_features(source_code: str) -> dict[str, Any]:
    """Extract all complexity-related features from Python source code.

    Computes cyclomatic complexity, nesting depth statistics, and counts
    of specific control-flow constructs (branches, loops, try-except, etc.).

    Args:
        source_code: Raw Python source code as a string.

    Returns:
        Dictionary mapping complexity feature names to numeric values.
        Sets 'syntax_error': 1 if parsing fails.
    """
    features: dict[str, Any] = {
        "cyclomatic_complexity": 1,
        "max_nesting_depth": 0,
        "avg_nesting_depth": 0.0,
        "num_branches": 0,
        "num_loops": 0,
        "num_try_except": 0,
        "num_assertions": 0,
        "num_lambda_functions": 0,
        "syntax_error": 0,
    }

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        logger.warning("Syntax error in complexity extraction: %s", exc)
        features["syntax_error"] = 1
        return features

    features["cyclomatic_complexity"] = _count_cyclomatic_complexity(tree)
    features["max_nesting_depth"] = _max_nesting_depth(tree)
    features["avg_nesting_depth"] = _avg_nesting_depth(tree)

    for node in ast.walk(tree):
        if isinstance(node, (ast.If,)):
            features["num_branches"] += 1
        elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
            features["num_loops"] += 1
        elif isinstance(node, ast.Try):
            features["num_try_except"] += 1
        elif isinstance(node, ast.Assert):
            features["num_assertions"] += 1
        elif isinstance(node, ast.Lambda):
            features["num_lambda_functions"] += 1

    return features
