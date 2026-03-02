"""Code quality metrics: docstrings, type hints, identifiers, globals.

Extracts higher-level code quality signals that correlate with
maintainability and lower bug probability.
"""

import ast
import keyword
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _has_docstring(node: ast.AST) -> bool:
    """Check whether an AST node has a docstring as its first statement.

    Args:
        node: An AST node (Module, FunctionDef, AsyncFunctionDef, ClassDef).

    Returns:
        True if the first body statement is a string constant (docstring).
    """
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr):
        val = body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return True
    return False


def _count_type_hints(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
    """Count total arguments and how many have type annotations.

    Args:
        func_node: A function or async-function AST node.

    Returns:
        Tuple of (total_args, annotated_args).
    """
    args = func_node.args
    all_args = args.args + args.posonlyargs + args.kwonlyargs
    if args.vararg:
        all_args.append(args.vararg)
    if args.kwarg:
        all_args.append(args.kwarg)
    total = len(all_args)
    annotated = sum(1 for a in all_args if a.annotation is not None)
    return total, annotated


def extract_code_metrics(source_code: str) -> dict[str, Any]:
    """Extract code quality indicator features from Python source code.

    Analyzes docstring coverage, type hint coverage, identifier naming,
    global variable usage, nested functions, and star imports.

    Args:
        source_code: Raw Python source code as a string.

    Returns:
        Dictionary mapping quality feature names to numeric values.
        Sets 'syntax_error': 1 if parsing fails.
    """
    features: dict[str, Any] = {
        "has_docstrings": 0,
        "docstring_coverage": 0.0,
        "has_type_hints": 0,
        "type_hint_coverage": 0.0,
        "avg_identifier_length": 0.0,
        "num_global_variables": 0,
        "num_nested_functions": 0,
        "num_return_statements": 0,
        "uses_star_import": 0,
        "syntax_error": 0,
    }

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        logger.warning("Syntax error in code_metrics extraction: %s", exc)
        features["syntax_error"] = 1
        return features

    # Module-level docstring
    if _has_docstring(tree):
        features["has_docstrings"] = 1

    # Collect all functions and methods
    all_funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if all_funcs:
        funcs_with_docstrings = sum(1 for f in all_funcs if _has_docstring(f))
        features["docstring_coverage"] = round(
            funcs_with_docstrings / len(all_funcs), 4
        )
        if funcs_with_docstrings > 0 or _has_docstring(tree):
            features["has_docstrings"] = 1

        total_args = 0
        annotated_args = 0
        for func in all_funcs:
            t, a = _count_type_hints(func)
            total_args += t
            annotated_args += a

        if total_args > 0:
            features["type_hint_coverage"] = round(annotated_args / total_args, 4)
            features["has_type_hints"] = int(features["type_hint_coverage"] > 0)

    # Identifier lengths
    identifier_lengths: list[int] = []
    for node in ast.walk(tree):
        name: str | None = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.arg):
            name = node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name

        if name and not keyword.iskeyword(name) and len(name) > 1:
            identifier_lengths.append(len(name))

    if identifier_lengths:
        features["avg_identifier_length"] = round(
            sum(identifier_lengths) / len(identifier_lengths), 2
        )

    # Global variables (module-level assignments)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            features["num_global_variables"] += 1

    # Nested functions (function defined inside another function)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is not node and isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    features["num_nested_functions"] += 1

    # Return statements
    features["num_return_statements"] = sum(
        1 for n in ast.walk(tree) if isinstance(n, ast.Return)
    )

    # Star imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    features["uses_star_import"] = 1
                    break

    return features
