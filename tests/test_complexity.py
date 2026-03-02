"""Tests for cyclomatic complexity and nesting depth calculations."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.complexity import extract_complexity_features

SIMPLE_CODE = "x = 1"

LINEAR_CODE = """
def foo(a, b):
    x = a + b
    return x
"""

ONE_IF = """
def foo(x):
    if x > 0:
        return x
    return -x
"""

NESTED_CODE = """
def process(data):
    for item in data:
        if item > 0:
            for sub in item:
                if sub > 10:
                    pass
"""

EXCEPTION_CODE = """
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        result = 0
    except ValueError:
        result = -1
    return result
"""

LOOPS_CODE = """
def count():
    for i in range(10):
        pass
    while True:
        break
    for j in range(5):
        while j > 0:
            j -= 1
"""


class TestCyclomaticComplexity:
    def test_simple_assignment_has_base_complexity(self):
        features = extract_complexity_features(SIMPLE_CODE)
        assert features["cyclomatic_complexity"] == 1

    def test_linear_function_complexity(self):
        features = extract_complexity_features(LINEAR_CODE)
        assert features["cyclomatic_complexity"] == 1

    def test_single_if_increases_complexity(self):
        features = extract_complexity_features(ONE_IF)
        assert features["cyclomatic_complexity"] > 1

    def test_nested_conditions_add_complexity(self):
        simple = extract_complexity_features(LINEAR_CODE)["cyclomatic_complexity"]
        nested = extract_complexity_features(NESTED_CODE)["cyclomatic_complexity"]
        assert nested > simple

    def test_try_except_adds_complexity(self):
        features = extract_complexity_features(EXCEPTION_CODE)
        assert features["cyclomatic_complexity"] > 2
        assert features["num_try_except"] == 1


class TestNestingDepth:
    def test_no_nesting_returns_zero(self):
        features = extract_complexity_features("x = 1\ny = 2")
        assert features["max_nesting_depth"] == 0

    def test_single_if_is_depth_one(self):
        features = extract_complexity_features(ONE_IF)
        assert features["max_nesting_depth"] >= 1

    def test_nested_code_depth(self):
        features = extract_complexity_features(NESTED_CODE)
        assert features["max_nesting_depth"] >= 3

    def test_avg_nesting_less_than_max(self):
        features = extract_complexity_features(NESTED_CODE)
        assert features["avg_nesting_depth"] <= features["max_nesting_depth"]


class TestCounters:
    def test_loop_counting(self):
        features = extract_complexity_features(LOOPS_CODE)
        assert features["num_loops"] >= 4

    def test_branch_counting(self):
        code = """
if a:
    pass
elif b:
    pass
if c:
    pass
"""
        features = extract_complexity_features(code)
        assert features["num_branches"] >= 2

    def test_assertion_counting(self):
        code = "assert x > 0\nassert y != 0\nassert z is not None"
        features = extract_complexity_features(code)
        assert features["num_assertions"] == 3

    def test_lambda_counting(self):
        code = "f = lambda x: x\ng = lambda x, y: x + y"
        features = extract_complexity_features(code)
        assert features["num_lambda_functions"] == 2

    def test_syntax_error_returns_defaults(self):
        features = extract_complexity_features("def broken(:")
        assert features["syntax_error"] == 1
        assert features["cyclomatic_complexity"] == 1
