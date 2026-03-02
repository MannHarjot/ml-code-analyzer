"""Tests for the AST feature extraction engine."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.ast_extractor import (
    extract_all_features,
    extract_structural_features,
    extract_line_features,
)

SIMPLE_FUNCTION = '''
def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        Sum of a and b.
    """
    return a + b
'''

COMPLEX_CODE = '''
import os
import sys
from pathlib import Path
from collections import *

x = 10
y = 20
z = 30

def outer():
    def inner():
        def innermost():
            if True:
                for i in range(10):
                    while True:
                        pass
    inner()

class MyClass:
    def method_one(self):
        pass

    def method_two(self):
        pass
'''

BUGGY_CODE = '''
from os import *
from sys import *

GLOB1 = []
GLOB2 = {}
GLOB3 = None
GLOB4 = 0
GLOB5 = ""

def f(a,b,c,d,e,f,g):
    if a:
        if b:
            if c:
                if d:
                    for i in range(a):
                        for j in range(b):
                            if i and j:
                                pass
    return a,b,c

lam1 = lambda x,y,z: x+y+z
lam2 = lambda p,q: p*q
lam3 = lambda n: n**2
'''


class TestExtractLineFeatures:
    def test_counts_lines_correctly(self):
        code = "x = 1\n# comment\n\ny = 2\n"
        features = extract_line_features(code)
        assert features["total_lines"] == 4
        assert features["blank_lines"] == 1
        assert features["comment_lines"] == 1

    def test_comment_ratio_correct(self):
        code = "# a\n# b\nx = 1\ny = 2\n"
        features = extract_line_features(code)
        assert features["comment_ratio"] == pytest.approx(0.5, abs=0.01)

    def test_empty_code(self):
        features = extract_line_features("")
        assert features["total_lines"] == 0
        assert features["comment_ratio"] == 0.0


class TestExtractStructuralFeatures:
    def test_counts_functions(self):
        features = extract_structural_features(SIMPLE_FUNCTION)
        assert features["num_functions"] == 1
        assert features["syntax_error"] == 0

    def test_counts_classes_and_methods(self):
        code = '''
class Foo:
    def bar(self): pass
    def baz(self): pass
'''
        features = extract_structural_features(code)
        assert features["num_classes"] == 1
        assert features["num_methods"] == 2
        assert features["num_functions"] == 0

    def test_counts_imports(self):
        code = "import os\nimport sys\nfrom pathlib import Path\n"
        features = extract_structural_features(code)
        assert features["num_imports"] >= 3

    def test_syntax_error_flag(self):
        features = extract_structural_features("def (broken:")
        assert features["syntax_error"] == 1

    def test_function_length(self):
        features = extract_structural_features(SIMPLE_FUNCTION)
        assert features["max_function_length"] > 0
        assert features["avg_function_length"] > 0


class TestExtractAllFeatures:
    def test_returns_all_keys(self):
        features = extract_all_features(SIMPLE_FUNCTION)
        required_keys = [
            "total_lines", "num_functions", "cyclomatic_complexity",
            "max_nesting_depth", "docstring_coverage", "uses_star_import",
            "syntax_error",
        ]
        for key in required_keys:
            assert key in features, f"Missing feature: {key}"

    def test_clean_code_has_docstrings(self):
        features = extract_all_features(SIMPLE_FUNCTION)
        assert features["has_docstrings"] == 1
        assert features["docstring_coverage"] == pytest.approx(1.0, abs=0.01)

    def test_star_import_detected(self):
        features = extract_all_features(BUGGY_CODE)
        assert features["uses_star_import"] == 1

    def test_lambda_count(self):
        features = extract_all_features(BUGGY_CODE)
        assert features["num_lambda_functions"] == 3

    def test_global_variables_counted(self):
        features = extract_all_features(BUGGY_CODE)
        assert features["num_global_variables"] >= 5

    def test_nested_functions(self):
        features = extract_all_features(COMPLEX_CODE)
        assert features["num_nested_functions"] >= 2

    def test_syntax_error_graceful(self):
        features = extract_all_features("this is not python !!!")
        assert features["syntax_error"] == 1
        # Must still return a complete dict
        assert "total_lines" in features

    def test_type_hints_detected(self):
        features = extract_all_features(SIMPLE_FUNCTION)
        assert features["has_type_hints"] == 1
        assert features["type_hint_coverage"] > 0

    def test_nesting_depth(self):
        features = extract_all_features(COMPLEX_CODE)
        assert features["max_nesting_depth"] >= 3
