"""Omit policy: what the rewriter wraps depends on whether the result is bound."""
import ast

import pytest

from tracker.ast_rewriter import ProvTrackTransformer, _is_omitted


def _wrapped(src: str) -> bool:
    tree = ProvTrackTransformer().visit(ast.parse(src))
    return "trace_step" in ast.unparse(tree)


@pytest.mark.parametrize("name", ["copy", "head", "describe", "value_counts", "sample", "nunique"])
def test_inspection_methods_are_omitted_when_unbound_but_not_when_bound(name):
    assert _is_omitted(f"df.{name}") is True
    assert _is_omitted(f"df.{name}", bound=True) is False


@pytest.mark.parametrize("src", ["x = df.copy()", "stats = df['a'].value_counts()", "summary = df.describe()"])
def test_a_bound_result_of_an_inspection_method_is_tracked(src):
    assert _wrapped(src)


@pytest.mark.parametrize("src", ["df.head()", "df.describe()", "print(df)", "display(df)", "x = len(df)", "y = print(df)"])
def test_display_calls_and_builtins_stay_untracked(src):
    assert not _wrapped(src)


def test_set_option_is_not_an_analysis_step():
    assert not _wrapped("pd.set_option('display.max_columns', None)")
