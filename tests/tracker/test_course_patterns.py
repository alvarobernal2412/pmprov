"""Statement forms found in notebooks, run through the rewriter and a real tracker.

Each test executes cells the way a kernel does (rewrite, compile, exec in one
namespace) and checks what pmprov recorded: a step per analysis statement and,
through ``restore_state``, every variable exactly as the analyst had it.
"""
import ast

import pandas as pd
import pytest

import tracker.kernel_hooks as kernel_hooks
from tracker.ast_rewriter import ProvTrackTransformer
from tracker.runtime import RuntimeTracker
from tracker.storage import DuckDBSQLiteBackend


class Notebook:
    def __init__(self, tmp_path, monkeypatch):
        storage = DuckDBSQLiteBackend(db_path=tmp_path / "p.db", artifact_dir=tmp_path / "a")
        self.rt = RuntimeTracker(storage=storage, session_id="s", history_name="h")
        monkeypatch.setattr(kernel_hooks, "_runtime", self.rt)
        self.ns: dict = {"pd": pd}

    def run(self, src: str) -> list[dict]:
        """Execute a cell; return the steps it recorded."""
        before = {s["step_id"] for s in self.steps()}
        tree = ProvTrackTransformer().visit(ast.parse(src))
        ast.fix_missing_locations(tree)
        exec(compile(tree, "<cell>", "exec"), self.ns)
        return [s for s in self.steps() if s["step_id"] not in before]

    def steps(self) -> list[dict]:
        self.rt.storage._executor.submit(lambda: None).result()
        return self.rt.storage.load_graph(self.rt._history.history_id)["steps"]

    def state(self) -> str:
        self.rt.storage._executor.submit(lambda: None).result()
        return self.rt._current_state_id

    def restored(self, state_id=None, names=None):
        return self.rt.restore_state(state_id or self.state(), names=names).variables

    def delta(self, step):
        return self.rt.storage.load_state_detail(step["output_state_id"]).get("delta") or {}


@pytest.fixture
def nb(tmp_path, monkeypatch):
    notebook = Notebook(tmp_path, monkeypatch)
    notebook.run("df = pd.DataFrame({'a': [1, 2, 3, 4], 'g': ['x', 'y', 'x', 'y']})\nload = 0")
    return notebook


def assert_same(got, expected):
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(got, expected)
    elif isinstance(expected, pd.Series):
        pd.testing.assert_series_equal(got, expected)
    else:
        assert got == expected


# ---- statements that are not calls --------------------------------------------

@pytest.mark.parametrize("src, name", [
    ("sub = df[df['a'] > 2]", "sub"),
    ("cols = df[['a']]", "cols"),
    ("head = df[:2]", "head"),
    ("row = df.loc[1]", "row"),
    ("pos = df.iloc[0:2]", "pos"),
    ("col = df['a']", "col"),
    ("mask = df['a'] > 2", "mask"),
    ("both = (df['a'] > 1) & (df['a'] < 4)", "both"),
    ("flipped = df.T", "flipped"),
])
def test_a_subscript_or_expression_producing_a_frame_is_a_step_with_an_exact_output(nb, src, name):
    steps = nb.run(src)

    assert len(steps) == 1
    assert_same(nb.restored(names=[name])[name], nb.ns[name])


@pytest.mark.parametrize("src", ["d = {'a': 1}", "x = d['a']", "y = [1, 2, 3][1:]", "z = 1 + 2", "w = d['a'] > 0"])
def test_subscripts_and_expressions_on_plain_python_values_stay_untracked(nb, src):
    nb.ns["d"] = {"a": 1}
    assert nb.run(src) == []


def test_a_column_assignment_is_a_step_and_earlier_states_keep_the_old_frame(nb):
    nb.run("df['b'] = df['a'] * 2")
    first = nb.state()
    nb.run("df['c'] = df['b'] + 1")

    assert list(nb.restored(first)["df"].columns) == ["a", "g", "b"]
    assert list(nb.restored()["df"].columns) == ["a", "g", "b", "c"]
    assert_same(nb.restored()["df"], nb.ns["df"])


def test_a_column_assignment_from_a_call_is_one_step_whose_output_is_the_frame(nb):
    steps = nb.run("df['up'] = df['g'].str.upper()")

    assert [s["func_name"] for s in steps] == ["setitem"]
    assert_same(nb.restored()["df"], nb.ns["df"])


def test_loc_assignment_and_attribute_assignment_are_steps(nb):
    nb.run("df.loc[df['a'] > 2, 'a'] = 0")
    assert_same(nb.restored()["df"], nb.ns["df"])
    nb.run("df.columns = ['first', 'second']")
    assert list(nb.restored()["df"].columns) == ["first", "second"]


def test_a_loop_that_changes_a_frame_is_one_step_holding_the_result(nb):
    steps = nb.run("for k in ['p', 'q']:\n    df[k] = df['a'] > 2")

    assert [s["func_name"] for s in steps] == ["for"]
    assert_same(nb.restored()["df"], nb.ns["df"])
    assert nb.delta(steps[0])["columns_added"] == ["p", "q"]


def test_a_loop_that_only_prints_is_not_a_step(nb):
    assert nb.run("for k in [1, 2]:\n    print(k)") == []


def test_an_inplace_method_records_the_changed_frame(nb):
    steps = nb.run("df.sort_values(by='a', ascending=False, inplace=True)")

    assert len(steps) == 1
    assert_same(nb.restored()["df"], nb.ns["df"])
    assert nb.restored()["df"]["a"].tolist() == [4, 3, 2, 1]


# ---- calls the omit list used to swallow ---------------------------------------

@pytest.mark.parametrize("src, name", [
    ("snapshot = df.copy()", "snapshot"),
    ("counts = df['g'].value_counts()", "counts"),
    ("summary = df.describe()", "summary"),
    ("some = df.head(2)", "some"),
])
def test_a_bound_inspection_call_is_a_step(nb, src, name):
    assert len(nb.run(src)) == 1
    assert_same(nb.restored(names=[name])[name], nb.ns[name])


@pytest.mark.parametrize("src", ["df.head()", "df.describe()", "df['g'].value_counts()", "df.info()"])
def test_a_bare_inspection_call_is_not_a_step(nb, src):
    assert nb.run(src) == []


# ---- results that are not frames ----------------------------------------------

def test_series_scalars_lists_and_dicts_are_restored_exactly(nb):
    nb.run("first_by_group = df.groupby('g')['a'].first()")
    nb.run("total = df['a'].sum()")
    nb.run("groups = list(df['g'].unique())")
    nb.run("sizes = dict(df['g'].value_counts())")

    restored = nb.restored()
    for name in ("first_by_group", "total", "groups", "sizes"):
        assert_same(restored[name], nb.ns[name])


def test_a_tuple_result_bound_to_several_names_is_stored_per_name(nb):
    nb.ns["split"] = lambda frame: (frame[frame["a"] > 2], {"parts": 2}, 7)

    steps = nb.run("big, meta, seven = split(df)")

    assert len(steps) == 1
    restored = nb.restored()
    assert_same(restored["big"], nb.ns["big"])
    assert restored["meta"] == {"parts": 2} and restored["seven"] == 7


class Model:
    def __init__(self):
        self.rules = []

    def fit(self, frame):
        self.rules = list(frame.columns)


def test_a_method_that_mutates_a_tracked_object_updates_its_variable(nb):
    nb.ns["Model"] = Model
    nb.run("model = Model()")
    before = nb.state()
    nb.run("model.fit(df)")

    assert nb.restored(before)["model"].rules == []
    assert nb.restored()["model"].rules == ["a", "g"]


def test_a_plotly_figure_is_restored(nb):
    go = pytest.importorskip("plotly.graph_objects")
    nb.ns["go"] = go
    nb.run("fig = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))")
    nb.run("fig.update_layout(title='t')")

    assert nb.restored()["fig"] == nb.ns["fig"]


# ---- frames Parquet cannot hold exactly ----------------------------------------

def test_a_mixed_type_object_column_is_stored_and_restored(nb):
    nb.ns["make"] = lambda: pd.DataFrame({"v": [1, "x", None]})
    steps = nb.run("mixed = make()")

    assert len(steps) == 1
    assert_same(nb.restored()["mixed"], nb.ns["mixed"])


def test_an_object_column_of_bools_keeps_its_dtype(nb):
    nb.run("flags = df.assign(f=df['a'] > 2)")
    nb.run("flags['f'] = flags['f'].astype('object')")

    restored = nb.restored()["flags"]
    assert restored["f"].dtype == object
    assert_same(restored, nb.ns["flags"])


# ---- deltas, parameters ---------------------------------------------------------

@pytest.mark.parametrize("src, added", [
    ("renamed = df.rename(columns={'a': 'b'})", ["b"]),
    ("agg = df.groupby('g').agg(n=('a', 'count'))", ["n"]),
    ("more = df.assign(flag=1)", ["flag"]),
])
def test_a_method_call_reports_the_change_to_the_frame_it_was_called_on(nb, src, added):
    (step,) = nb.run(src)

    assert nb.delta(step)["columns_added"] == added


def test_a_type_valued_parameter_is_recorded_with_the_others(nb):
    nb.ns["read"] = lambda path, dtype=None: pd.DataFrame({"a": [1]})
    (step,) = nb.run("loaded = read('f.csv', dtype={'a': str})")

    params = nb.rt.storage.load_state_detail(step["output_state_id"])["params"]
    assert {p["param_id"] for p in params} == {"read:path", "read:dtype"}


def test_set_option_is_not_a_step(nb):
    assert nb.run("pd.set_option('display.max_columns', None)") == []


# ---- exactness guards ------------------------------------------------------------

def test_a_function_that_filters_and_resets_the_index_is_restored_exactly(nb):
    nb.ns["keep_big"] = lambda frame: frame[frame["a"] > 2].reset_index(drop=True)
    nb.run("big = keep_big(df)")

    assert_same(nb.restored()["big"], nb.ns["big"])


def test_a_filter_of_a_frame_with_duplicate_labels_is_restored_exactly(nb):
    nb.ns["dup"] = pd.concat([nb.ns["df"], nb.ns["df"]])
    nb.run("keep = dup[dup['a'] > 2]")

    assert_same(nb.restored()["keep"], nb.ns["keep"])


def test_a_variable_whose_value_is_not_stored_does_not_keep_its_old_value(nb):
    nb.run("handle = df.copy()")
    nb.ns["open_it"] = lambda: (_ for _ in ())  # a generator cannot be stored
    nb.run("handle = open_it()")

    assert "handle" not in nb.restored()
