"""Tests for compare_states and compare_histories."""
import pytest
import pandas as pd
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker
import tracker.comparison  # noqa: F401


@pytest.fixture
def tmp_path_a(tmp_path):
    return tmp_path / "a"


@pytest.fixture
def tmp_path_b(tmp_path):
    return tmp_path / "b"


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="t", history_name="test")


@pytest.fixture
def rt2(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov2.db", artifact_dir=tmp_path / "art2")
    return RuntimeTracker(storage=s, session_id="t2", history_name="test2")


@pytest.fixture
def event_log():
    return pd.DataFrame({
        "case:concept:name": ["A1", "A1", "A2", "A2"],
        "concept:name": ["Create Fine", "Send Fine", "Create Fine", "Send Fine"],
        "time:timestamp": pd.to_datetime(["2020-01-01", "2020-04-15", "2020-01-10", "2020-02-01"]),
        "org:resource": ["r1", "r2", "r1", "r3"],
    })


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_compare_states_same_data(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="f",
                  raw_line="df=f(df)", args=[event_log], kwargs={})
    sid_a = rt._current_state_id
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="g",
                  raw_line="df=g(df)", args=[event_log], kwargs={})
    sid_b = rt._current_state_id
    settle(rt)

    cmp = rt.compare_states(sid_a, sid_b)
    assert sorted(cmp["common_columns"]) == sorted(list(event_log.columns) + ["x"])
    assert cmp["unique_to_a"] == []
    assert cmp["unique_to_b"] == []
    assert cmp["shape_a"] == cmp["shape_b"]


def test_compare_states_different_columns(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(col_a=1), func_name="add_a",
                  raw_line="df=add_a(df)", args=[event_log], kwargs={})
    sid_a = rt._current_state_id
    rt.trace_step(func=lambda df: df.assign(col_b=2), func_name="add_b",
                  raw_line="df=add_b(df)", args=[event_log], kwargs={})
    sid_b = rt._current_state_id
    settle(rt)

    cmp = rt.compare_states(sid_a, sid_b)
    assert "col_a" in cmp["unique_to_a"]
    assert "col_b" in cmp["unique_to_b"]


def test_compare_states_no_artifact_returns_partial(rt, event_log):
    cmp = rt.compare_states(rt._root_state_id, rt._root_state_id)
    assert "error" in cmp or cmp.get("common_columns") == []


def test_compare_histories_finds_shared_operations(rt, rt2, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="shared_op",
                  raw_line="df=shared_op(df)", args=[event_log], kwargs={})
    rt2.trace_step(func=lambda df: df.assign(x=1), func_name="shared_op",
                   raw_line="df=shared_op(df)", args=[event_log], kwargs={})
    settle(rt)
    settle(rt2)

    result = rt.compare_histories(rt2)
    assert "shared_op" in result["shared_operations"]
    assert isinstance(result["unique_to_a"], list)
    assert isinstance(result["unique_to_b"], list)
    assert "summary" in result


def test_compare_histories_unique_operations(rt, rt2, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="only_in_a",
                  raw_line="df=only_in_a(df)", args=[event_log], kwargs={})
    rt2.trace_step(func=lambda df: df.assign(y=2), func_name="only_in_b",
                   raw_line="df=only_in_b(df)", args=[event_log], kwargs={})
    settle(rt)
    settle(rt2)

    result = rt.compare_histories(rt2)
    assert "only_in_a" in result["unique_to_a"]
    assert "only_in_b" in result["unique_to_b"]


def test_compare_histories_by_category(rt, rt2, event_log):
    from tracker.operation_registry import step_category, _registry
    step_category("log_enrichment", "attribute_derivation_test")
    step_category("conformance_checking", "conformance_check_test")
    _registry["enrich_op_c6"] = "attribute_derivation_test"
    _registry["check_op_c6"] = "conformance_check_test"

    rt.trace_step(func=lambda df: df.assign(x=1), func_name="enrich_op_c6",
                  raw_line="df=enrich_op_c6(df)", args=[event_log], kwargs={})
    rt2.trace_step(func=lambda df: df.assign(ok=True), func_name="check_op_c6",
                   raw_line="df=check_op_c6(df)", args=[event_log], kwargs={})
    settle(rt)
    settle(rt2)

    result = rt.compare_histories(rt2, group_by="category")
    assert "log_enrichment" in result["unique_to_a"]
    assert "conformance_checking" in result["unique_to_b"]
    assert "category_summary_a" in result
    assert result["category_summary_a"].get("log_enrichment", 0) >= 1


def test_register_and_apply_abstraction(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="f",
                  raw_line="df=f(df)", args=[event_log], kwargs={})
    state_id = rt._current_state_id
    settle(rt)

    rt.register_abstraction("col_count", lambda df, sid: len(df.columns))
    rt.apply_abstractions(state_id)

    cached = rt._abstraction_cache.get(state_id, {})
    assert "col_count" in cached
    assert cached["col_count"] == len(event_log.columns) + 1


def test_apply_abstractions_silently_skips_missing_artifact(rt):
    rt.register_abstraction("cols", lambda df, sid: list(df.columns))
    rt.apply_abstractions(rt._root_state_id)
    assert rt._abstraction_cache.get(rt._root_state_id) == {}


def test_compare_states_abstracted(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(flag=True), func_name="branch_a",
                  raw_line="df=branch_a(df)", args=[event_log], kwargs={})
    sid_a = rt._current_state_id
    settle(rt)
    rt.trace_step(func=lambda df: df.assign(flag=False), func_name="branch_b",
                  raw_line="df=branch_b(df)", args=[event_log], kwargs={})
    sid_b = rt._current_state_id
    settle(rt)

    rt.register_abstraction("row_count", lambda df, sid: len(df))
    rt.apply_abstractions(sid_a)
    rt.apply_abstractions(sid_b)

    result = rt.compare_states_abstracted(sid_a, sid_b)
    assert "row_count" in result
    assert result["row_count"]["a"] == len(event_log)
    assert result["row_count"]["b"] == len(event_log)
    assert "summary" in result


def test_register_abstraction_overwrite(rt, event_log):
    rt.register_abstraction("x", lambda df, sid: 1)
    rt.register_abstraction("x", lambda df, sid: 2, overwrite=True)
    rt.trace_step(func=lambda df: df.assign(y=1), func_name="f",
                  raw_line="df=f(df)", args=[event_log], kwargs={})
    sid = rt._current_state_id
    settle(rt)
    rt.apply_abstractions(sid)
    assert rt._abstraction_cache[sid]["x"] == 2

