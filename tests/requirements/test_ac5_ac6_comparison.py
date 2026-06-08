"""
R5 – Comparison of Histories (AC 5.1, 5.2)
R6 – Comparison of States (AC 6.1, 6.2, 6.3)
"""
import pytest
import pandas as pd
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker
import tracker.comparison  # noqa: F401


@pytest.fixture
def rt_alt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "alt.db", artifact_dir=tmp_path / "art_alt")
    return RuntimeTracker(storage=s, session_id="alt", history_name="alt")


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_ac5_1_history_divergences_exposed(rt, rt_alt, event_log):
    """AC 5.1 – divergences across steps between two histories are surfaced."""
    rt.trace_step(func=lambda df: df.assign(shared=1), func_name="shared_step",
                  raw_line="df=shared_step(df)", args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(only_a=1), func_name="only_in_a",
                  raw_line="df=only_in_a(df)", args=[event_log], kwargs={})
    rt_alt.trace_step(func=lambda df: df.assign(shared=1), func_name="shared_step",
                      raw_line="df=shared_step(df)", args=[event_log], kwargs={})
    rt_alt.trace_step(func=lambda df: df.assign(only_b=2), func_name="only_in_b",
                      raw_line="df=only_in_b(df)", args=[event_log], kwargs={})
    settle(rt)
    settle(rt_alt)

    result = rt.compare_histories(rt_alt)

    assert "shared_step" in result["shared_operations"]
    assert "only_in_a" in result["unique_to_a"]
    assert "only_in_b" in result["unique_to_b"]


def test_ac5_2_steps_grouped_by_category(rt, rt_alt, event_log):
    """AC 5.2 – steps can be grouped into higher-level phases via StepCategory."""
    from tracker.operation_registry import step_category, _registry

    step_category("data_loading_phase", "import_op_ac52")
    step_category("feature_engineering_phase", "enrich_op_ac52")
    _registry["load_data_ac52"] = "import_op_ac52"
    _registry["add_feature_ac52"] = "enrich_op_ac52"

    rt.trace_step(func=lambda df: df, func_name="load_data_ac52",
                  raw_line="df=load_data_ac52(df)", args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="add_feature_ac52",
                  raw_line="df=add_feature_ac52(df)", args=[event_log], kwargs={})
    settle(rt)

    rt_alt.trace_step(func=lambda df: df.assign(x=1), func_name="add_feature_ac52",
                      raw_line="df=add_feature_ac52(df)", args=[event_log], kwargs={})
    settle(rt_alt)

    result = rt.compare_histories(rt_alt, group_by="category")

    assert "feature_engineering_phase" in result["shared_operations"]
    assert "data_loading_phase" in result["unique_to_a"]
    assert result["category_summary_a"]["data_loading_phase"] == 1
    assert result["category_summary_a"]["feature_engineering_phase"] == 1


def test_ac6_1_granular_state_diff(rt, event_log):
    """AC 6.1 – granular column/dtype/row differences between states are exposed."""
    rt.trace_step(func=lambda df: df.assign(col_a=1), func_name="add_a",
                  raw_line="df=add_a(df)", args=[event_log], kwargs={})
    sid_a = rt._current_state_id

    rt.trace_step(func=lambda df: df.assign(col_b=2.0), func_name="add_b",
                  raw_line="df=add_b(df)", args=[event_log], kwargs={})
    sid_b = rt._current_state_id
    settle(rt)

    cmp = rt.compare_states(sid_a, sid_b)

    assert "col_a" in cmp["unique_to_a"]
    assert "col_b" in cmp["unique_to_b"]
    assert cmp["shape_a"][0] == cmp["shape_b"][0]


def test_ac6_2_abstractions_registered_and_stored(rt, event_log):
    """AC 6.2 – analyst-defined abstraction functions are computed and stored per state."""
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="enrich",
                  raw_line="df=enrich(df)", args=[event_log], kwargs={})
    sid = rt._current_state_id
    settle(rt)

    rt.register_abstraction("row_count", lambda df, s: len(df))
    rt.register_abstraction("col_names", lambda df, s: sorted(df.columns.tolist()))
    rt.apply_abstractions(sid)

    assert rt._abstraction_cache[sid]["row_count"] == len(event_log)
    assert "x" in rt._abstraction_cache[sid]["col_names"]


def test_ac6_3_abstracted_comparison(rt, event_log):
    """AC 6.3 – abstracted metrics are structurally aligned for cross-state comparison."""
    rt.trace_step(func=lambda df: df[df["case:concept:name"] == "A1"], func_name="filter_a1",
                  raw_line="df=filter_a1(df)", args=[event_log], kwargs={})
    sid_a = rt._current_state_id

    rt.trace_step(func=lambda df: df[df["case:concept:name"] == "A2"], func_name="filter_a2",
                  raw_line="df=filter_a2(df)", args=[event_log], kwargs={})
    sid_b = rt._current_state_id
    settle(rt)

    rt.register_abstraction("row_count", lambda df, s: len(df))
    rt.apply_abstractions(sid_a)
    rt.apply_abstractions(sid_b)

    result = rt.compare_states_abstracted(sid_a, sid_b)

    assert "row_count" in result
    assert result["row_count"]["a"] == 2
    assert result["row_count"]["b"] == 2
    assert "summary" in result

