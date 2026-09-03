import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def storage(tmp_path):
    return StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_find_latest_history_by_name_returns_none_when_no_match(storage):
    assert storage.find_latest_history_by_name("nope") is None


def test_find_latest_history_by_name_returns_the_only_match(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="my analysis")
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                  raw_line="load()", args=[], kwargs={})
    settle(rt)

    found = storage.find_latest_history_by_name("my analysis")
    assert found == rt._history.history_id


def test_find_latest_history_by_name_picks_most_recently_touched(storage):
    rt_old = RuntimeTracker(storage=storage, session_id="s1", history_name="dup")
    rt_old.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                       raw_line="load()", args=[], kwargs={})
    settle(rt_old)

    rt_new = RuntimeTracker(storage=storage, session_id="s2", history_name="dup")
    rt_new.trace_step(func=lambda: pd.DataFrame({"a": [2]}), func_name="load",
                       raw_line="load()", args=[], kwargs={})
    settle(rt_new)

    # Touch the OLD history again, after the new one -- it should now win.
    rt_old.trace_step(func=lambda: pd.DataFrame({"a": [3]}), func_name="load2",
                       raw_line="load2()", args=[], kwargs={})
    settle(rt_old)

    found = storage.find_latest_history_by_name("dup")
    assert found == rt_old._history.history_id


def test_load_history_returns_none_for_unknown_id(storage):
    assert storage.load_history("does-not-exist") is None


def test_load_history_returns_name_and_active_state(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="my analysis")
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                   raw_line="load()", args=[], kwargs={})
    settle(rt)

    row = storage.load_history(rt._history.history_id)
    assert row == {
        "history_id": rt._history.history_id,
        "name": "my analysis",
        "active_state_id": rt._current_state_id,
    }


def test_load_history_active_state_id_is_none_before_any_step(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="empty")
    row = storage.load_history(rt._history.history_id)
    assert row["active_state_id"] is None


def test_find_root_state_id_returns_none_when_history_unknown(storage):
    assert storage.find_root_state_id("does-not-exist") is None


def test_find_root_state_id_returns_the_root(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    root_id = rt._current_state_id  # nothing traced yet: pointer is still the root
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                  raw_line="load()", args=[], kwargs={})
    settle(rt)

    assert storage.find_root_state_id(rt._history.history_id) == root_id


def test_load_state_branch_id_returns_none_when_state_unknown(storage):
    assert storage.load_state_branch_id("does-not-exist") is None


def test_load_state_branch_id_returns_branch(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                  raw_line="load()", args=[], kwargs={})
    settle(rt)

    assert storage.load_state_branch_id(rt._current_state_id) == rt._branch.branch_id


def test_load_cell_executions_groups_by_func_name_in_order(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    root = storage.find_root_state_id(rt._history.history_id)

    # Two calls to the SAME func_name with IDENTICAL args: same param
    # fingerprint, so no auto-branch fires (divergence detection only
    # forks when the fingerprint differs) -- this keeps both calls on one
    # branch, chained input->output, which is what "in order" is testing.
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                  raw_line="load()", args=[], kwargs={})
    load1_output = rt._current_state_id
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                  raw_line="load()", args=[], kwargs={})
    load2_output = rt._current_state_id

    # A different func_name, to prove grouping is keyed by func_name.
    rt.trace_step(func=lambda: pd.DataFrame({"b": [1]}), func_name="save",
                  raw_line="save()", args=[], kwargs={})
    save_output = rt._current_state_id
    settle(rt)

    executions = storage.load_cell_executions(rt._history.history_id)

    assert list(executions.keys()) == ["load", "save"]
    assert executions["load"] == [
        {
            "input_state_id": root,
            "output_state_id": load1_output,
            "param_fingerprint": executions["load"][0]["param_fingerprint"],
            "branch_id": rt._branch.branch_id,
        },
        {
            "input_state_id": load1_output,
            "output_state_id": load2_output,
            "param_fingerprint": executions["load"][1]["param_fingerprint"],
            "branch_id": rt._branch.branch_id,
        },
    ]
    assert executions["save"] == [
        {
            "input_state_id": load2_output,
            "output_state_id": save_output,
            "param_fingerprint": executions["save"][0]["param_fingerprint"],
            "branch_id": rt._branch.branch_id,
        },
    ]


def test_load_cell_executions_returns_empty_dict_for_untouched_history(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    assert storage.load_cell_executions(rt._history.history_id) == {}


def test_find_operation_by_name_returns_none_when_unrecorded(storage):
    assert storage.find_operation_by_name("never_called") is None


def test_find_operation_by_name_returns_recorded_operation(storage):
    rt = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    rt.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                  raw_line="load()", args=[], kwargs={})
    settle(rt)

    found = storage.find_operation_by_name("load")
    op, op_type, _ = rt._operation_cache["load"]
    assert found == {
        "operation_id": op.operation_id,
        "operation_type_id": op_type.type_id,
        "operation_type_name": op_type.name,
        "step_category_id": op.step_category_id,
    }
