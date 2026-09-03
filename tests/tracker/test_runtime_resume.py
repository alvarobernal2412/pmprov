# tests/tracker/test_runtime_resume.py
import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def storage(tmp_path):
    return StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_resume_restores_current_state_and_history_name(storage):
    original = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    original.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                         raw_line="load()", args=[], kwargs={})
    settle(original)

    resumed = RuntimeTracker.resume(
        storage=storage, history_id=original._history.history_id, session_id="s2",
    )

    assert resumed._history.history_id == original._history.history_id
    assert resumed._history.name == "h"
    assert resumed._current_state_id == original._current_state_id
    assert resumed._root_state_id == original._root_state_id


def test_resume_falls_back_to_root_when_nothing_traced_yet(storage):
    original = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    # No trace_step call: active_state_id is still empty in storage.

    resumed = RuntimeTracker.resume(
        storage=storage, history_id=original._history.history_id, session_id="s2",
    )

    assert resumed._current_state_id == original._root_state_id


def test_resume_restores_the_branch_the_current_state_belongs_to(storage):
    original = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    original.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                         raw_line="load()", args=[], kwargs={})
    # Force a second branch by re-running "load" with different args on purpose.
    original.trace_step(func=lambda x=9: pd.DataFrame({"a": [x]}), func_name="load",
                         raw_line="load(9)", args=[9], kwargs={})
    settle(original)
    assert original._branch.name != "main"  # sanity: auto-branch actually fired

    resumed = RuntimeTracker.resume(
        storage=storage, history_id=original._history.history_id, session_id="s2",
    )

    assert resumed._branch.branch_id == original._branch.branch_id
    assert resumed._branch.name == original._branch.name


def test_resume_rebuilds_cell_executions_for_divergence_detection(storage):
    original = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    original.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                         raw_line="load()", args=[], kwargs={})
    settle(original)

    resumed = RuntimeTracker.resume(
        storage=storage, history_id=original._history.history_id, session_id="s2",
    )

    assert "load" in resumed._cell_executions
    assert len(resumed._cell_executions["load"]) == 1
    assert resumed._cell_executions["load"][0]["output_state_id"] == original._current_state_id

    # Prove it actually WORKS: calling "load" again with different args on the
    # resumed tracker must auto-branch, exactly as it would have if the
    # process had never restarted.
    resumed.trace_step(func=lambda x=9: pd.DataFrame({"a": [x]}), func_name="load",
                        raw_line="load(9)", args=[9], kwargs={})
    assert resumed._branch.branch_id != original._branch.branch_id


def test_resume_raises_for_unknown_history_id(storage):
    with pytest.raises(ValueError, match="no such history"):
        RuntimeTracker.resume(storage=storage, history_id="nope", session_id="s2")
