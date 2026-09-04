# tests/tracker/test_runtime_resume.py
import pandas as pd
import pytest
from tracker.kernel_hooks import init_marimo
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def db_paths(tmp_path):
    return tmp_path / "prov.db", tmp_path / "art"


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


def test_resume_raises_when_history_has_no_states(storage):
    # A history row exists but its root state was never persisted (the spec's
    # "sub-second crash window" edge case). __init__ always writes a root
    # state synchronously, so the only honest way to reach this state -- an
    # existing history with zero rows in analysis_states -- is to delete it
    # out from under the tracker, which is exactly what find_root_state_id
    # sees in the real crash-window scenario (no row with a NULL
    # produced_by_step_id).
    original = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    history_id = original._history.history_id

    con = storage._connect()
    try:
        con.execute("DELETE FROM analysis_states WHERE history_id = ?", (history_id,))
        con.commit()
    finally:
        con.close()

    with pytest.raises(ValueError, match="history has no states, cannot resume"):
        RuntimeTracker.resume(storage=storage, history_id=history_id, session_id="s2")


def test_resume_raises_when_branch_for_current_state_is_missing(storage):
    # Construct a state that points at a branch_id no longer present in
    # analysis_branches -- genuinely reaches the load_branches()-lookup-fails
    # path in resume(), without violating any other invariant the rest of the
    # method relies on (the state row itself is still valid and has a
    # branch_id, it's just one load_branches() can no longer resolve).
    original = RuntimeTracker(storage=storage, session_id="s1", history_name="h")
    history_id = original._history.history_id

    con = storage._connect()
    try:
        con.execute("DELETE FROM analysis_branches WHERE history_id = ?", (history_id,))
        con.commit()
    finally:
        con.close()

    with pytest.raises(ValueError, match="no branch found for state"):
        RuntimeTracker.resume(storage=storage, history_id=history_id, session_id="s2")


def test_last_call_params_none_when_never_called(db_paths):
    db_path, artifact_dir = db_paths
    rt = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="p")
    assert rt.last_call_params("apply_folds") is None


def test_last_call_params_decodes_most_recent_call(db_paths):
    db_path, artifact_dir = db_paths
    rt = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="p")

    def apply_folds(log, fold_specs):
        return log

    rt.trace_step(func=apply_folds, func_name="apply_folds",
                   raw_line="apply_folds(log, fs)",
                   args=[pd.DataFrame({"a": [1]}), [{"name": "F", "activities": ["x"]}]],
                   kwargs={})
    rt.storage._executor.submit(lambda: None).result()

    params = rt.last_call_params("apply_folds")
    assert params == {"fold_specs": [{"name": "F", "activities": ["x"]}]}


def test_last_call_params_excludes_artifact_ref_params(db_paths):
    """Verify that params with artifact_state_ref value_type are excluded."""
    db_path, artifact_dir = db_paths
    rt = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="p")

    def apply_folds(log, fold_specs):
        return log

    # Call with a DataFrame (becomes artifact_state_ref) and a config list
    rt.trace_step(func=apply_folds, func_name="apply_folds",
                   raw_line="apply_folds(log, fs)",
                   args=[pd.DataFrame({"a": [1]}), [{"name": "F", "activities": ["x"]}]],
                   kwargs={})
    rt.storage._executor.submit(lambda: None).result()

    params = rt.last_call_params("apply_folds")
    # Only fold_specs (non-artifact) should be in the result
    assert "log" not in params, "artifact-ref param 'log' should be excluded"
    assert "fold_specs" in params, "non-artifact param 'fold_specs' should be included"
    assert params == {"fold_specs": [{"name": "F", "activities": ["x"]}]}
