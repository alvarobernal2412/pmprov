import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend as StorageBackend, _p
from tracker.runtime import RuntimeTracker


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="t", history_name="test")


@pytest.fixture
def event_log():
    return pd.DataFrame({
        "case:concept:name": ["A1", "A1"],
        "concept:name": ["Create Fine", "Send Fine"],
    })


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_materialize_curated_history_creates_independent_history(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="step1",
                  raw_line="df=step1(df)", args=[event_log], kwargs={})
    step1_output = rt._current_state_id
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="step2",
                  raw_line="df=step2(df)", args=[event_log.assign(x=1)], kwargs={})
    target = rt._current_state_id
    settle(rt)

    # Every step writes an artifact synchronously today (see
    # test_storage_ancestor_chain.py), so force a 2-entry chain the same way that
    # test file does: delete the target's own artifact_states row (content_ref is
    # NOT NULL, so deleting is how "no snapshot" is simulated pre-FC-3).
    con = rt.storage._connect()
    con.execute("DELETE FROM artifact_states WHERE analysis_state_id = ?", _p(target))
    con.commit()
    con.close()

    chain = rt.storage.load_ancestor_chain(rt._current_state_id)
    step_ids = [c["step_id"] for c in chain]
    assert len(step_ids) == 2  # step1's artifact is the nearest ancestor snapshot

    new_history_id = rt.storage.materialize_curated_history(step_ids, name="curated-finding")

    assert new_history_id != rt._history.history_id

    con = rt.storage._connect(read_only=True)
    try:
        # New history has its own steps, not references to the originals.
        new_step_ids = [
            r[0] for r in con.execute(
                "SELECT step_id FROM analysis_steps WHERE history_id = ? ORDER BY timestamp",
                _p(new_history_id),
            ).fetchall()
        ]
        assert len(new_step_ids) == 2
        assert set(new_step_ids).isdisjoint(set(step_ids))  # cloned, not shared, step_ids

        func_names = [
            r[0] for r in con.execute(
                "SELECT func_name FROM analysis_steps WHERE history_id = ? ORDER BY timestamp",
                _p(new_history_id),
            ).fetchall()
        ]
        assert func_names == ["step1", "step2"]

        # Original artifact's content_ref is reused verbatim (no parquet duplication).
        orig_content_ref = con.execute(
            """SELECT ast.content_ref FROM artifact_states ast
               WHERE ast.analysis_state_id = ?""",
            _p(step1_output),
        ).fetchone()
        new_content_refs = [
            r[0] for r in con.execute(
                """SELECT ast.content_ref FROM artifact_states ast
                   JOIN artifacts a ON a.artifact_id = ast.artifact_id
                   WHERE a.history_id = ?""",
                _p(new_history_id),
            ).fetchall()
        ]
        assert orig_content_ref[0] in new_content_refs
    finally:
        con.close()


def test_materialize_curated_history_rejects_empty_step_ids(rt):
    with pytest.raises(ValueError):
        rt.storage.materialize_curated_history([], name="empty")


def test_materialize_curated_history_rejects_unknown_step_id(rt):
    with pytest.raises(ValueError):
        rt.storage.materialize_curated_history(["does-not-exist"], name="bad")
