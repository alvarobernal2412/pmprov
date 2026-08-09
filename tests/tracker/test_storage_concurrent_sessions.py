"""
Concurrency coverage beyond test_storage_concurrency.py's single-tracker
read/write race: two independent RuntimeTracker *sessions* (the realistic
shape of contention — e.g. two analysts, or two marimo kernels, both writing
to the same shared provenance DB file) branching off the same parent state at
the same time. A single RuntimeTracker instance mutates its own
_current_state_id in-process and was never designed for concurrent trace_step
calls on itself (that's not how notebooks execute), so that scenario isn't
tested here — this is the composition that's actually plausible.
"""
from __future__ import annotations

import threading

import pandas as pd
import pytest

from tracker.runtime import RuntimeTracker
from tracker.storage import DuckDBSQLiteBackend as StorageBackend


@pytest.fixture
def event_log():
    return pd.DataFrame({"case:concept:name": ["A1", "A1"], "concept:name": ["a", "b"]})


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def test_two_sessions_branching_from_same_parent_land_as_siblings(tmp_path, event_log):
    db_path = tmp_path / "shared.db"
    artifact_dir = tmp_path / "art"

    seed_storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    seed_rt = RuntimeTracker(storage=seed_storage, session_id="seed", history_name="shared")
    seed_rt.trace_step(func=lambda df: df.assign(x=1), func_name="shared_parent",
                        raw_line="df=shared_parent(df)", args=[event_log], kwargs={})
    parent_state = seed_rt._current_state_id
    history_id = seed_rt._history.history_id
    settle(seed_rt)

    # Two independent tracker instances — separate Python objects, separate
    # session_ids — both pointed at the SAME db file and SAME history/parent
    # state, as if two analysts opened the same shared history concurrently.
    storage_a = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    storage_b = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)

    barrier = threading.Barrier(2)
    errors: list[Exception] = []
    results: dict[str, str] = {}

    def _run(storage, session_id, func_name):
        try:
            rt = RuntimeTracker(
                storage=storage, session_id=session_id,
                history_name="shared", branch_name="main",
            )
            rt._history.history_id = history_id  # attach to the seeded shared history
            rt._current_state_id = parent_state
            barrier.wait(timeout=5)  # maximize actual overlap
            rt.trace_step(
                func=lambda df: df.assign(**{func_name: 1}), func_name=func_name,
                raw_line=f"df={func_name}(df)", args=[event_log.assign(x=1)], kwargs={},
            )
            rt.storage._executor.submit(lambda: None).result()
            results[session_id] = rt._current_state_id
        except Exception as e:  # pragma: no cover - failure path under test
            errors.append(e)

    t_a = threading.Thread(target=_run, args=(storage_a, "session-a", "step_from_a"))
    t_b = threading.Thread(target=_run, args=(storage_b, "session-b", "step_from_b"))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    assert errors == [], f"Concurrent sibling-branch writes raised: {errors}"
    assert "session-a" in results and "session-b" in results
    assert results["session-a"] != results["session-b"]  # two distinct output states

    # Both new states must independently trace back to the same shared parent
    # — neither session's write should have corrupted or overwritten the
    # other's parent pointer.
    con = storage_a._connect(read_only=True)
    rows = con.execute(
        "SELECT output_state_id, input_state_id, func_name FROM analysis_steps "
        "WHERE history_id = ? AND func_name IN ('step_from_a', 'step_from_b')",
        [history_id],
    ).fetchall()
    con.close()

    assert len(rows) == 2
    for output_state_id, input_state_id, func_name in rows:
        assert input_state_id == parent_state
        assert output_state_id == results["session-a" if func_name == "step_from_a" else "session-b"]
