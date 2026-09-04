# tests/tracker/test_session_resume_integration.py
"""
End-to-end proof of the full session-resume story, independent of any
notebook: two "sessions" (separate RuntimeTracker instances, as a kernel
restart would produce) sharing one on-disk db, producing three branches --
matching the acceptance scenario from
docs/superpowers/specs/2026-09-03-session-resume-design.md.
"""
import pandas as pd
import pytest
from tracker.kernel_hooks import init_marimo
from tracker.storage import DuckDBSQLiteBackend as StorageBackend


@pytest.fixture
def db_paths(tmp_path):
    return tmp_path / "prov.db", tmp_path / "art"


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


def _sankey(df, activities):
    return {"activities": activities, "rows": len(df)}


def test_three_sessions_same_name_produce_one_history_three_branches(db_paths):
    db_path, artifact_dir = db_paths
    event_log = pd.DataFrame({"case": ["A", "A", "B"], "activity": ["x", "y", "x"]})

    # Session 1: load data, build first Sankey.
    rt1 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="proc")
    rt1.trace_step(func=lambda: event_log, func_name="load",
                    raw_line="load()", args=[], kwargs={})
    rt1.trace_step(func=_sankey, func_name="sankey", raw_line="sankey(df, [x])",
                    args=[event_log, ["x"]], kwargs={})
    settle(rt1)
    history_id = rt1._history.history_id

    # Session 2 ("kernel restart"): resumes by default, different Sankey config
    # diverges into a new branch.
    rt2 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="proc")
    assert rt2._history.history_id == history_id
    rt2.trace_step(func=_sankey, func_name="sankey", raw_line="sankey(df, [y])",
                    args=[event_log, ["y"]], kwargs={})
    settle(rt2)
    branch_after_2 = rt2._branch.branch_id
    assert branch_after_2 != rt1._branch.branch_id

    # Session 3: resumes session 2's branch, another distinct config diverges again.
    rt3 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="proc")
    assert rt3._history.history_id == history_id
    assert rt3._branch.branch_id == branch_after_2
    rt3.trace_step(func=_sankey, func_name="sankey", raw_line="sankey(df, [x, y])",
                    args=[event_log, ["x", "y"]], kwargs={})
    settle(rt3)

    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    branches = storage.load_branches(history_id)
    assert len(branches) == 3

    # One Operation row per function name, not one per session.
    con = storage._connect(read_only=True)
    try:
        sankey_ops = con.execute(
            "SELECT COUNT(*) FROM operations WHERE name = 'sankey'"
        ).fetchone()[0]
    finally:
        con.close()
    assert sankey_ops == 1


def test_fresh_true_starts_an_independent_history_under_the_same_name(db_paths):
    db_path, artifact_dir = db_paths
    rt1 = init_marimo(db_path=db_path, artifact_dir=artifact_dir, history_name="proc")
    rt1.trace_step(func=lambda: pd.DataFrame({"a": [1]}), func_name="load",
                    raw_line="load()", args=[], kwargs={})
    settle(rt1)

    rt2 = init_marimo(
        db_path=db_path, artifact_dir=artifact_dir, history_name="proc", fresh=True,
    )

    assert rt2._history.history_id != rt1._history.history_id
    assert rt2._history.name == "proc"

    storage = StorageBackend(db_path=db_path, artifact_dir=artifact_dir)
    con = storage._connect(read_only=True)
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM analysis_histories WHERE name = 'proc'"
        ).fetchone()[0]
    finally:
        con.close()
    assert count == 2
