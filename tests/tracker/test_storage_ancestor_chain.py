import pandas as pd
import pytest
from tracker.storage import DuckDBSQLiteBackend as StorageBackend
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


def test_ancestor_chain_returns_single_step_when_parent_has_artifact(rt, event_log):
    # Every step today writes an artifact synchronously (save_artifact is sync),
    # so the immediate parent always qualifies as the nearest artifact-bearing ancestor.
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="step1",
                  raw_line="df=step1(df)", args=[event_log], kwargs={})
    target = rt._current_state_id
    settle(rt)

    chain = rt.storage.load_ancestor_chain(target)

    assert len(chain) == 1
    assert chain[0]["state_id"] == target
    assert chain[0]["func_name"] == "step1"
    assert chain[0]["has_artifact"] is True


def test_ancestor_chain_returns_multi_step_sequence(rt, event_log):
    # The walk starts AT target and checks target's own artifact first (that's what
    # makes the single-step test's short-circuit correct). So to force a 2-entry
    # chain, it's the TARGET (step2's own state) that must lack an artifact — not an
    # intermediate ancestor. Every step writes an artifact synchronously today, so
    # this simulates what FC-3 (configurable snapshotting) will eventually produce
    # naturally: a step deliberately not snapshotted.
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="step1",
                  raw_line="df=step1(df)", args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="step2",
                  raw_line="df=step2(df)", args=[event_log.assign(x=1)],
                  kwargs={})
    target = rt._current_state_id
    settle(rt)

    from tracker.storage import _p
    con = rt.storage._connect()
    # content_ref is NOT NULL in the schema, so simulate "no snapshot" by deleting
    # the row entirely rather than nulling the column.
    con.execute("DELETE FROM artifact_states WHERE analysis_state_id = ?", _p(target))
    con.commit()
    con.close()

    chain = rt.storage.load_ancestor_chain(target)

    assert [c["func_name"] for c in chain] == ["step1", "step2"]
    assert chain[-1]["state_id"] == target
    assert chain[0]["has_artifact"] is True   # step1 (nearest ancestor with a snapshot)
    assert chain[1]["has_artifact"] is False  # step2/target (deliberately un-snapshotted)


def test_ancestor_chain_empty_for_unknown_state(rt):
    assert rt.storage.load_ancestor_chain("does-not-exist") == []


def test_ancestor_chain_empty_for_root_state(rt):
    assert rt.storage.load_ancestor_chain(rt._root_state_id) == []
