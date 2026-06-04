"""Tests for StorageBackend read helpers added in Task 1."""
import json
import pytest
import pandas as pd
from tracker.storage import StorageBackend
from tracker.runtime import RuntimeTracker


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="t", history_name="test")


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


def test_load_state_detail_returns_step_metadata(rt, event_log):
    rt.trace_step(
        func=lambda df: df[df["case:concept:name"] == "A1"],
        func_name="filter",
        raw_line="df = filter(df)",
        args=[event_log],
        kwargs={},
    )
    settle(rt)
    state_id = rt._current_state_id
    detail = rt.storage.load_state_detail(state_id)

    assert detail["state_id"] == state_id
    assert detail["func_name"] == "filter"
    assert detail["raw_line"] == "df = filter(df)"
    assert "operation" in detail
    assert "agent" in detail
    assert "environment" in detail
    assert "params" in detail
    assert isinstance(detail["params"], list)


def test_load_state_detail_unknown_state_returns_empty(rt):
    detail = rt.storage.load_state_detail("nonexistent-id")
    assert detail == {}


def test_load_branches_returns_main_branch(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="assign",
                  raw_line="df=df.assign(x=1)", args=[event_log], kwargs={})
    settle(rt)
    branches = rt.storage.load_branches(rt._history.history_id)

    assert len(branches) >= 1
    names = [b["name"] for b in branches]
    assert "main" in names
    main = next(b for b in branches if b["name"] == "main")
    assert main["step_count"] >= 1
    assert main["divergence_point_id"] is None


def test_load_branches_shows_divergence_point_on_new_branch(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="f1",
                  raw_line="df=f1(df)", args=[event_log], kwargs={})
    fork_state = rt._current_state_id
    settle(rt)
    rt.checkout(fork_state, branch_name="experiment")
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="f2",
                  raw_line="df=f2(df)", args=[event_log], kwargs={})
    settle(rt)

    branches = rt.storage.load_branches(rt._history.history_id)
    exp = next((b for b in branches if b["name"] == "experiment"), None)
    assert exp is not None
    assert exp["divergence_point_id"] == fork_state
