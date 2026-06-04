"""Tests for RuntimeTracker introspection methods."""
import pytest
import pandas as pd
from tracker.storage import StorageBackend
from tracker.runtime import RuntimeTracker
import tracker.introspection  # noqa: F401 — patches methods onto RuntimeTracker


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


def test_describe_state_returns_all_keys(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="assign",
                  raw_line="df=df.assign(x=1)", args=[event_log], kwargs={})
    settle(rt)
    detail = rt.describe_state(rt._current_state_id)

    for key in ("state_id", "func_name", "raw_line", "operation", "agent",
                "environment", "params", "delta", "branch_name"):
        assert key in detail, f"Missing key: {key}"
    assert detail["func_name"] == "assign"
    assert detail["agent"]["username"] != ""


def test_describe_state_returns_empty_for_root(rt):
    detail = rt.describe_state(rt._root_state_id)
    assert detail == {}


def test_list_branches_returns_dataframe(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="f",
                  raw_line="df=f(df)", args=[event_log], kwargs={})
    settle(rt)
    df = rt.list_branches()

    assert isinstance(df, pd.DataFrame)
    assert "name" in df.columns
    assert "step_count" in df.columns
    assert "divergence_point_id" in df.columns
    assert "main" in df["name"].values


def test_list_branches_shows_new_branch(rt, event_log):
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="f",
                  raw_line="df=f(df)", args=[event_log], kwargs={})
    fork = rt._current_state_id
    settle(rt)
    rt.checkout(fork, branch_name="alt")
    settle(rt)

    df = rt.list_branches()
    assert "alt" in df["name"].values
    alt_row = df[df["name"] == "alt"].iloc[0]
    assert alt_row["divergence_point_id"] == fork


def test_replay_state_produces_new_state(rt, event_log):
    filtered = rt.trace_step(
        func=lambda df: df[df["case:concept:name"] == "A1"],
        func_name="filter",
        raw_line="df=filter(df)",
        args=[event_log],
        kwargs={},
    )
    original_state = rt._current_state_id
    settle(rt)

    replayed = rt.replay_state(original_state)

    assert replayed is not None
    assert rt._current_state_id != original_state


def test_replay_state_unknown_returns_none(rt):
    result = rt.replay_state("nonexistent-state-id")
    assert result is None


def test_replay_pipeline_applies_to_new_input(rt, event_log):
    original_func = lambda df: df.assign(flag=True)
    rt.trace_step(func=original_func, func_name="assign",
                  raw_line="df=df.assign(flag=True)", args=[event_log], kwargs={})
    settle(rt)
    con = rt.storage._connect(read_only=True)
    step_ids = [r[0] for r in con.execute("SELECT step_id FROM analysis_steps").fetchall()]
    con.close()
    pipeline_id = rt.create_pipeline("test_pipe", step_ids)
    settle(rt)

    result = rt.replay_pipeline(
        pipeline_id=pipeline_id,
        func_map={"assign": original_func},
        initial_input=event_log.copy(),
    )

    assert result["errors"] == []
    assert "flag" in result["output"].columns


def test_replay_pipeline_param_override_changes_output(rt, event_log):
    original_func = lambda df, col_name="orig": df.assign(**{col_name: 1})
    rt.trace_step(func=original_func, func_name="assign",
                  raw_line="df=assign(df)", args=[event_log], kwargs={"col_name": "orig"})
    settle(rt)
    con = rt.storage._connect(read_only=True)
    step_ids = [r[0] for r in con.execute("SELECT step_id FROM analysis_steps").fetchall()]
    con.close()
    pipeline_id = rt.create_pipeline("override_pipe", step_ids)
    settle(rt)

    result = rt.replay_pipeline(
        pipeline_id=pipeline_id,
        func_map={"assign": original_func},
        initial_input=event_log.copy(),
        param_overrides={"assign": {"col_name": "overridden"}},
    )

    assert result["errors"] == []
    assert "overridden" in result["output"].columns


def test_show_artifact_lifecycle_runs_without_error(rt, event_log, monkeypatch):
    import tracker.visualizations  # noqa: F401
    import matplotlib
    matplotlib.use("Agg")

    rt.trace_step(func=lambda df: df.assign(x=1), func_name="enrich",
                  raw_line="df=enrich(df)", args=[event_log], kwargs={})
    settle(rt)
    state_id = rt._current_state_id

    try:
        rt.show_artifact_lifecycle(state_id)
    except Exception as exc:
        pytest.fail(f"show_artifact_lifecycle raised: {exc}")
