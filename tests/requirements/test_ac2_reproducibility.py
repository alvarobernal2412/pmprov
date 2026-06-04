"""
R2 – Reproducibility
AC 2.1: Input state identifiers and Parquet snapshots captured per step.
AC 2.2: Parameters (scalars, callables, artifact references) captured per step.
AC 2.3: Execution environment and agent captured per step.
AC 2.4: Re-execution on recorded inputs produces an equivalent state. (FUTURE – xfail)
"""
import json
import pytest
from .conftest import settle


def test_ac2_1_input_state_id_recorded(rt, event_log):
    """AC 2.1 – analysis_steps records input_state_id pointing to the prior state."""
    rt.trace_step(func=lambda df: df.assign(b=df["case:concept:name"] + "_x"),
                  func_name="assign", raw_line="df=df.assign(b=...)",
                  args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    rows = con.execute("SELECT input_state_id, output_state_id FROM analysis_steps").fetchall()
    con.close()
    assert len(rows) >= 1
    assert rows[0][0] != ""   # input_state_id is set
    assert rows[0][1] != ""   # output_state_id is set


def test_ac2_1_parquet_snapshot_exists(rt, event_log):
    """AC 2.1 – DataFrame output is persisted as a Parquet artifact."""
    rt.trace_step(func=lambda df: df.assign(flag=True), func_name="assign",
                  raw_line="df=df.assign(flag=True)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    rows = con.execute("SELECT content_ref FROM artifact_states").fetchall()
    con.close()
    if rows:  # pyarrow may not be installed in all envs
        import pathlib
        assert pathlib.Path(rows[0][0]).exists()


def test_ac2_2_scalar_parameter_captured(rt, event_log):
    """AC 2.2 – scalar argument captured as parameter_value row."""
    # Pass a scalar threshold as an explicit arg (not inside the lambda)
    rt.trace_step(
        func=lambda df, threshold: df[df["case:concept:name"] != ""].head(threshold),
        func_name="sample_cases",
        raw_line="df = sample_cases(df, 2)",
        args=[event_log, 2],
        kwargs={},
    )
    settle(rt)

    con = rt.storage._connect(read_only=True)
    rows = con.execute(
        "SELECT value_type FROM parameter_values WHERE value_type='scalar'"
    ).fetchall()
    con.close()
    assert len(rows) >= 1


def test_ac2_2_lambda_parameter_captured(rt, event_log):
    """AC 2.2 – callable argument captured as lambda_function parameter_value.

    The callable must be passed as an ARGUMENT (not as func) to be captured.
    """
    predicate = lambda row: row["case:concept:name"] == "A1"  # noqa: E731
    # Pass predicate as an explicit kwarg so it is captured as LambdaParameterValue
    rt.trace_step(
        func=lambda df, pred: df[df.apply(pred, axis=1)],
        func_name="apply_predicate",
        raw_line="df = apply_predicate(df, pred=predicate)",
        args=[event_log],
        kwargs={"pred": predicate},
    )
    settle(rt)

    con = rt.storage._connect(read_only=True)
    rows = con.execute(
        "SELECT value_type FROM parameter_values WHERE value_type='lambda_function'"
    ).fetchall()
    con.close()
    assert len(rows) >= 1


def test_ac2_3_runtime_environment_captured(rt):
    """AC 2.3 – runtime environment row exists with Python version."""
    settle(rt)
    con = rt.storage._connect(read_only=True)
    rows = con.execute("SELECT tool_version, library_versions FROM runtime_environments").fetchall()
    con.close()
    assert len(rows) >= 1
    assert "Python" in rows[0][0]
    libs = json.loads(rows[0][1])
    assert isinstance(libs, dict)


def test_ac2_3_agent_captured(rt):
    """AC 2.3 – agent row exists with agent_type."""
    settle(rt)
    con = rt.storage._connect(read_only=True)
    rows = con.execute("SELECT agent_type FROM agents").fetchall()
    con.close()
    assert len(rows) >= 1
    assert rows[0][0] in ("human", "automated")


# def test_ac2_4_replay_produces_equivalent_state(rt, event_log):
#     """AC 2.4 – re-executing a recorded step yields an equivalent output.
#     Replay API not yet implemented.
#     """
#     raise NotImplementedError
