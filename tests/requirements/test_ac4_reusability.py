"""
R4 – Reusability of Analysis Steps
AC 4.1: A specific step is extracted and uniquely identified as a pipeline fragment.
AC 4.2: Parameters can be abstracted and overridden for reuse. (FUTURE – xfail)
AC 4.3: The pipeline is instantiated on distinct input data to generate new states.
"""
import json
import pytest
from .conftest import settle


def test_ac4_1_pipeline_fragment_created(rt, event_log):
    """AC 4.1 – create_pipeline records a pipeline row and a fragment row in the DB."""
    rt.trace_step(func=lambda df: df.assign(flag=True), func_name="assign",
                  raw_line="df=df.assign(flag=True)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    step_ids = [r[0] for r in con.execute("SELECT step_id FROM analysis_steps").fetchall()]
    con.close()

    pipeline_id = rt.create_pipeline("delay_feature_pipe", step_ids)
    settle(rt)

    con = rt.storage._connect(read_only=True)
    pipelines = con.execute(
        "SELECT pipeline_id, name FROM pipelines WHERE pipeline_id=?", [pipeline_id]
    ).fetchall()
    fragments = con.execute(
        "SELECT step_ids FROM pipeline_fragments WHERE pipeline_id=?", [pipeline_id]
    ).fetchall()
    con.close()

    assert len(pipelines) == 1
    assert pipelines[0][1] == "delay_feature_pipe"
    assert len(fragments) == 1
    recorded = json.loads(fragments[0][0])
    assert recorded == step_ids


# def test_ac4_2_parameter_override(rt, event_log):
#     """AC 4.2 – a pipeline fragment can be replayed with different parameter values.
#     Parameter override / replay API not yet implemented.
#     """
#     raise NotImplementedError


def test_ac4_3_pipeline_covers_distinct_steps(rt, event_log):
    """AC 4.3 – pipeline records the exact step_ids from a completed analysis."""
    rt.trace_step(func=lambda df: df[df["case:concept:name"] == "A1"],
                  func_name="filter", raw_line="df=filter(df)",
                  args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(rel=1), func_name="enrich",
                  raw_line="df=enrich(df)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    step_ids = [r[0] for r in con.execute("SELECT step_id FROM analysis_steps").fetchall()]
    con.close()

    pipeline_id = rt.create_pipeline("full_pipe", step_ids)
    settle(rt)

    con = rt.storage._connect(read_only=True)
    fragments = con.execute(
        "SELECT step_ids FROM pipeline_fragments WHERE pipeline_id=?", [pipeline_id]
    ).fetchall()
    con.close()

    recorded = json.loads(fragments[0][0])
    assert set(recorded) == set(step_ids)
