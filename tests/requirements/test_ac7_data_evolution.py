"""
R7 – Data Evolution Transparency
AC 7.1: Fine-grained operation and execution context captured per state transition.
AC 7.2: Incremental artifact changes recorded as delta entities.
AC 7.3: Intermediate states preserved and linked (derived_from_state_id chain).
"""
import json
import pytest
from .conftest import settle


def test_ac7_1_operation_recorded_per_step(rt, event_log):
    """AC 7.1 – every step has an operation_id pointing to an operations row."""
    rt.trace_step(func=lambda df: df[df["case:concept:name"] == "A1"],
                  func_name="filter", raw_line="df=filter(df)",
                  args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    steps = con.execute("SELECT operation_id FROM analysis_steps").fetchall()
    op_ids = {r[0] for r in steps}
    ops = con.execute("SELECT operation_id FROM operations").fetchall()
    recorded_op_ids = {r[0] for r in ops}
    con.close()

    assert op_ids.issubset(recorded_op_ids), (
        f"Step operation_ids {op_ids} not found in operations table"
    )


def test_ac7_2_delta_recorded_with_modification_type(rt, event_log):
    """AC 7.2 – a row-filter step produces a delta with modification_type=removal."""
    rt.trace_step(func=lambda df: df[df["case:concept:name"] == "A1"],
                  func_name="filter", raw_line="df=filter(df)",
                  args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    rows = con.execute("SELECT modification_type, rows_delta FROM deltas").fetchall()
    con.close()

    assert len(rows) >= 1
    modification_type, rows_delta = rows[0]
    assert modification_type == "removal"
    assert rows_delta < 0


def test_ac7_2_delta_columns_added_on_enrich(rt, event_log):
    """AC 7.2 – an assign step produces a delta with columns_added populated."""
    rt.trace_step(func=lambda df: df.assign(new_col=1), func_name="assign",
                  raw_line="df=df.assign(new_col=1)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    rows = con.execute("SELECT columns_added, modification_type FROM deltas").fetchall()
    con.close()

    assert len(rows) >= 1
    cols_added = json.loads(rows[0][0])
    assert "new_col" in cols_added
    assert rows[0][1] == "addition"


def test_ac7_3_intermediate_states_linked(rt, event_log):
    """AC 7.3 – each output state's derived_from_state_id points to the prior state."""
    rt.trace_step(func=lambda df: df.assign(step1=1), func_name="step1",
                  raw_line="df=step1(df)", args=[event_log], kwargs={})
    after_step1 = rt._current_state_id

    rt.trace_step(func=lambda df: df.assign(step2=2), func_name="step2",
                  raw_line="df=step2(df)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    states = con.execute(
        "SELECT state_id, derived_from_state_id FROM analysis_states"
    ).fetchall()
    con.close()

    # Find the second output state and confirm it derives from the first
    linkage = {r[0]: r[1] for r in states}
    second_state = rt._current_state_id
    assert linkage.get(second_state) == after_step1, (
        f"State {second_state} should derive from {after_step1}, "
        f"got {linkage.get(second_state)}"
    )
