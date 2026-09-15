"""
R3 – Branching (manual-only model)
AC 3.1: Divergence point and shared ancestry explicit in branch records.
AC 3.2: Divergent branches represented independently with branch identifiers and step counts.

Branching in this build is manual-only: calling the same func_name twice with
different arguments during ordinary execution never forks anything on its
own — it just appends. A branch is only ever created as a side effect of the
one step run immediately after checkout(), and only if that step doesn't
exactly repeat something that already ran from the checked-out state
(func_name + param_fingerprint match => rejoin instead of forking again).
"""
import pytest
from .conftest import settle


def test_ac3_0_no_auto_branch_without_checkout(rt, event_log):
    """Baseline for the new model: repeating the same func_name with
    different arguments, with no checkout in between, must NOT branch."""
    def apply_threshold(df, threshold):
        return df[df.index < threshold]

    rt.trace_step(func=apply_threshold, func_name="apply_threshold",
                  raw_line="df = apply_threshold(df, 2)",
                  args=[event_log, 2], kwargs={})
    rt.trace_step(func=apply_threshold, func_name="apply_threshold",
                  raw_line="df = apply_threshold(df, 1)",
                  args=[event_log, 1], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    branches = con.execute("SELECT branch_id FROM analysis_branches").fetchall()
    con.close()
    assert len(branches) == 1, "no checkout happened — everything must append to 'main'"


def test_ac3_1_checkout_alone_creates_no_branch(rt, event_log):
    """checkout() by itself is not a fork — only a subsequent, diverging step is."""
    rt.trace_step(func=lambda df: df.assign(b=1), func_name="assign",
                  raw_line="df=df.assign(b=1)", args=[event_log], kwargs={})
    settle(rt)
    target_state = rt._current_state_id

    rt.checkout(target_state, branch_name="experiment")
    settle(rt)

    con = rt.storage._connect(read_only=True)
    branches = con.execute("SELECT branch_id FROM analysis_branches").fetchall()
    con.close()
    assert len(branches) == 1, "checkout() alone must not create a branch yet"


def test_ac3_1_diverging_step_after_checkout_creates_branch_at_checkout_point(rt, event_log):
    """AC 3.1 — once a diverging step follows checkout(), the resulting branch's
    starts_at_state_id is exactly the checked-out state."""
    rt.trace_step(func=lambda df: df.assign(b=1), func_name="assign",
                  raw_line="df=df.assign(b=1)", args=[event_log], kwargs={})
    settle(rt)
    target_state = rt._current_state_id

    rt.checkout(target_state, branch_name="experiment")
    rt.trace_step(func=lambda df: df.assign(c=2), func_name="different_step",
                  raw_line="df=df.assign(c=2)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    row = con.execute(
        "SELECT starts_at_state_id FROM analysis_branches WHERE name = ?", ["experiment"]
    ).fetchone()
    con.close()
    assert row is not None, "diverging step after checkout must create the named branch"
    assert row[0] == target_state


def test_ac3_2_branches_independent(rt, event_log):
    """AC 3.2 — two separate checkouts from the same trunk, each followed by a
    distinct step, produce two independent, distinctly-identified branches."""
    def filter_by_idx(df, n):
        return df.head(n)

    rt.trace_step(func=filter_by_idx, func_name="filter_by_idx",
                  raw_line="df=filter_by_idx(df, 3)",
                  args=[event_log, 3], kwargs={})
    settle(rt)
    trunk = rt._current_state_id

    rt.checkout(trunk, branch_name="branch_a")
    rt.trace_step(func=filter_by_idx, func_name="filter_by_idx",
                  raw_line="df=filter_by_idx(df, 1)",
                  args=[event_log, 1], kwargs={})
    settle(rt)

    rt.checkout(trunk, branch_name="branch_b")
    rt.trace_step(func=filter_by_idx, func_name="filter_by_idx",
                  raw_line="df=filter_by_idx(df, 2)",
                  args=[event_log, 2], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    branch_ids = [r[0] for r in con.execute("SELECT branch_id FROM analysis_branches").fetchall()]
    con.close()

    assert len(branch_ids) == len(set(branch_ids))
    assert len(branch_ids) >= 3  # main + branch_a + branch_b


def test_ac3_2_branch_step_count(rt, event_log):
    """AC 3.2 — after a diverging checkout, the step it produces is recorded
    under the history like any other; total step count grows accordingly."""
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="assign",
                  raw_line="df=df.assign(x=1)", args=[event_log], kwargs={})
    settle(rt)
    tip = rt._current_state_id

    rt.checkout(tip, branch_name="new-branch")
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="different_assign",
                  raw_line="df=df.assign(y=2)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    steps = con.execute(
        "SELECT step_id FROM analysis_steps WHERE history_id=?",
        [rt._history.history_id]
    ).fetchall()
    con.close()
    assert len(steps) >= 2  # one on main, one on the new branch


def test_ac3_3_repeating_the_same_step_after_checkout_rejoins_without_branching(rt, event_log):
    """The 'deep comparison' rule: checking out to a state and then re-running
    the EXACT step (same func_name + arguments) that already ran from THAT
    state must rejoin the existing branch rather than forking a redundant
    duplicate."""
    def step(df):
        return df.assign(z=1)

    root = rt._current_state_id  # nothing has run yet — this is where "step" will run from
    rt.trace_step(func=step, func_name="step", raw_line="df=step(df)",
                  args=[event_log], kwargs={})
    settle(rt)

    # Rewind to the SAME starting point and re-issue the identical step.
    rt.checkout(root, branch_name="side")
    rt.trace_step(func=step, func_name="step", raw_line="df=step(df)",
                  args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    branches = con.execute(
        "SELECT branch_id FROM analysis_branches WHERE name = ?", ["side"]
    ).fetchall()
    con.close()
    assert len(branches) == 0, "an identical repeat must rejoin, not create 'side'"
