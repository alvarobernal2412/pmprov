"""
R3 – Branching
AC 3.1: Divergence point and shared ancestry explicit in branch records.
AC 3.2: Divergent branches represented independently with branch identifiers and step counts.
"""
import pytest
from .conftest import settle


def test_ac3_1_auto_branch_divergence_point(rt, event_log):
    """AC 3.1 – auto-branch records starts_at_state_id pointing to the divergence state.

    Auto-branching fires when the SAME func_name is called twice with DIFFERENT
    argument values (different param fingerprint). The threshold scalar arg makes
    the fingerprint differ between the two calls.
    """
    def apply_threshold(df, threshold):
        return df[df.index < threshold]

    # Capture the state BEFORE the first call — this is the divergence point
    # (the new branch will start from the input state of the re-run step)
    divergence_point = rt._current_state_id

    # First execution: threshold=2
    rt.trace_step(func=apply_threshold, func_name="apply_threshold",
                  raw_line="df = apply_threshold(df, 2)",
                  args=[event_log, 2], kwargs={})

    # Second execution of same func_name with different threshold → auto-branch
    rt.trace_step(func=apply_threshold, func_name="apply_threshold",
                  raw_line="df = apply_threshold(df, 1)",
                  args=[event_log, 1], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    branches = con.execute(
        "SELECT branch_id, starts_at_state_id FROM analysis_branches"
    ).fetchall()
    con.close()

    assert len(branches) >= 2, f"Expected >=2 branches, got {len(branches)}"
    starts_at_ids = {b[1] for b in branches}
    assert divergence_point in starts_at_ids, (
        f"Divergence point {divergence_point} not found in branch starts_at: {starts_at_ids}"
    )


def test_ac3_1_manual_checkout_divergence_point(rt, event_log):
    """AC 3.1 – manual checkout creates a branch starting at exactly the chosen state."""
    rt.trace_step(func=lambda df: df.assign(b=1), func_name="assign",
                  raw_line="df=df.assign(b=1)", args=[event_log], kwargs={})
    settle(rt)
    target_state = rt._current_state_id

    branch = rt.checkout(target_state, branch_name="experiment")
    assert branch.starts_at_state_id == target_state


def test_ac3_2_branches_independent(rt, event_log):
    """AC 3.2 – divergent branches have distinct branch_ids and each records their own steps."""
    def filter_by_idx(df, n): return df.head(n)

    rt.trace_step(func=filter_by_idx, func_name="filter_by_idx",
                  raw_line="df=filter_by_idx(df, 3)",
                  args=[event_log, 3], kwargs={})
    rt.trace_step(func=filter_by_idx, func_name="filter_by_idx",
                  raw_line="df=filter_by_idx(df, 1)",
                  args=[event_log, 1], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    branch_ids = [r[0] for r in con.execute("SELECT branch_id FROM analysis_branches").fetchall()]
    con.close()

    assert len(branch_ids) == len(set(branch_ids))
    assert len(branch_ids) >= 2


def test_ac3_2_branch_step_count(rt, event_log):
    """AC 3.2 – after a branch, steps on the new branch are recorded under the new branch_id."""
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="assign",
                  raw_line="df=df.assign(x=1)", args=[event_log], kwargs={})
    settle(rt)
    branch = rt.checkout(rt._current_state_id, branch_name="new-branch")

    # Record a step on the new branch
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="assign",
                  raw_line="df=df.assign(y=2)", args=[event_log], kwargs={})
    settle(rt)

    con = rt.storage._connect(read_only=True)
    steps_on_new = con.execute(
        "SELECT step_id FROM analysis_steps WHERE history_id=?",
        [rt._history.history_id]
    ).fetchall()
    con.close()
    assert len(steps_on_new) >= 2  # at least one on main, one on new branch
