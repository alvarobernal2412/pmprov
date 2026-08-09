"""
Scale regression guard for the pure pruning-graph functions.

_cascade_hidden_state_ids does a BFS over the full step list per call, and
_compute_pruned_graph does a chain-walk per head step when
group_by_category=True — both are called on every build_pruned_view()/
load_pruned_view() render. Neither takes a storage backend, so this exercises
them directly with synthetic data (~5k steps / 50 branches) instead of
round-tripping through RuntimeTracker.trace_step 5000 times (which would
mostly measure Parquet snapshot I/O, not the algorithm under test here).

This intentionally does NOT stress-test storage.load_ancestor_chain (FC-1) or
a real DB-backed build_pruned_view() at this scale — that would need either
5000 real trace_step calls (too slow to be a routine regression test) or
direct SQL fixture-seeding of the schema (a second, separate maintenance
burden). Scoped down from docs/claude/testing-plan-fc1-3.md item 12 to the
part with a concrete, already-flagged O(n²) risk (see pruning.py's
_render_pruned_view: it reloads and rebuils categories_by_step_id on every
single call).
"""
from __future__ import annotations

import time

from tracker.pruning import _cascade_hidden_state_ids, _compute_pruned_graph

N_BRANCHES = 50
STEPS_PER_BRANCH = 100  # 50 * 100 = 5,000 steps total
WALL_CLOCK_BUDGET_SECONDS = 5.0


def _build_synthetic_forest() -> tuple[list[dict], list[dict]]:
    """A shared root, fanning out into N_BRANCHES independent linear chains
    of STEPS_PER_BRANCH steps each — same shape multi-branch pruning tests
    already use, just at scale."""
    states = [{"state_id": "root"}]
    steps: list[dict] = []
    for b in range(N_BRANCHES):
        parent = "root"
        for i in range(STEPS_PER_BRANCH):
            state_id = f"b{b}_s{i}"
            step_id = f"b{b}_step{i}"
            steps.append({
                "step_id": step_id, "input_state_id": parent, "output_state_id": state_id,
                "func_name": f"op_{i % 3}",  # 3 rotating categories per branch
            })
            states.append({"state_id": state_id})
            parent = state_id
    return states, steps


def test_cascade_hidden_state_ids_scales_linearly():
    _, steps = _build_synthetic_forest()

    # Hide one state near the head of a single branch — must cascade through
    # that whole branch's ~100 descendants without walking the other 49.
    seed = {"b0_s5"}

    start = time.perf_counter()
    hidden = _cascade_hidden_state_ids(steps, seed)
    elapsed = time.perf_counter() - start

    assert elapsed < WALL_CLOCK_BUDGET_SECONDS, (
        f"_cascade_hidden_state_ids took {elapsed:.2f}s for {len(steps)} steps — "
        f"budget is {WALL_CLOCK_BUDGET_SECONDS}s, investigate for an accidental O(n^2)"
    )
    # b0_s5..b0_s99 hidden (95 descendants) + the seed itself.
    assert len(hidden) == (STEPS_PER_BRANCH - 5)
    assert all(sid.startswith("b0_") for sid in hidden)


def test_compute_pruned_graph_group_by_category_scales_linearly():
    states, steps = _build_synthetic_forest()
    categories_by_step_id = {s["step_id"]: s["func_name"] for s in steps}  # 1:1 stand-in

    start = time.perf_counter()
    result = _compute_pruned_graph(
        states, steps, categories_by_step_id,
        group_by_category=True, hidden_state_ids=set(),
    )
    elapsed = time.perf_counter() - start

    assert elapsed < WALL_CLOCK_BUDGET_SECONDS, (
        f"_compute_pruned_graph took {elapsed:.2f}s for {len(steps)} steps — "
        f"budget is {WALL_CLOCK_BUDGET_SECONDS}s, investigate for an accidental O(n^2)"
    )
    assert len(result["nodes"]) == len(states)
    # op_0/op_1/op_2 rotate per-step, so nothing consecutive shares a category
    # -> no collapsing possible -> one edge per step, same as ungrouped.
    assert len(result["edges"]) == len(steps)
