"""
Cross-feature composition tests for FC-1 (shortest-path replay), FC-2 (pruned
views), and FC-3 (snapshot policy).

Each feature has solid unit coverage in isolation (test_introspection.py,
test_pruning.py, test_snapshot_policy.py) but they're explicitly designed to be
used together — FC-1's own docstring for create_independent_history_from_state
calls out FC-3's snapshot_policy by name as something it must handle — and
nothing exercised that composition until now. See docs/claude/testing-plan-fc1-3.md
section 2 for the full rationale behind each test below.
"""
from __future__ import annotations

import pandas as pd
import pytest

from tracker.operation_registry import _registry, step_category
from tracker.runtime import RuntimeTracker
from tracker.snapshot_policy import snapshot_policy
from tracker.storage import DuckDBSQLiteBackend as StorageBackend


@pytest.fixture
def rt(tmp_path):
    s = StorageBackend(db_path=tmp_path / "prov.db", artifact_dir=tmp_path / "art")
    return RuntimeTracker(storage=s, session_id="t", history_name="test")


@pytest.fixture
def event_log():
    return pd.DataFrame({"case:concept:name": ["A1", "A1"], "concept:name": ["a", "b"]})


def settle(rt):
    rt.storage._executor.submit(lambda: None).result()


# ── 1. FC-3 -> FC-1: mixed snapshot policy along a replay chain ─────────────

def test_shortest_replay_path_correct_with_mixed_snapshot_policy(rt, event_log):
    """step1 (snapshotted) -> step2 (never) -> step3 (never, = target). The
    target's own state has no artifact, so the nearest artifact-bearing
    ancestor is step1 — find_shortest_replay_path must return all three
    steps (step1..step3), not just a naive "immediate parent" shortcut."""
    snapshot_policy("no_snap_step2", "never")
    snapshot_policy("no_snap_step3", "never")

    rt.trace_step(func=lambda df: df.assign(x=1), func_name="snap_step1",
                  raw_line="df=snap_step1(df)", args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="no_snap_step2",
                  raw_line="df=no_snap_step2(df)", args=[event_log.assign(x=1)], kwargs={})
    rt.trace_step(func=lambda df: df.assign(z=3), func_name="no_snap_step3",
                  raw_line="df=no_snap_step3(df)", args=[event_log.assign(x=1, y=2)], kwargs={})
    target = rt._current_state_id
    settle(rt)

    # Sanity: target itself really has no artifact (proves the policy took effect).
    assert rt.storage.load_output_artifact_state_id(target) is None

    step_ids = rt.find_shortest_replay_path(target)
    func_names = [rt.describe_step(sid)["func_name"] for sid in step_ids]

    assert func_names == ["snap_step1", "no_snap_step2", "no_snap_step3"]


def test_create_independent_history_picks_nearest_snapshotted_ancestor(rt, event_log):
    """Same shape as above: materializing a curated history from the
    never-snapshotted target must walk back to step1's artifact as the
    starting point, not fail or silently start from an empty/missing input."""
    snapshot_policy("mid_never", "never")
    snapshot_policy("tail_never", "never")

    rt.trace_step(func=lambda df: df.assign(x=1), func_name="head_snap",
                  raw_line="df=head_snap(df)", args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="mid_never",
                  raw_line="df=mid_never(df)", args=[event_log.assign(x=1)], kwargs={})
    rt.trace_step(func=lambda df: df.assign(z=3), func_name="tail_never",
                  raw_line="df=tail_never(df)", args=[event_log.assign(x=1, y=2)], kwargs={})
    target = rt._current_state_id
    settle(rt)

    history_id = rt.create_independent_history_from_state(target, name="curated-mixed-policy")
    assert history_id is not None

    graph = rt.storage.load_graph(history_id)
    assert [s["func_name"] for s in graph["steps"]] == ["head_snap", "mid_never", "tail_never"]


def test_create_independent_history_rejects_when_every_ancestor_unsnapshotted(rt, event_log):
    """FC-1's own documented invariant: if snapshot_policy disables every
    operation on the replay chain, there is no loadable starting artifact —
    materialization must raise, not silently produce a broken history."""
    snapshot_policy("root_never", "never")
    snapshot_policy("leaf_never", "never")

    rt.trace_step(func=lambda df: df.assign(x=1), func_name="root_never",
                  raw_line="df=root_never(df)", args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="leaf_never",
                  raw_line="df=leaf_never(df)", args=[event_log.assign(x=1)], kwargs={})
    target = rt._current_state_id
    settle(rt)

    with pytest.raises(ValueError):
        rt.create_independent_history_from_state(target, name="should-fail")


# ── 2. FC-3 -> FC-2: never-snapshotted steps inside a collapsed chain ───────

def test_pruned_view_collapses_chain_with_never_snapshotted_steps(rt, event_log):
    """Pruning is presentation-only and reads states/steps, not artifacts — a
    same-category run must still collapse into one edge even when some (or
    all) of its steps never got a snapshot."""
    # _registry maps func_name -> operation_type; step_category maps
    # operation_type -> category. All three funcs share one operation_type
    # ("enrich_optype") so they land in the same category.
    step_category("enrichment", "enrich_optype")
    _registry["enrich_a"] = "enrich_optype"
    _registry["enrich_b"] = "enrich_optype"
    _registry["enrich_c"] = "enrich_optype"
    snapshot_policy("enrich_b", "never")

    rt.trace_step(func=lambda df: df.assign(a=1), func_name="enrich_a",
                  raw_line="df=enrich_a(df)", args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(b=2), func_name="enrich_b",
                  raw_line="df=enrich_b(df)", args=[event_log.assign(a=1)], kwargs={})
    rt.trace_step(func=lambda df: df.assign(c=3), func_name="enrich_c",
                  raw_line="df=enrich_c(df)", args=[event_log.assign(a=1, b=2)], kwargs={})
    settle(rt)

    view = rt.build_pruned_view(group_by_category=True)

    assert len(view["edges"]) == 1
    edge = view["edges"][0]
    assert len(edge["collapsed_step_ids"]) == 3
    assert edge["category"] == "enrichment"


# ── 3. FC-1 -> FC-2: pruning a curated history FC-1 just materialized ───────

def test_pruned_view_of_materialized_curated_history_is_self_contained(rt, event_log):
    """build_pruned_view on a freshly materialize_curated_history() result
    must use that new history's own fresh step_ids, never the source
    history's — proving the "independent, never references source" claim
    from the pruning side, not just from materialization's own tests."""
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="source_step1",
                  raw_line="df=source_step1(df)", args=[event_log], kwargs={})
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="source_step2",
                  raw_line="df=source_step2(df)", args=[event_log.assign(x=1)], kwargs={})
    target = rt._current_state_id
    settle(rt)

    source_step_ids = {s["step_id"] for s in rt.storage.load_graph(rt._history.history_id)["steps"]}

    new_history_id = rt.create_independent_history_from_state(target, name="pruned-of-curated")

    new_storage = rt.storage
    new_view = _build_pruned_view_for_history(new_storage, new_history_id)
    new_step_ids = {e["step_id"] for e in new_view["edges"] if e["step_id"]}

    assert new_step_ids, "expected at least one non-collapsed edge with a step_id"
    assert new_step_ids.isdisjoint(source_step_ids)


def _build_pruned_view_for_history(storage, history_id: str) -> dict:
    """Call the module-level pruning renderer directly against an arbitrary
    history_id — RuntimeTracker.build_pruned_view() only ever targets
    self._history, so there's no public API for "prune a different,
    already-materialized history" from a live tracker."""
    from tracker.pruning import _render_pruned_view

    return _render_pruned_view(storage, history_id, config={})


# ── 4. FC-2 hidden branch + FC-1 replay target on that branch ───────────────

def test_replay_path_unaffected_by_pruned_view_hiding_its_branch(rt, event_log):
    """Hiding a branch in a pruned view is presentation-only (view-level) —
    find_shortest_replay_path must still work normally for a state on that
    hidden branch, proving the two features stay decoupled. branch_step is
    marked never-snapshot so the chain is forced to walk back across the
    branch boundary to trunk_step — otherwise (every step snapshotted by
    default) the chain would trivially be branch_step alone and never touch
    the branch boundary at all."""
    snapshot_policy("branch_step", "never")

    rt.trace_step(func=lambda df: df.assign(x=1), func_name="trunk_step",
                  raw_line="df=trunk_step(df)", args=[event_log], kwargs={})
    trunk_state = rt._current_state_id
    settle(rt)

    branch = rt.checkout(trunk_state, branch_name="experiment")
    rt.trace_step(func=lambda df: df.assign(y=2), func_name="branch_step",
                  raw_line="df=branch_step(df)", args=[event_log.assign(x=1)], kwargs={})
    branch_target = rt._current_state_id
    settle(rt)

    # Baseline, computed before the branch is ever hidden from anything.
    baseline_step_ids = rt.find_shortest_replay_path(branch_target)

    view = rt.build_pruned_view(hidden_branch_ids=[branch.branch_id])
    hidden_ids = {n["state_id"] for n in view["nodes"]}
    assert branch_target not in hidden_ids  # confirm it really is hidden from the view

    # The hidden view must have zero effect on FC-1 reading the live history
    # directly — same result before and after the pruned view existed, and
    # it must correctly cross the branch boundary back to trunk_step.
    step_ids = rt.find_shortest_replay_path(branch_target)
    assert step_ids == baseline_step_ids
    assert [rt.describe_step(sid)["func_name"] for sid in step_ids] == [
        "trunk_step", "branch_step",
    ]


# ── 5. Multi-branch pruning: cascade-hide doesn't leak across siblings ──────

def test_cascade_hide_does_not_leak_into_sibling_branches(rt, event_log):
    """Three branches off the same trunk state; hide one mid-branch state.
    Only that branch's downstream subtree should disappear — the other two
    sibling branches (and the shared trunk) must be completely unaffected."""
    rt.trace_step(func=lambda df: df.assign(x=1), func_name="trunk",
                  raw_line="df=trunk(df)", args=[event_log], kwargs={})
    trunk_state = rt._current_state_id
    settle(rt)

    branch_leaves = []
    for i in range(3):
        rt.checkout(trunk_state, branch_name=f"branch{i}")  # fork from the shared trunk
        rt.trace_step(
            func=lambda df, i=i: df.assign(**{f"branch{i}": i}),
            func_name=f"branch{i}_step",
            raw_line=f"df=branch{i}_step(df)", args=[event_log.assign(x=1)], kwargs={},
        )
        branch_leaves.append(rt._current_state_id)
    settle(rt)

    # Hide branch 1's leaf only.
    view = rt.build_pruned_view(hidden_state_ids=[branch_leaves[1]])
    visible_ids = {n["state_id"] for n in view["nodes"]}

    assert trunk_state in visible_ids
    assert branch_leaves[0] in visible_ids
    assert branch_leaves[2] in visible_ids
    assert branch_leaves[1] not in visible_ids
