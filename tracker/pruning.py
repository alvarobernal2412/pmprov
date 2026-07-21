"""
Pruned/curated view computation for RuntimeTracker (FC-2, docs/claude/checklist.md).

Imported as a side-effect from tracker/__init__.py.
Patches build_pruned_view(), save_pruned_view(), load_pruned_view() onto RuntimeTracker.

A pruned view is presentation metadata over the full provenance record, never a lossy
rewrite of it: every collapsed edge always carries collapsed_step_ids so drill-down
back to the original operations is always possible. Saved views persist only the
pruning configuration (docs/claude/checklist.md FC-2's resolved design decision), not a
duplicate of the states/steps themselves — rendering always re-reads the source
history live.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional
from collections import defaultdict

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker


def _cascade_hidden_state_ids(steps: list[dict], seed_hidden_state_ids: set) -> set:
    """
    Expand a seed set of hidden state_ids to include every state reachable
    downstream from any seed (the domain model is a tree — see
    docs/claude/domain-model.md — so hiding a state hides its whole subtree).
    """
    children: dict[str, list[dict]] = defaultdict(list)
    for st in steps:
        children[st["input_state_id"]].append(st)

    hidden = set(seed_hidden_state_ids)
    stack = list(seed_hidden_state_ids)
    while stack:
        sid = stack.pop()
        for st in children.get(sid, []):
            out = st["output_state_id"]
            if out not in hidden:
                hidden.add(out)
                stack.append(out)
    return hidden


def _compute_pruned_graph(
    states: list[dict],
    steps: list[dict],
    categories_by_step_id: dict,
    group_by_category: bool,
    hidden_state_ids: set,
) -> dict:
    """
    Filter out hidden states/steps, then (if group_by_category) collapse maximal
    runs of consecutive same-category steps along non-branching chains.

    Returns {"nodes": [...state dicts...], "edges": [...]} where every edge has
    step_id (None if collapsed), input_state_id, output_state_id, func_name,
    category, and collapsed_step_ids (always present, len 1 when not collapsed).
    """
    visible_states = [s for s in states if s["state_id"] not in hidden_state_ids]
    visible_steps = [
        st for st in steps
        if st["input_state_id"] not in hidden_state_ids
        and st["output_state_id"] not in hidden_state_ids
    ]

    if not group_by_category:
        edges = [
            {
                "step_id": st["step_id"],
                "input_state_id": st["input_state_id"],
                "output_state_id": st["output_state_id"],
                "func_name": st["func_name"],
                "category": categories_by_step_id.get(st["step_id"]),
                "collapsed_step_ids": [st["step_id"]],
            }
            for st in visible_steps
        ]
        return {"nodes": visible_states, "edges": edges}

    children: dict[str, list[dict]] = defaultdict(list)
    parent_step_by_state: dict[str, dict] = {}
    for st in visible_steps:
        children[st["input_state_id"]].append(st)
        parent_step_by_state[st["output_state_id"]] = st

    def is_head(st: dict) -> bool:
        parent_step = parent_step_by_state.get(st["input_state_id"])
        if parent_step is None:
            return True  # input state is the root (or a hidden-subtree cut point)
        same_category = (
            categories_by_step_id.get(parent_step["step_id"])
            == categories_by_step_id.get(st["step_id"])
        )
        single_child = len(children[st["input_state_id"]]) == 1
        return not (same_category and single_child)

    edges = []
    for st in visible_steps:
        if not is_head(st):
            continue
        chain = [st]
        current = st
        while True:
            kids = children.get(current["output_state_id"], [])
            if len(kids) != 1:
                break
            nxt = kids[0]
            if categories_by_step_id.get(nxt["step_id"]) != categories_by_step_id.get(current["step_id"]):
                break
            chain.append(nxt)
            current = nxt
        edges.append({
            "step_id": chain[0]["step_id"] if len(chain) == 1 else None,
            "input_state_id": chain[0]["input_state_id"],
            "output_state_id": chain[-1]["output_state_id"],
            "func_name": chain[0]["func_name"] if len(chain) == 1
                else f"[{categories_by_step_id.get(chain[0]['step_id']) or 'uncategorized'}] {len(chain)} steps",
            "category": categories_by_step_id.get(chain[0]["step_id"]),
            "collapsed_step_ids": [c["step_id"] for c in chain],
        })

    return {"nodes": visible_states, "edges": edges}
