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

import uuid
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


def _collapsed_label(category: Optional[str], count: int) -> str:
    """Presentation label for a synthetic edge collapsing `count` same-category steps."""
    return f"[{category or 'uncategorized'}] {count} steps"


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
                else _collapsed_label(categories_by_step_id.get(chain[0]["step_id"]), len(chain)),
            "category": categories_by_step_id.get(chain[0]["step_id"]),
            "collapsed_step_ids": [c["step_id"] for c in chain],
        })

    return {"nodes": visible_states, "edges": edges}


def _resolve_hidden_state_ids(
    storage: Any,
    history_id: str,
    hidden_state_ids: Optional[list],
    hidden_step_ids: Optional[list],
    hidden_branch_ids: Optional[list],
) -> set:
    """Combine all three hide-inputs into one cascaded set of hidden state_ids."""
    graph = storage.load_graph(history_id=history_id)
    steps = graph["steps"]
    states = graph["states"]

    seed = set(hidden_state_ids or [])

    if hidden_step_ids:
        step_by_id = {st["step_id"]: st for st in steps}
        for step_id in hidden_step_ids:
            st = step_by_id.get(step_id)
            if st:
                seed.add(st["output_state_id"])

    if hidden_branch_ids:
        branch_set = set(hidden_branch_ids)
        for s in states:
            if s.get("branch_id") in branch_set:
                seed.add(s["state_id"])

    return _cascade_hidden_state_ids(steps, seed)


def _render_pruned_view(storage: Any, history_id: str, config: dict) -> dict:
    """Shared rendering path for build_pruned_view and load_pruned_view."""
    graph = storage.load_graph(history_id=history_id)
    states, steps = graph["states"], graph["steps"]

    hidden_state_ids = _resolve_hidden_state_ids(
        storage, history_id,
        config.get("hidden_state_ids"), config.get("hidden_step_ids"),
        config.get("hidden_branch_ids"),
    )

    categories_by_step_id = {
        row["func_name"]: row["category"]
        for row in storage.load_operations_by_category(history_id)
    }
    # load_operations_by_category is keyed by func_name, not step_id; re-key by
    # step_id using the step list so collapse logic can look up per-step categories.
    func_name_by_step_id = {st["step_id"]: st["func_name"] for st in steps}
    categories_by_step_id = {
        step_id: categories_by_step_id.get(func_name)
        for step_id, func_name in func_name_by_step_id.items()
    }

    pruned = _compute_pruned_graph(
        states, steps, categories_by_step_id,
        group_by_category=bool(config.get("group_by_category")),
        hidden_state_ids=hidden_state_ids,
    )
    pruned["config"] = {
        "group_by_category": bool(config.get("group_by_category")),
        "hidden_state_ids": config.get("hidden_state_ids") or [],
        "hidden_step_ids": config.get("hidden_step_ids") or [],
        "hidden_branch_ids": config.get("hidden_branch_ids") or [],
    }
    return pruned


def _build_pruned_view(
    self: "RuntimeTracker",
    group_by_category: bool = False,
    hidden_state_ids: Optional[list] = None,
    hidden_step_ids: Optional[list] = None,
    hidden_branch_ids: Optional[list] = None,
) -> dict:
    """
    Compute a pruned/curated, presentation-only view of this tracker's current
    AnalysisHistory: hide dead-end subtrees/branches and optionally collapse
    consecutive same-StepCategory operations into single edges.

    Never mutates the source history. Every collapsed edge carries
    collapsed_step_ids so drill-down to the original operations is always
    possible (docs/claude/checklist.md FC-2).

    Returns {"nodes": [...], "edges": [...], "config": {...}} — pass the return
    value directly to save_pruned_view() to persist the configuration.
    """
    config = {
        "group_by_category": group_by_category,
        "hidden_state_ids": hidden_state_ids or [],
        "hidden_step_ids": hidden_step_ids or [],
        "hidden_branch_ids": hidden_branch_ids or [],
    }
    return _render_pruned_view(self.storage, self._history.history_id, config)


def _save_pruned_view(self: "RuntimeTracker", pruned_view: dict, name: str) -> str:
    """
    Persist a pruned view's configuration (not its computed nodes/edges) so it can
    be reloaded and re-rendered against the live source history later.

    Parameters
    ----------
    pruned_view:
        The dict returned by build_pruned_view() — only pruned_view["config"] is
        stored.
    name:
        Human-readable name for the saved view.
    """
    view_id = str(uuid.uuid4())
    self.storage.save_pruned_view_sync(
        view_id, self._history.history_id, name, pruned_view["config"]
    )
    return view_id


def _load_pruned_view(self: "RuntimeTracker", view_id: str) -> dict:
    """
    Reload a saved pruned view by re-applying its saved configuration against the
    live source history (never a stored duplicate — see module docstring).

    Raises ValueError if view_id is unknown.
    """
    record = self.storage.load_pruned_view_config(view_id)
    if record is None:
        raise ValueError(f"No pruned view found for view_id={view_id!r}")
    return _render_pruned_view(self.storage, record["history_id"], record["config"])


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.build_pruned_view = _build_pruned_view
RuntimeTracker.save_pruned_view = _save_pruned_view
RuntimeTracker.load_pruned_view = _load_pruned_view
