"""
Recover the content of an AnalysisState ("continue from here").

Imported as a side-effect from tracker/__init__.py.
Patches restore_state() and restore_latest() onto RuntimeTracker.

Unlike replay_state(), which re-executes a step and records a NEW node,
restore_state() only reads: it loads the artifacts the immutable snapshot
holds and hands them back, optionally injecting them into a namespace.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from tracker.logger import log_trace_warning

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker


@dataclass
class RestoredState:
    """Result of restore_state()."""

    state_id: str
    variables: dict[str, Any] = field(default_factory=dict)
    superseded: dict[str, Any] = field(default_factory=dict)
    skipped: list[dict] = field(default_factory=list)


def _variable_name(raw_line: Optional[str]) -> Optional[str]:
    """Name bound by a single-target ``name = call(...)`` line, else None."""
    if not raw_line:
        return None
    try:
        node = ast.parse(raw_line.strip()).body[0]
    except (SyntaxError, IndexError):
        return None
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _restore_state(
    self: "RuntimeTracker",
    state_id: Optional[str] = None,
    *,
    into: Optional[dict] = None,
    checkout: bool = False,
) -> RestoredState:
    """
    Load every artifact held by *state_id* (default: the last state of the
    history by timestamp) and return them keyed by the variable name the
    analyst assigned them to.

    Variable names come from the producing step's raw line; unnamed artifacts
    keep their artifact name. If several artifacts share a name, the most
    recently produced one is in ``variables`` and the rest in ``superseded``.
    Artifacts that cannot be loaded are listed in ``skipped``.

    into:
        A dict (e.g. ``globals()``) updated with the variables.
    checkout:
        Also move the analysis head to *state_id* so the next step continues
        from it (the existing checkout() semantics).
    """
    if state_id is None:
        states = self.storage.load_states_rich(self._history.history_id)
        if not states:
            raise ValueError("restore_state: history has no states")
        state_id = states[-1]["state_id"]
    elif self.storage.load_state_branch_id(state_id) is None:
        raise ValueError(f"restore_state: unknown state_id {state_id!r}")

    result = RestoredState(state_id=state_id)
    loaded: list[tuple[str, str, Any]] = []  # (name, artifact_id, obj), oldest origin first
    for art in self.storage.load_state_artifacts(state_id):
        name = _variable_name(art["origin_raw_line"]) or art["name"]
        reason = "artifact content not found"
        try:
            obj = self.storage.load_artifact(art["artifact_state_id"])
        except NotImplementedError as e:
            obj = None
            reason = str(e)
        if obj is None:
            log_trace_warning("restore_state: artifact skipped", step="restore_state",
                              artifact_id=art["artifact_id"], reason=reason)
            result.skipped.append({"artifact_id": art["artifact_id"], "name": name, "reason": reason})
            continue
        loaded.append((name, art["artifact_id"], obj))

    for name, artifact_id, obj in loaded:  # oldest origin first, so later ones win
        if name in result.variables:
            result.superseded[f"{name}@{artifact_id[:8]}"] = result.variables[name]
        result.variables[name] = obj

    if into is not None:
        into.update(result.variables)
    if checkout:
        self.checkout(state_id)
    return result


def _restore_latest(self: "RuntimeTracker", **kwargs: Any) -> RestoredState:
    """restore_state() for the last state of the history by timestamp."""
    return _restore_state(self, None, **kwargs)


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.restore_state = _restore_state
RuntimeTracker.restore_latest = _restore_latest
