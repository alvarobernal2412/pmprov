"""
Recover the content of an AnalysisState ("continue from here").

Imported as a side-effect from tracker/__init__.py.
Patches restore_state() and restore_latest() onto RuntimeTracker.

Unlike replay_state(), which re-executes a step and records a NEW node,
restore_state() only reads: it loads the artifacts the immutable snapshot
holds and hands them back, optionally injecting them into a namespace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from tracker.logger import log_trace_warning
from tracker.runtime import _bound_name as _variable_name

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker


@dataclass
class RestoredState:
    """Result of restore_state()."""

    state_id: str
    variables: dict[str, Any] = field(default_factory=dict)
    superseded: dict[str, Any] = field(default_factory=dict)
    skipped: list[dict] = field(default_factory=list)


def _load(self: "RuntimeTracker", artifact_state_id: str) -> tuple[Any, str]:
    reason = "artifact content not found"
    try:
        obj = self.storage.load_artifact(artifact_state_id)
    except NotImplementedError as e:
        obj, reason = None, str(e)
    return obj, reason


def _restore_bound(self: "RuntimeTracker", bindings: dict[str, str],
                   names: Optional[list[str]], result: RestoredState) -> None:
    """Exact path: load what each name was bound to when the state was created."""
    wanted = list(bindings) if names is None else list(dict.fromkeys(names))
    for name in wanted:
        if name not in bindings:
            result.skipped.append({"name": name, "reason": "not bound at this state"})
            continue
        obj, reason = _load(self, bindings[name])
        if obj is None:
            log_trace_warning("restore_state: artifact skipped", step="restore_state",
                              artifact_state_id=bindings[name], reason=reason)
            result.skipped.append({"artifact_state_id": bindings[name], "name": name, "reason": reason})
            continue
        result.variables[name] = obj


def _restore_legacy(self: "RuntimeTracker", state_id: str, names: Optional[list[str]],
                    result: RestoredState) -> None:
    """States recorded before bindings existed: names parsed from raw_line."""
    arts = [(_variable_name(a["origin_raw_line"]) or a["name"], a)
            for a in self.storage.load_state_artifacts(state_id)]
    if names is not None:
        known = {n for n, _ in arts}
        result.skipped.extend({"name": n, "reason": "not bound at this state"}
                              for n in dict.fromkeys(names) if n not in known)
        arts = [(n, a) for n, a in arts if n in names]
    loaded: list[tuple[str, str, Any]] = []  # (name, artifact_id, obj), oldest origin first
    for name, art in arts:
        obj, reason = _load(self, art["artifact_state_id"])
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


def _restore_state(
    self: "RuntimeTracker",
    state_id: Optional[str] = None,
    *,
    into: Optional[dict] = None,
    checkout: bool = False,
    names: Optional[list[str]] = None,
) -> RestoredState:
    """
    Return the variables bound at *state_id* (default: the last state of the
    history by timestamp), exactly as they were when the state was created.

    Names and artifact versions come from the bindings recorded with the
    state. States recorded before bindings existed fall back to parsing the
    producing step's raw line (``superseded`` is only filled on that path).
    Variables that cannot be loaded are listed in ``skipped``.

    names:
        Load only these variables. Names not bound at the state go to
        ``skipped`` instead of raising.
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
    bindings = self.storage.load_state_bindings(state_id)
    if bindings:
        _restore_bound(self, bindings, names, result)
    else:
        _restore_legacy(self, state_id, names, result)

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
