"""
Structured, log-style export of a whole AnalysisHistory.

Imported as a side-effect from tracker/__init__.py.
Patches get_history() and export_history() onto RuntimeTracker. Both are
read-only: they never create provenance nodes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from models.annotations import AnnotatableType

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker

_FORMATS = ("json", "jsonl")


def _notes_for(storage: Any, target_type: AnnotatableType, target_id: str) -> dict:
    return {
        "annotations": [
            {k: a[k] for k in ("annotation_id", "text", "agent_id", "created_at")}
            for a in storage.load_annotations(target_type.value, target_id)
        ],
        "tags": [t["name"] for t in storage.load_tags(target_type.value, target_id)],
    }


def _get_history(self: "RuntimeTracker", history_id: Optional[str] = None) -> dict:
    """
    Return the whole analysis as a plain, JSON-serialisable dict::

        {"history": {...}, "branches": [...], "events": [...]}

    ``events`` has one entry per recorded step, in timestamp order. Each carries
    the step/state ids, branch, function, raw line, operation, agent,
    environment, parameters, delta, plus the annotations and tags attached to
    its step and output state.
    """
    history_id = history_id or self._history.history_id
    storage = self.storage
    history_row = storage.load_history(history_id)
    if history_row is None:
        raise ValueError(f"no such history: {history_id!r}")

    steps = sorted(storage.load_graph(history_id=history_id)["steps"], key=lambda s: s["timestamp"])
    events = []
    for sequence, step in enumerate(steps):
        detail = storage.load_state_detail(step["output_state_id"])
        events.append({
            "sequence": sequence,
            "timestamp": step["timestamp"],
            "step_id": step["step_id"],
            "input_state_id": step["input_state_id"],
            "output_state_id": step["output_state_id"],
            "branch": detail.get("branch_name"),
            "func_name": step["func_name"],
            "raw_line": step["raw_line"],
            "operation": detail.get("operation"),
            "agent": detail.get("agent"),
            "environment": detail.get("environment"),
            "params": detail.get("params", []),
            "delta": detail.get("delta"),
            "step_notes": _notes_for(storage, AnnotatableType.STEP, step["step_id"]),
            "state_notes": _notes_for(storage, AnnotatableType.ANALYSIS_STATE, step["output_state_id"]),
        })

    return {
        "history": {
            **history_row,
            "root_state_id": storage.find_root_state_id(history_id),
            **_notes_for(storage, AnnotatableType.ANALYSIS_HISTORY, history_id),
        },
        "branches": storage.load_branches(history_id),
        "events": events,
    }


def _export_history(
    self: "RuntimeTracker",
    path: "str | Path",
    format: str = "jsonl",
    history_id: Optional[str] = None,
) -> Path:
    """
    Write the structured history log (see get_history) to *path*.

    format="json"  : one JSON document.
    format="jsonl" : one JSON object per line, each with a "type" of
                     "history", "branch" or "event" -- streamable log form.
    """
    if format not in _FORMATS:
        raise ValueError(f"export_history: format must be one of {_FORMATS}, got {format!r}")
    data = self.get_history(history_id)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if format == "json":
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
        else:
            records = (
                [{"type": "history", **data["history"]}]
                + [{"type": "branch", **b} for b in data["branches"]]
                + [{"type": "event", **e} for e in data["events"]]
            )
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
    return path


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.get_history = _get_history
RuntimeTracker.export_history = _export_history
