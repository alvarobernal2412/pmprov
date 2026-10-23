"""
Read-only review mode: step back and forth through an analysis like a debugger.

Imported as a side-effect from tracker/__init__.py.
Patches review() onto RuntimeTracker.

The reviewer is a cursor over one linear path of the history. It only reads:
it never calls checkout(), never updates active_state_id and never records a
step, so the stored provenance is left exactly as it was. To resume working
from a reviewed state, call ``reviewer.restore(checkout=True)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from models.annotations import AnnotatableType

if TYPE_CHECKING:
    from tracker.restore import RestoredState
    from tracker.runtime import RuntimeTracker


class HistoryReviewer:
    def __init__(self, tracker: "RuntimeTracker", state_id: Optional[str] = None) -> None:
        self._rt = tracker
        graph = tracker.storage.load_graph(history_id=tracker._history.history_id)
        self._parent = {s["output_state_id"]: s["input_state_id"] for s in graph["steps"]}
        self._branch = {s["state_id"]: s["branch_id"] for s in graph["states"]}
        self._timestamp = {s["state_id"]: s["timestamp"] for s in graph["states"]}
        self._root = tracker._root_state_id
        self._path: list[str] = []
        self._index = 0
        self._plan(state_id or tracker._current_state_id)

    def _plan(self, state_id: str) -> None:
        """Path = root -> latest state of the branch containing state_id, kept
        if state_id is already on the current path."""
        if state_id not in self._branch:
            raise ValueError(f"review: unknown state_id {state_id!r}")
        if state_id in self._path:
            self._index = self._path.index(state_id)
            return
        branch_id = self._branch[state_id]
        tip = max(
            (s for s, b in self._branch.items() if b == branch_id),
            key=lambda s: self._timestamp[s],
        )
        # the tip may sit on a later fork of this branch; keep state_id on the path
        tip = tip if self._is_ancestor(state_id, tip) else state_id
        chain = [tip]
        while chain[-1] in self._parent:
            chain.append(self._parent[chain[-1]])
        self._path = list(reversed(chain))
        self._index = self._path.index(state_id)

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        current: Optional[str] = descendant
        while current is not None:
            if current == ancestor:
                return True
            current = self._parent.get(current)
        return False

    # -- position ------------------------------------------------------
    @property
    def path(self) -> list[str]:
        return list(self._path)

    @property
    def index(self) -> int:
        return self._index

    @property
    def state_id(self) -> str:
        return self._path[self._index]

    @property
    def can_back(self) -> bool:
        return self._index > 0

    @property
    def can_forward(self) -> bool:
        return self._index < len(self._path) - 1

    # -- movement (return False at a boundary instead of raising) ------
    def back(self) -> bool:
        if not self.can_back:
            return False
        self._index -= 1
        return True

    def forward(self) -> bool:
        if not self.can_forward:
            return False
        self._index += 1
        return True

    def first(self) -> str:
        self._index = 0
        return self.state_id

    def last(self) -> str:
        self._index = len(self._path) - 1
        return self.state_id

    def goto(self, state_id: str) -> str:
        self._plan(state_id)
        return self.state_id

    # -- inspection ----------------------------------------------------
    def describe(self) -> dict:
        """describe_state() of the current position ({} for the root)."""
        return self._rt.storage.load_state_detail(self.state_id)

    def annotations(self) -> Any:
        return self._rt.list_annotations(AnnotatableType.ANALYSIS_STATE, self.state_id)

    def restore(self, **kwargs: Any) -> "RestoredState":
        """restore_state() for the current position (pass checkout=True to resume from it)."""
        return self._rt.restore_state(self.state_id, **kwargs)


def _review(self: "RuntimeTracker", state_id: Optional[str] = None) -> HistoryReviewer:
    """Open a read-only reviewer positioned at *state_id* (default: the current state)."""
    return HistoryReviewer(self, state_id)


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.review = _review
