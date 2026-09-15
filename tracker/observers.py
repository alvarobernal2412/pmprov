"""
Runtime step observers for RuntimeTracker.

Imported as a side-effect from tracker/__init__.py.
Patches on_step() onto RuntimeTracker.

Same-process, in-memory alternative to reading provenance back out of
storage (DuckDB or otherwise): a callback registered via on_step() fires
synchronously right after trace_step()/trace_ui_step() records a new step,
with a detail dict built entirely from data already in hand at the call
site -- deliberately NOT by reading the step back via describe_state()/
storage, since trace_step()'s own persistence (save_state_async etc.) is
fire-and-forget async and may not have landed yet. No polling, no DB read,
no file-lock contention with another process reading the same .pmprov
directory.

This is deliberately NOT a message bus or pub/sub system: callbacks run
synchronously, in-process, on the same thread that called trace_step(). A
slow or blocking callback delays the notebook cell that triggered it.
Keep callbacks fast (e.g. just flip a flag or trigger a widget refresh);
do real work asynchronously if you need to.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from tracker.logger import log_trace_warning

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker

StepObserver = Callable[[dict], None]


def _on_step(self: "RuntimeTracker", callback: StepObserver) -> None:
    """
    Register *callback* to be called with a step's detail dict every time
    trace_step() or trace_ui_step() records a new step on this tracker.

    Multiple callbacks may be registered; there is currently no way to
    unregister one (call sites that re-run, e.g. an interactively re-run
    marimo cell, will accumulate duplicate callbacks — same known caveat as
    ProvenancePanel's auto-refresh thread in marimo-pmprov).

    The detail dict has keys: state_id, step_id, func_name, raw_line,
    branch_name, params, delta. It is NOT guaranteed identical to
    describe_state()'s shape (no operation/agent/environment sub-dicts) —
    it's built purely from in-memory data at the call site, not a storage
    read, precisely so it never races the async write queue.
    """
    self._step_observers.append(callback)


def _notify_step_observers(
    self: "RuntimeTracker",
    *,
    state_id: str,
    step_id: str,
    func_name: str,
    raw_line: str,
    branch_name: str,
    params: list[dict],
    delta: Optional[dict] = None,
) -> None:
    """Call every registered on_step() callback with this step's detail.
    Exceptions are isolated per-callback (logged, not raised) so a broken
    observer can never break provenance tracking itself."""
    if not self._step_observers:
        return
    detail = {
        "state_id": state_id,
        "step_id": step_id,
        "func_name": func_name,
        "raw_line": raw_line,
        "branch_name": branch_name,
        "params": params,
        "delta": delta or {},
    }
    for callback in self._step_observers:
        try:
            callback(detail)
        except Exception as e:
            log_trace_warning("step observer raised", step="notify_step_observers", error=e)


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.on_step = _on_step
RuntimeTracker._notify_step_observers = _notify_step_observers
