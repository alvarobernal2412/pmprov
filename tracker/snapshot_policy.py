"""
Configurable snapshotting policy: decides whether a step's output gets persisted
as an artifact (Parquet snapshot), independent of whether the AnalysisStep itself
is recorded (docs/claude/checklist.md FC-3).

The AnalysisStep, AnalysisState, Operation, parameter values, and delta are ALWAYS
recorded by RuntimeTracker.trace_step regardless of this policy — skipping a
snapshot only affects whether the step's output can be loaded directly later, or
must be recomputed via replay (see FC-1's find_shortest_replay_path, which already
treats "no artifact" states as points requiring replay rather than assuming every
state has a snapshot).

Mirrors tracker/operation_registry.py's two-tier registration pattern.

Usage
-----
Per-function override (most specific)::

    from tracker import snapshot_policy
    snapshot_policy("df.head", "never")

Per-OperationType default (used when no func_name override matches)::

    from tracker import snapshot_policy_for_type
    snapshot_policy_for_type("attribute_derivation", "never")

Lookup
------
should_snapshot(func_name, operation_type_name, elapsed_seconds) checks, in order:
  1. Exact func_name match in the per-function registry.
  2. Trailing dotted segment of func_name (e.g. "df.assign" -> "assign"),
     matching operation_registry.lookup's own fallback convention.
  3. operation_type_name match in the per-type registry.
  4. Measured-cost default: only reached when NEITHER of the above applies.
     elapsed_seconds (this call's own wall-clock time, timed by trace_step)
     decides automatically — a call slower than
     set_default_snapshot_threshold()'s threshold gets snapshotted, a
     faster one doesn't, with no per-function declaration needed. The
     threshold defaults to 0.0, i.e. "always snapshot" — preserving
     pre-FC-3/pre-this-feature behavior for anything unconfigured, until an
     analyst explicitly opts into the timing-based default.

An explicit snapshot_policy()/snapshot_policy_for_type() registration always
wins over the measured-cost default — this only fills gaps, it never
overrides a deliberate decision.

Caveat: this is a per-run heuristic, not a deterministic contract like an
explicit policy — the same function can measure differently run to run
(machine load, input size, cache state), so whether a given state ends up
with an artifact can vary across sessions for anything relying on the
timing default. It also only measures THIS call's own cost, not how
expensive it would be to replay everything upstream of it if this snapshot
didn't exist.
"""
from __future__ import annotations

from typing import Optional

_VALID_MODES = {"always", "never"}

_func_policy: dict[str, str] = {}
_type_policy: dict[str, str] = {}

# Elapsed-time threshold (seconds) for the measured-cost default fallback —
# only consulted when neither an explicit per-function nor per-type policy
# matches. 0.0 preserves the original "always snapshot" default.
_default_snapshot_min_seconds: float = 0.0


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}")


def snapshot_policy(func_name: str, mode: str) -> None:
    """Register a per-function snapshot override. mode: "always" or "never"."""
    _validate_mode(mode)
    _func_policy[func_name] = mode


def snapshot_policy_for_type(operation_type_name: str, mode: str) -> None:
    """Register a per-OperationType snapshot default. mode: "always" or "never"."""
    _validate_mode(mode)
    _type_policy[operation_type_name] = mode


def set_default_snapshot_threshold(seconds: float) -> None:
    """
    Set the elapsed-time threshold (seconds) used by should_snapshot()'s
    measured-cost fallback, for any function with no explicit
    snapshot_policy()/snapshot_policy_for_type() registration.

    A call that takes at least *seconds* to run gets its output snapshotted
    automatically (treated as too expensive to casually recompute via
    replay); a faster call doesn't. Explicit registrations are never
    affected by this — they always take priority.

    Pass 0.0 (the default) to restore "always snapshot everything
    unconfigured", i.e. opt back out of the automatic behavior.
    """
    global _default_snapshot_min_seconds
    _default_snapshot_min_seconds = seconds


def should_snapshot(
    func_name: str,
    operation_type_name: str,
    elapsed_seconds: Optional[float] = None,
) -> bool:
    """Return whether func_name's output should be persisted as an artifact."""
    if func_name in _func_policy:
        return _func_policy[func_name] == "always"

    tail = func_name.rsplit(".", 1)[-1]
    if tail in _func_policy:
        return _func_policy[tail] == "always"

    if operation_type_name in _type_policy:
        return _type_policy[operation_type_name] == "always"

    if elapsed_seconds is not None:
        return elapsed_seconds >= _default_snapshot_min_seconds

    return True
