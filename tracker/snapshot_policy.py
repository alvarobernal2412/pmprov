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
should_snapshot(func_name, operation_type_name) checks, in order:
  1. Exact func_name match in the per-function registry.
  2. Trailing dotted segment of func_name (e.g. "df.assign" -> "assign"),
     matching operation_registry.lookup's own fallback convention.
  3. operation_type_name match in the per-type registry.
  4. Default: True (always snapshot — preserves pre-FC-3 behavior for anything
     unconfigured).
"""
from __future__ import annotations

_VALID_MODES = {"always", "never"}

_func_policy: dict[str, str] = {}
_type_policy: dict[str, str] = {}


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


def should_snapshot(func_name: str, operation_type_name: str) -> bool:
    """Return whether func_name's output should be persisted as an artifact."""
    if func_name in _func_policy:
        return _func_policy[func_name] == "always"

    tail = func_name.rsplit(".", 1)[-1]
    if tail in _func_policy:
        return _func_policy[tail] == "always"

    if operation_type_name in _type_policy:
        return _type_policy[operation_type_name] == "always"

    return True
