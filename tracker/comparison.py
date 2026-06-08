"""
State and history comparison methods for RuntimeTracker.

Imported as a side-effect from tracker/__init__.py.
Patches compare_states(), compare_histories(), register_abstraction(),
apply_abstractions(), and compare_states_abstracted() onto RuntimeTracker.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker


def _compare_states(
    self: "RuntimeTracker",
    state_id_a: str,
    state_id_b: str,
) -> dict:
    """
    Compare two states structurally using their Parquet artifacts.

    Returns a dict with keys:
        common_columns  – columns present in both (sorted)
        unique_to_a     – columns only in state A (sorted)
        unique_to_b     – columns only in state B (sorted)
        shape_a         – (rows, cols) tuple for state A
        shape_b         – (rows, cols) tuple for state B
        dtype_diffs     – {col: (dtype_a, dtype_b)} for mismatched dtypes
        row_count_diff  – rows_a - rows_b
    Returns {"error": "...", "common_columns": []} when neither state has an artifact.
    """
    import pandas as pd

    def _load(state_id: str):
        ast_id = self.storage.load_output_artifact_state_id(state_id)
        if not ast_id:
            return None
        content_ref = self.storage.load_artifact_path(ast_id)
        if content_ref and Path(content_ref).exists():
            return pd.read_parquet(content_ref)
        return None

    df_a = _load(state_id_a)
    df_b = _load(state_id_b)

    if df_a is None and df_b is None:
        return {"error": "Neither state has a recorded Parquet artifact.", "common_columns": []}

    cols_a = set(df_a.columns if df_a is not None else [])
    cols_b = set(df_b.columns if df_b is not None else [])
    common = sorted(cols_a & cols_b)
    unique_a = sorted(cols_a - cols_b)
    unique_b = sorted(cols_b - cols_a)

    dtype_diffs: dict[str, tuple] = {}
    if df_a is not None and df_b is not None:
        for col in common:
            ta = str(df_a[col].dtype)
            tb = str(df_b[col].dtype)
            if ta != tb:
                dtype_diffs[col] = (ta, tb)

    shape_a = tuple(df_a.shape) if df_a is not None else (0, 0)
    shape_b = tuple(df_b.shape) if df_b is not None else (0, 0)

    return {
        "common_columns": common,
        "unique_to_a": unique_a,
        "unique_to_b": unique_b,
        "shape_a": shape_a,
        "shape_b": shape_b,
        "dtype_diffs": dtype_diffs,
        "row_count_diff": shape_a[0] - shape_b[0],
    }


def _compare_histories(
    self: "RuntimeTracker",
    other: "RuntimeTracker",
    group_by: str = "operation",
) -> dict:
    """
    Compare two RuntimeTracker sessions.

    Parameters
    ----------
    other:
        The second RuntimeTracker to compare against.
    group_by:
        ``"operation"`` (default) — compare by individual func_name.
        ``"category"`` — compare by StepCategory name.

    Returns a dict with shared_operations, unique_to_a, unique_to_b,
    step_count_a, step_count_b, divergent_branches, summary.
    When group_by="category", also returns category_summary_a and category_summary_b.
    """
    def _step_count(rt: "RuntimeTracker") -> int:
        return len(rt.storage.load_graph(history_id=rt._history.history_id)["steps"])

    def _divergent_branches(rt: "RuntimeTracker") -> list[str]:
        return [b["branch_id"] for b in rt.storage.load_branches(rt._history.history_id)
                if b["divergence_point_id"] is not None]

    count_a = _step_count(self)
    count_b = _step_count(other)
    div_branches = _divergent_branches(self) + _divergent_branches(other)

    if group_by == "category":
        from collections import Counter
        rows_a = self.storage.load_operations_by_category(self._history.history_id)
        rows_b = other.storage.load_operations_by_category(other._history.history_id)
        cats_a = {r["category"] for r in rows_a}
        cats_b = {r["category"] for r in rows_b}
        shared = sorted((cats_a & cats_b) - {None}, key=lambda x: x or "")
        unique_a = sorted((cats_a - cats_b) - {None}, key=lambda x: x or "")
        unique_b = sorted((cats_b - cats_a) - {None}, key=lambda x: x or "")
        cat_summary_a = dict(Counter(r["category"] for r in rows_a))
        cat_summary_b = dict(Counter(r["category"] for r in rows_b))
        summary = (
            f"HistoryComparison (by category):\n"
            f"  Shared categories: {shared}\n"
            f"  Unique to A: {unique_a}\n"
            f"  Unique to B: {unique_b}\n"
            f"  Steps A: {count_a} {cat_summary_a}, Steps B: {count_b} {cat_summary_b}\n"
            f"  Divergent branches: {len(div_branches)}"
        )
        return {
            "shared_operations": shared,
            "unique_to_a": unique_a,
            "unique_to_b": unique_b,
            "step_count_a": count_a,
            "step_count_b": count_b,
            "divergent_branches": div_branches,
            "category_summary_a": cat_summary_a,
            "category_summary_b": cat_summary_b,
            "summary": summary,
        }

    graph_a = self.storage.load_graph(history_id=self._history.history_id)
    graph_b = other.storage.load_graph(history_id=other._history.history_id)
    ops_a = {st["func_name"] for st in graph_a["steps"]}
    ops_b = {st["func_name"] for st in graph_b["steps"]}
    shared = sorted(ops_a & ops_b)
    unique_a = sorted(ops_a - ops_b)
    unique_b = sorted(ops_b - ops_a)
    summary = (
        f"HistoryComparison (by operation):\n"
        f"  Shared operations: {len(shared)}\n"
        f"  Unique to A: {len(unique_a)}\n"
        f"  Unique to B: {len(unique_b)}\n"
        f"  Steps A: {count_a}, Steps B: {count_b}\n"
        f"  Divergent branches: {len(div_branches)}"
    )
    return {
        "shared_operations": shared,
        "unique_to_a": unique_a,
        "unique_to_b": unique_b,
        "step_count_a": count_a,
        "step_count_b": count_b,
        "divergent_branches": div_branches,
        "summary": summary,
    }


def _register_abstraction(
    self: "RuntimeTracker",
    name: str,
    fn: Any,
    overwrite: bool = False,
) -> None:
    """
    Register an abstraction function under *name*.

    Parameters
    ----------
    name:
        Identifier (e.g. "row_count", "numeric_means").
    fn:
        Callable with signature ``(df: DataFrame, state_id: str) -> Any``.
    overwrite:
        If False (default), raise ValueError when *name* is already registered.
    """
    if name in self._abstraction_registry and not overwrite:
        raise ValueError(
            f"Abstraction '{name}' already registered. Pass overwrite=True to replace it."
        )
    self._abstraction_registry[name] = fn


def _apply_abstractions(self: "RuntimeTracker", state_id: str) -> None:
    """
    Run all registered abstraction functions against the artifact for *state_id*
    and cache results in ``rt._abstraction_cache[state_id]``.
    Silently skips when the state has no Parquet artifact.
    """
    import pandas as pd

    ast_id = self.storage.load_output_artifact_state_id(state_id)
    content_ref = self.storage.load_artifact_path(ast_id) if ast_id else None
    if not content_ref or not Path(content_ref).exists():
        self._abstraction_cache[state_id] = {}
        return
    path = Path(content_ref)

    try:
        df = pd.read_parquet(str(path))
    except Exception:
        self._abstraction_cache[state_id] = {}
        return

    results: dict[str, Any] = {}
    for name, fn in self._abstraction_registry.items():
        try:
            results[name] = fn(df, state_id)
        except Exception:
            results[name] = None

    self._abstraction_cache[state_id] = results


def _compare_states_abstracted(
    self: "RuntimeTracker",
    state_id_a: str,
    state_id_b: str,
) -> dict:
    """
    Compare two states using all registered abstractions.
    Calls apply_abstractions for each state if not already cached.
    Returns a dict where each key is an abstraction name and the value is
    {"a": result_for_a, "b": result_for_b}, plus a "summary" string.
    """
    if state_id_a not in self._abstraction_cache:
        self.apply_abstractions(state_id_a)
    if state_id_b not in self._abstraction_cache:
        self.apply_abstractions(state_id_b)

    cache_a = self._abstraction_cache.get(state_id_a, {})
    cache_b = self._abstraction_cache.get(state_id_b, {})
    all_keys = sorted(set(cache_a) | set(cache_b))

    result: dict[str, Any] = {}
    for key in all_keys:
        result[key] = {"a": cache_a.get(key), "b": cache_b.get(key)}

    lines = [f"AbstractionComparison ({state_id_a[:8]} vs {state_id_b[:8]}):"]
    for key in all_keys:
        equal = cache_a.get(key) == cache_b.get(key)
        lines.append(f"  {key}: {'equal' if equal else 'different'}")
    result["summary"] = "\n".join(lines)
    return result


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.compare_states = _compare_states
RuntimeTracker.compare_histories = _compare_histories
RuntimeTracker.register_abstraction = _register_abstraction
RuntimeTracker.apply_abstractions = _apply_abstractions
RuntimeTracker.compare_states_abstracted = _compare_states_abstracted
