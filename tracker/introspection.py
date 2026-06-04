"""
State introspection, branch listing, and replay methods for RuntimeTracker.

Imported as a side-effect from tracker/__init__.py.
Patches describe_state(), list_branches(), replay_state(), replay_pipeline() onto RuntimeTracker.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker


def _describe_state(self: "RuntimeTracker", state_id: str) -> dict:
    """
    Return full metadata for a recorded analysis state.

    Returns a dict with keys: state_id, func_name, raw_line, timestamp,
    branch_name, operation, agent, environment, params, delta.
    Returns {} if state_id is the root state (no producing step).

    Parameters
    ----------
    state_id:
        The UUID of the state to describe. Pass rt._current_state_id
        to describe the most recent step.
    """
    return self.storage.load_state_detail(state_id)


def _list_branches(self: "RuntimeTracker") -> Any:
    """
    Return a DataFrame listing all branches in the current session.

    Columns
    -------
    branch_id           UUID of the branch.
    name                Human-readable branch label.
    starts_at_state_id  State where this branch begins.
    step_count          Number of steps recorded on this branch.
    divergence_point_id State from which this branch diverged (None for main).
    """
    import pandas as pd
    rows = self.storage.load_branches(self._history.history_id)
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["branch_id", "name", "starts_at_state_id", "step_count", "divergence_point_id"]
    )

    try:
        from IPython import get_ipython
        if get_ipython() is None:
            print(df.to_string(index=False))
    except ImportError:
        print(df.to_string(index=False))

    return df


def _replay_state(self: "RuntimeTracker", state_id: str) -> Any:
    """
    Re-execute the step that produced state_id and record a new provenance node.

    Reconstructible parameter types: scalar (int/float/str/bool/None) and
    artifact references (DataFrames that were outputs of prior tracked steps).
    Lambda and list parameters cannot be reconstructed from stored source code;
    those arguments are silently omitted from the replay call.

    Returns the function output on success, or None if state_id has no
    recorded producing step or no reconstructible arguments.

    Parameters
    ----------
    state_id:
        UUID of the state to reproduce. Use rt.list_states() to find IDs.
    """
    detail = self.storage.load_state_detail(state_id)
    if not detail:
        return None

    func_name = detail["func_name"]
    raw_line = detail.get("raw_line", "")
    params = detail.get("params", [])

    # Reconstruct args from stored parameter values
    reconstructed_args: list[Any] = []
    for pv in params:
        value_type = pv["value_type"]
        value_data = pv["value"]

        if value_type == "scalar":
            reconstructed_args.append(value_data.get("value"))

        elif value_type == "artifact_state_ref":
            ast_id = value_data.get("artifact_state_id")
            if ast_id:
                path = self.storage.load_artifact_path(ast_id)
                if path:
                    try:
                        import pandas as pd
                        reconstructed_args.append(pd.read_parquet(path))
                    except Exception:
                        pass

        # lambda / list / dict: omit — cannot reliably reconstruct

    # All reconstructed args are passed positionally; replay of calls with
    # keyword-only args may produce different results.

    if not reconstructed_args:
        return None

    func = self._replay_func_registry.get(func_name)
    if func is None:
        return None

    return self.trace_step(
        func=func,
        func_name=func_name,
        raw_line=f"[replay] {raw_line}",
        args=reconstructed_args,
        kwargs={},
    )


def _replay_pipeline(
    self: "RuntimeTracker",
    pipeline_id: str,
    func_map: "dict[str, Any]",
    initial_input: Any = None,
    param_overrides: "dict[str, dict] | None" = None,
) -> dict:
    """
    Re-execute all steps recorded in a pipeline on new input data.

    Parameters
    ----------
    pipeline_id:
        UUID returned by rt.create_pipeline().
    func_map:
        Mapping of func_name → callable. Required because function objects
        cannot be reconstructed from stored names alone.
    initial_input:
        The DataFrame to feed as the first argument of the first step.
    param_overrides:
        Optional {func_name: {kwarg_name: new_value}} to override kwargs during replay.

    Returns
    -------
    dict with keys "output" (last step's output, or None) and "errors" (list of str).
    """
    steps = self.storage.load_pipeline_steps(pipeline_id)
    if not steps:
        return {"output": None, "errors": ["pipeline not found or empty"]}

    overrides = param_overrides or {}
    errors: list[str] = []
    current_input = initial_input

    for step_info in steps:
        func_name = step_info["func_name"]
        func = func_map.get(func_name)
        if func is None:
            errors.append(f"No function provided for '{func_name}' in func_map")
            continue

        step_kwargs = dict(overrides.get(func_name, {}))

        try:
            output = self.trace_step(
                func=func,
                func_name=func_name,
                raw_line=f"[replay] {step_info.get('raw_line', func_name)}",
                args=[current_input] if current_input is not None else [],
                kwargs=step_kwargs,
            )
            current_input = output
        except Exception as exc:
            errors.append(f"{func_name}: {exc}")

    return {"output": current_input, "errors": errors}


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.describe_state = _describe_state
RuntimeTracker.list_branches = _list_branches
RuntimeTracker.replay_state = _replay_state
RuntimeTracker.replay_pipeline = _replay_pipeline
