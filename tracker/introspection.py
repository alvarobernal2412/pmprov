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


def _describe_step(self: "RuntimeTracker", step_id: str) -> dict:
    """
    Return full metadata for a recorded analysis step.

    Returns a dict with keys: step_id, output_state_id, func_name, raw_line,
    timestamp, branch_name, operation, agent, environment, params, delta.
    Returns {} if step_id is not found.

    Parameters
    ----------
    step_id:
        The UUID of the step to describe. Use rt.list_states() to find
        step_ids via the produced_by_step_id column.
    """
    return self.storage.load_step_detail(step_id)


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
    receiver: Any = None
    for pv in params:
        param_id = pv.get("param_id", "")
        suffix = param_id.split(":", 1)[1] if ":" in param_id else param_id
        value_type = pv["value_type"]
        value_data = pv["value"]

        if suffix == "__receiver__":
            if value_type == "artifact_state_ref":
                ast_id = value_data.get("artifact_state_id")
                if ast_id:
                    path = self.storage.load_artifact_path(ast_id)
                    if path:
                        try:
                            import pandas as pd
                            receiver = pd.read_parquet(path)
                        except Exception:
                            pass
            continue

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

    if not reconstructed_args and receiver is None:
        return None

    func = self._replay_func_registry.get(func_name)
    if func is None:
        return None

    # If the original call was a bound method, re-bind to the reconstructed receiver.
    if receiver is not None:
        method_name = func_name.rsplit(".", 1)[-1]
        bound_func = getattr(receiver, method_name, None)
        if bound_func is None:
            return None
        func = bound_func

    return self.trace_step(
        func=func,
        func_name=func_name,
        raw_line=f"[replay] {raw_line}",
        args=reconstructed_args,
        kwargs={},
    )


def _reconstruct_value(param: dict, storage: Any, replacements: dict) -> Any:
    """Reconstruct a single runtime value from a stored ParameterValue record."""
    vtype = param["value_type"]
    val = param["value"]
    if vtype == "scalar":
        return val.get("value")
    if vtype == "artifact_state_ref":
        ast_id = val.get("artifact_state_id")
        # Prefer a replacement produced earlier in this replay (new input data)
        if ast_id in replacements:
            return replacements[ast_id]
        return storage.load_artifact(ast_id)
    if vtype in ("list", "dict"):
        return val.get("value")
    # lambda_function: source code only — caller must supply via func_map/param_overrides
    return None


def _reconstruct_args(
    params: list[dict],
    func_name: str,
    initial_input: Any,
    is_first_step: bool,
    param_overrides: dict,
    storage: Any,
    replacements: dict,
) -> tuple[list, dict, Any]:
    """
    Rebuild positional args and kwargs for a step from its stored ParameterValue records.

    param_id format:  "<func_name>:arg_<i>"      →  positional index i
                      "<func_name>:<name>"        →  keyword argument
                      "<func_name>:__receiver__"  →  bound-method receiver (returned separately)

    Returns (args, kwargs, receiver) where receiver is the reconstructed object the
    method was called on, or None for standalone function calls.
    """
    positional: dict[int, Any] = {}
    keyword: dict[str, Any] = {}
    receiver: Any = None

    for p in params:
        suffix = p["param_id"].split(":", 1)[1] if ":" in p["param_id"] else p["param_id"]
        if suffix == "__receiver__":
            receiver = _reconstruct_value(p, storage, replacements)
            continue
        if suffix.startswith("arg_"):
            try:
                positional[int(suffix[4:])] = _reconstruct_value(p, storage, replacements)
            except ValueError:
                pass
        else:
            keyword[suffix] = _reconstruct_value(p, storage, replacements)

    # Build ordered positional list
    args: list = []
    if positional:
        for i in range(max(positional) + 1):
            args.append(positional.get(i))

    # Replace the first DataFrame positional arg with initial_input in the first step
    if is_first_step and initial_input is not None:
        if args:
            args[0] = initial_input
        else:
            args = [initial_input]

    # Apply analyst param_overrides as kwargs
    keyword.update(param_overrides)

    return args, keyword, receiver


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
        cannot be reconstructed from stored names alone. For bound-method steps
        (e.g. ``case_log.apply``), supply the already-bound method so the correct
        receiver is used: ``{"case_log.apply": new_case_log.apply}``.
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
RuntimeTracker.describe_step = _describe_step
RuntimeTracker.list_branches = _list_branches
RuntimeTracker.replay_state = _replay_state
RuntimeTracker.replay_pipeline = _replay_pipeline
