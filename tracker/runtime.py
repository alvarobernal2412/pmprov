"""
Central execution interceptor.

``RuntimeTracker.trace_step`` is the single entry-point called by every
AST-rewritten cell statement.  It orchestrates:

  1. Pre-snapshot of all input arguments
  2. Safe execution of the original function
  3. Post-snapshot of the output (and re-check of inputs for in-place mutation)
  4. Delta computation
  5. Parameter value building + branch-divergence detection
  6. Pydantic model assembly (AnalysisStep, AnalysisState, Artifact, ArtifactState,
     ParameterValue, Operation objects)
  7. Async hand-off to StorageBackend
  8. Transparent return of the unmodified output

Branching
---------
Two cases trigger the creation of a new AnalysisBranch:

  Case 1 – Re-run with different parameters (auto-detected):
      The analyst re-runs a cell (identified by ``func_name``) that has already
      been executed on the current branch, but passes different argument values.
      The divergence point is the input state of the *previous* execution of
      that same function on the current branch.

  Case 2 – Manual checkout (analyst-triggered):
      The analyst calls ``rt.checkout(state_id)`` to rewind to any previously
      computed state and continue from there with different operations.
"""

from __future__ import annotations

import ast
import getpass
import hashlib
import inspect
import json
import platform
import sys
import time
import uuid
import weakref
from typing import Any, Optional

from models import (
    Agent,
    AgentType,
    AnalysisBranch,
    AnalysisHistory,
    AnalysisState,
    AnalysisStep,
    Artifact,
    ArtifactState,
    ArtifactStateParameterValue,
    ArtifactType,
    DictParameterValue,
    LambdaParameterValue,
    ListParameterValue,
    Operation,
    OperationType,
    StepCategory,
    RuntimeEnvironment,
    ScalarParameterValue,
)
from tracker.ast_rewriter import _is_omitted
from tracker.delta_calculator import compute_delta
from tracker.logger import log_trace_warning
from tracker.operation_registry import lookup as _lookup_operation_type, lookup_category as _lookup_category
from tracker.snapshot_engine import capture_snapshot
from tracker.snapshot_policy import should_snapshot
from tracker.storage import StorageBackend


def _bound_name(raw_line: Optional[str]) -> Optional[str]:
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


def _bound_names(raw_line: Optional[str]) -> list[str]:
    """Every name a single assignment line binds or mutates.

    ``a = f()`` -> [a]; ``a, b = f()`` -> [a, b]; ``df['c'] = x`` -> [df]
    (a subscript/attribute store mutates the object its root name refers to).
    """
    if not raw_line:
        return []
    try:
        node = ast.parse(raw_line.strip()).body[0]
    except (SyntaxError, IndexError):
        return []
    if not isinstance(node, ast.Assign):
        return []
    names: list[str] = []
    for target in node.targets:
        elts = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        for elt in elts:
            while isinstance(elt, (ast.Subscript, ast.Attribute)):
                elt = elt.value
            if isinstance(elt, ast.Name) and elt.id not in names:
                names.append(elt.id)
    return names


_FRAME_TYPES = ("DataFrame", "Series")


def _lead_frame(func: Any, root: Any) -> Any:
    """The DataFrame a call operates on: its root name, bound receiver, or a GroupBy's frame."""
    for candidate in (root, getattr(func, "__self__", None)):
        if candidate is None:
            continue
        if type(candidate).__name__ == "DataFrame":
            return candidate
        inner = getattr(candidate, "obj", None)  # DataFrameGroupBy keeps its frame here
        if type(inner).__name__ == "DataFrame":
            return inner
    return None


def _is_storable(obj: Any) -> bool:
    """Objects worth pickling as an artifact: data and models, not code or handles."""
    import io
    import types

    return obj is not None and not isinstance(
        obj, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType,
              types.GeneratorType, type, io.IOBase)
    )


def _is_row_subset(parent: Any, child: Any) -> bool:
    """True when *child* is *parent* restricted to some of its rows, by label.

    A filter-index artifact stores only the child's index labels and rebuilds
    it as ``parent.loc[labels]``; that is exact only when labels are unique,
    all come from the parent and select identical rows (a ``reset_index`` or a
    value change would not).
    """
    try:
        if not (child.index.is_unique and parent.index.is_unique and child.index.isin(parent.index).all()):
            return False
        head = child.index[:3]
        return bool(parent.loc[head].equals(child.iloc[:3]))
    except Exception:
        return False


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """*value* with everything JSON cannot hold (types, objects) replaced by its text."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _depth < 4 and isinstance(value, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in value]
    if _depth < 4 and isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if type(value).__name__ in _FRAME_TYPES:
        return f"<{type(value).__name__} {'x'.join(str(n) for n in value.shape)}>"
    return getattr(value, "__qualname__", None) or str(value)[:150]


def _uid() -> str:
    return str(uuid.uuid4())


def _checksum_and_size(path: str) -> tuple[str, int]:
    """Return (sha256_hex, size_bytes) for the file at *path*."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _param_fingerprint(args: list, kwargs: dict) -> str:
    """
    Stable MD5 hash of the call's argument values.

    Used only for branch-divergence detection — two calls are considered
    'different' when their fingerprints differ.  String-truncation keeps
    the payload bounded for large objects (DataFrames, etc.).
    """
    payload = json.dumps(
        {
            "args": [str(a)[:200] for a in args],
            "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _capture_env() -> RuntimeEnvironment:
    """Collect Python version + installed analysis libraries."""
    libs: dict[str, str] = {}
    for lib in ("pandas", "polars", "numpy", "pm4py", "pydantic", "duckdb"):
        mod = sys.modules.get(lib)
        if mod is None:
            try:
                mod = __import__(lib)
            except Exception:
                # A library can fail for reasons other than not being
                # installed (e.g. pm4py's import chain probes psutil for
                # the parent process, which can be unresolvable in a
                # sandboxed/containerized session) — env capture is best
                # effort, so skip that one entry rather than aborting init.
                continue
        libs[lib] = getattr(mod, "__version__", "?")

    return RuntimeEnvironment(
        env_id=_uid(),
        tool_version=f"Python {sys.version.split()[0]}",
        library_versions=libs,
        runtime=f"{platform.system()} {platform.release()}",
    )


def _infer_artifact_type(obj: Any) -> ArtifactType:
    # Conceptual type defaults to OTHER; future: inspect XES columns to detect EVENT_LOG
    return ArtifactType.OTHER


def _is_row_filter(pre_snap: dict, post_snap: dict) -> bool:
    """True when a DataFrame step looks like a pure row filter: same columns+dtypes, fewer rows."""
    return (
        pre_snap.get("kind") == "dataframe"
        and post_snap.get("kind") == "dataframe"
        and pre_snap.get("columns") == post_snap.get("columns")
        and pre_snap.get("dtypes") == post_snap.get("dtypes")
        and post_snap.get("shape", [0])[0] < pre_snap.get("shape", [0])[0]
    )


class RuntimeTracker:
    """
    Stateful provenance tracker for one notebook session.

    Injected into the notebook namespace as ``_provtrack_runtime`` so that
    AST-rewritten cells can call ``_provtrack_runtime.trace_step(...)``.

    Internal registries
    -------------------
    _artifact_registry : dict[int, str]
        Maps Python object identity (id(obj)) → artifact_id.  Used to detect
        when a DataFrame is the same *conceptual* artifact across steps so we
        don't create a redundant Artifact record.

    _artifact_state_registry : dict[int, str]
        Maps Python object identity (id(obj)) → the most recent artifact_state_id
        for that object.  Used by _make_param_value so that DataFrames passed as
        arguments become ArtifactStateParameterValue references rather than opaque
        scalar summaries.

    _operation_cache : dict[str, tuple[Operation, OperationType]]
        Caches (Operation, OperationType) per unique func_name within a session.

    _cell_executions : dict[str, list[dict]]
        Maps func_name → list of prior execution records, each containing:
        ``{input_state_id, output_state_id, param_fingerprint, branch_id}``.
        Used for Case-1 branch-divergence detection.
    """

    def __init__(
        self,
        storage: StorageBackend,
        session_id: str,
        agent_id: Optional[str] = None,
        history_name: Optional[str] = None,
        branch_name: str = "main",
    ) -> None:
        self.storage = storage
        self.session_id = session_id
        self._env = _capture_env()
        _username = _try_getuser()
        self._agent = Agent(
            agent_id=agent_id or _username,
            agent_type=AgentType.HUMAN,
            username=_username,
        )

        # Artifact identity registries (scoped to this session)
        self._artifact_registry: dict[int, str] = {}
        self._artifact_state_registry: dict[int, str] = {}
        self._registered_refs: dict[int, Any] = {}  # id -> weakref, to tell a live object from a reused id
        self._block_pre: dict[str, dict] = {}  # raw_line -> pre-loop snapshots (see begin_block)
        # artifact_id -> latest artifact_state_id, across ALL tracked artifacts
        # (not keyed by Python object identity like the two above) -- lets
        # every new AnalysisState link the complete current artifact set, not
        # just whatever this one step happened to touch. See
        # _link_untouched_artifacts().
        self._artifact_id_to_latest_state: dict[str, str] = {}
        # variable name -> artifact_state_id as of the current state; persisted
        # per state so restore_state() never re-derives names or lineage.
        self._bindings: dict[str, str] = {}

        # Operation cache: func_name → (Operation, OperationType)
        self._operation_cache: dict[str, tuple[Operation, OperationType]] = {}

        # Execution history: func_name → list of execution records. No longer
        # used to auto-detect divergence (branching is manual-only, see
        # checkout() / _resolve_input_state()) — kept for last_call_params()
        # and any future auditing use.
        self._cell_executions: dict[str, list[dict]] = {}

        # Set by checkout(); consumed by the very next trace_step/
        # trace_ui_step call. See _resolve_input_state() for the manual-only
        # branching rule this drives.
        self._pending_checkout_state_id: Optional[str] = None
        self._pending_checkout_branch_name: Optional[str] = None

        # Abstraction registry: name → callable(df, state_id) -> Any
        self._abstraction_registry: dict[str, Any] = {}
        # Computed abstraction results: state_id → {name → value}
        self._abstraction_cache: dict[str, dict[str, Any]] = {}

        # Maps func_name → callable, populated by trace_step for replay support.
        self._replay_func_registry: dict[str, Any] = {}

        # Runtime observers: callback(step_detail: dict) fired synchronously
        # after every trace_step()/trace_ui_step() — see tracker/observers.py.
        self._step_observers: list[Any] = []

        # ---- Seed session-level provenance records -------------------------
        self._history = AnalysisHistory(
            history_id=_uid(),
            name=history_name,
        )
        root_state_id = _uid()
        self._branch = AnalysisBranch(
            branch_id=_uid(),
            history_id=self._history.history_id,
            name=branch_name,
            starts_at_state_id=root_state_id,
        )
        self._root_state_id: str = root_state_id
        self._current_state_id: str = root_state_id

        # Persist session-level records synchronously so trace_step never races them.
        self.storage._executor.submit(self.storage._write_history, self._history).result()
        self.storage._executor.submit(self.storage._write_branch, self._branch).result()
        root_state = AnalysisState(
            state_id=root_state_id,
            history_id=self._history.history_id,
            branch_id=self._branch.branch_id,
        )
        self.storage._executor.submit(self.storage._write_state, root_state).result()
        self.storage._executor.submit(self.storage._write_agent, self._agent, self._history.history_id).result()
        self.storage._executor.submit(self.storage._write_env, self._env, self._history.history_id).result()

    @classmethod
    def resume(
        cls,
        storage: StorageBackend,
        history_id: str,
        session_id: str,
        agent_id: Optional[str] = None,
    ) -> "RuntimeTracker":
        """
        Continue a previously recorded AnalysisHistory instead of creating a
        new one -- used when init_marimo()/init_jupyter() find a matching
        history from an earlier session (see kernel_hooks.py).

        Restores the current-state pointer (from the history's active_state_id,
        falling back to its root state if nothing was ever traced), the branch
        that state belongs to, and the per-function call history divergence
        detection needs to keep working exactly as if the process had never
        restarted. Agent/RuntimeEnvironment are captured fresh, same as a new
        session -- only the analysis lineage itself is continued.
        """
        history_row = storage.load_history(history_id)
        if history_row is None:
            raise ValueError(f"no such history: {history_id!r}")

        self = cls.__new__(cls)
        self.storage = storage
        self.session_id = session_id
        self._env = _capture_env()
        _username = _try_getuser()
        self._agent = Agent(
            agent_id=agent_id or _username,
            agent_type=AgentType.HUMAN,
            username=_username,
        )

        self._artifact_registry = {}
        self._artifact_state_registry = {}
        self._registered_refs = {}
        self._block_pre = {}
        self._artifact_id_to_latest_state = {}
        self._bindings = {}
        self._operation_cache = {}
        self._abstraction_registry = {}
        self._abstraction_cache = {}
        self._replay_func_registry = {}

        self._history = AnalysisHistory(
            history_id=history_row["history_id"],
            name=history_row["name"],
            active_state_id=history_row["active_state_id"],
        )

        self._root_state_id = storage.find_root_state_id(history_id)
        self._current_state_id = history_row["active_state_id"] or self._root_state_id
        if self._current_state_id is None:
            raise ValueError(f"history has no states, cannot resume: {history_id!r}")

        branch_id = storage.load_state_branch_id(self._current_state_id)
        branch_row = next(
            (b for b in storage.load_branches(history_id) if b["branch_id"] == branch_id),
            None,
        )
        if branch_row is None:
            raise ValueError(
                f"no branch found for state {self._current_state_id!r} in history {history_id!r}"
            )
        self._branch = AnalysisBranch(
            branch_id=branch_row["branch_id"],
            history_id=history_id,
            name=branch_row["name"],
            starts_at_state_id=branch_row["starts_at_state_id"],
        )

        self._cell_executions = storage.load_cell_executions(history_id)
        self._bindings = storage.load_state_bindings(self._current_state_id)
        self._pending_checkout_state_id = None
        self._pending_checkout_branch_name = None
        self._step_observers = []

        self.storage._executor.submit(
            self.storage._write_agent, self._agent, self._history.history_id
        ).result()
        self.storage._executor.submit(
            self.storage._write_env, self._env, self._history.history_id
        ).result()

        return self

    def last_call_params(self, func_name: str) -> Optional[dict]:
        """Decoded params of the most recent call to func_name on the
        current branch, or None if func_name was never called here.

        Lets a resumed session recover what it was last configured with
        (e.g. a marimo notebook re-seeding its UI defaults on kernel
        restart) without needing to know pmprov's internal ParameterValue
        shape. Params with value_type "artifact_state_ref" are excluded
        from the result since artifact references are transient, not
        configuration values.
        """
        raw = self.storage.load_last_step_params(
            self._history.history_id, self._current_state_id, func_name
        )
        if raw is None:
            return None
        # local import: tracker.visualizations imports RuntimeTracker at module scope
        from tracker.visualizations import unwrap_param_value
        result = {}
        for pv in raw:
            name = pv["param_id"].split(":", 1)[-1]
            if name == "__receiver__":
                continue
            # Skip artifact-state-ref params as they are transient references,
            # not configuration values. Check the raw value_type before unwrapping.
            if pv.get("value_type") == "artifact_state_ref":
                continue
            value = unwrap_param_value(pv["value"])
            result[name] = value
        return result

    # ------------------------------------------------------------------
    # Public API – invoked by AST-rewritten cells
    # ------------------------------------------------------------------

    def trace_step(
        self,
        *,
        func,
        func_name: str,
        raw_line: str,
        args: list,
        kwargs: dict,
        targets: Optional[list[str]] = None,
        root: Any = None,
        named_outputs: bool = False,
        root_name: Optional[str] = None,
    ) -> Any:
        """
        Execute ``func(*args, **kwargs)``, capture provenance, and return the
        unmodified result.  Telemetry failures are isolated so they never
        prevent user code from completing.

        targets:
            Names the statement binds or mutates (the rewriter passes them;
            parsed from *raw_line* when absent). Several names split a
            tuple result into one artifact each.
        root:
            The object the call chain starts from (``df`` in
            ``df.groupby(k).agg(...)``), so a method call's delta is measured
            on the frame it transforms.
        root_name:
            Name of *root* in the notebook; a bare method call that returns
            the root itself, or nothing at all, has changed that variable.
        named_outputs:
            *func* returned ``{name: object}`` (a loop block): every entry is
            an output of the one step.
        """
        # ---- 0. Runtime omit check ------------------------------------------
        # The AST rewriter filters omitted calls at compile time, but a stale
        # transformer instance (registered before an omit-list update or a
        # code reload) may still route calls here.  This gate ensures omitted
        # functions are NEVER tracked regardless of transformer state.
        names = list(targets) if targets is not None else _bound_names(raw_line)
        if _is_omitted(func_name, bound=bool(names)):
            return func(*args, **kwargs)
        # Subscript/attribute helpers only mean something on frames; on a
        # dict, list or array the statement runs as plain Python.
        if getattr(func, "_provtrack_frames_only", False) and not (
            args and type(args[0]).__name__ in _FRAME_TYPES
        ):
            return func(*args, **kwargs)

        # ---- 1. Pre-snapshots ------------------------------------------------
        lead = _lead_frame(func, root)
        lead_pre = capture_snapshot(lead) if lead is not None else None
        block_pre = self._block_pre.pop(raw_line, None) if named_outputs else None
        pre_snaps: dict[str, dict] = {}
        for i, a in enumerate(args):
            pre_snaps[f"arg_{i}"] = capture_snapshot(a)
        for k, v in kwargs.items():
            pre_snaps[f"kwarg_{k}"] = capture_snapshot(v)

        # ---- 2. Execute (user exceptions always propagate) -------------------
        _call_start = time.perf_counter()
        output = func(*args, **kwargs)
        _elapsed_seconds = time.perf_counter() - _call_start
        self._replay_func_registry[func_name] = func
        if output is None and kwargs.get("inplace") is True:
            # df.sort_values(..., inplace=True) returns None; the result is the receiver.
            receiver = getattr(func, "__self__", None)
            output = receiver if type(receiver).__name__ in _FRAME_TYPES else lead

        if (
            not names and root_name and root is not None and type(root).__name__ not in _FRAME_TYPES
            and (output is root or (output is None and root_name in self._bindings))
            and _is_storable(root)
        ):
            # estimator.fit(...) / fig.update_layout(...): the method changed
            # the object a variable refers to, so that variable has a new value.
            names = [root_name]
            output = root

        # ---- 3. Post-snapshots -----------------------------------------------
        try:
            post_snap: dict = capture_snapshot(output) if output is not None else {}
            all_input_vals = list(args) + list(kwargs.values())
            post_input_snaps = {
                key: capture_snapshot(val)
                for key, val in zip(list(pre_snaps.keys()), all_input_vals)
            }
        except Exception as e:
            log_trace_warning("post-snapshot failed", component="snapshot", func_name=func_name, error=e)
            post_snap = {}
            post_input_snaps = {}

        # ---- 4. Delta --------------------------------------------------------
        delta: dict = {}
        try:
            first_pre = None if getattr(func, "_provtrack_no_delta", False) else lead_pre or next(
                (v for v in pre_snaps.values() if v.get("kind") not in ("other", None)),
                None,
            )
            if named_outputs and isinstance(output, dict) and output:
                first_name, first_obj = next(iter(output.items()))
                if block_pre and first_name in block_pre:
                    delta = compute_delta(block_pre[first_name], capture_snapshot(first_obj))
            elif first_pre and post_snap:
                delta = compute_delta(first_pre, post_snap)
            elif first_pre and pre_snaps:
                first_key = next(iter(pre_snaps))
                delta = compute_delta(first_pre, post_input_snaps.get(first_key, {}))
        except Exception as e:
            log_trace_warning("delta computation failed", component="delta", func_name=func_name, error=e)
            delta = {"kind": "error"}

        # ---- 5. Parameter values + branch-divergence detection ---------------
        # Parameters must be built before divergence detection so we can
        # compute the fingerprint from the actual argument values.

        step_id = _uid()
        output_state_id = _uid()

        # Ensure DataFrame args are persisted so they can be referenced as
        # ArtifactStateParameterValue (enabling replay_state to reconstruct them).
        try:
            for a in list(args) + list(kwargs.values()):
                if type(a).__name__ == "DataFrame" and self._lookup_artifact_state(a) is None:
                    _input_state_id = _uid()
                    _input_path = self.storage.save_artifact(_input_state_id, a)
                    if _input_path:
                        try:
                            _input_recs = self._build_artifact_records(a, _input_state_id, _input_path)
                            _input_art_obj = _input_recs.get("artifact_obj")
                            _input_ast_obj = _input_recs.get("artifact_state_obj")
                            if _input_ast_obj:
                                self.storage.save_artifact_records_async(
                                    _input_art_obj, _input_ast_obj, self._history.history_id
                                )
                        except Exception as e:
                            log_trace_warning("input artifact record build failed",
                                              step="input_artifact", func_name=func_name, error=e)
        except Exception as e:
            log_trace_warning("input DataFrame registration failed",
                              step="input_artifact_loop", func_name=func_name, error=e)

        param_values: list[dict] = []
        try:
            # Try to get parameter names from function signature
            param_names: list[str] = []
            try:
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
            except (ValueError, TypeError):
                # If inspection fails, fall back to positional names for args
                param_names = [f"arg_{i}" for i in range(len(args))]

            def _add_param(param_id: str, value: Any) -> None:
                # One unserialisable parameter must not drop the step's others.
                try:
                    param_values.append(self._make_param_value(param_id, step_id, value).model_dump(mode="json"))
                except Exception as e:
                    log_trace_warning("parameter value serialisation failed",
                                      step="param_values", func_name=func_name, param=param_id, error=e)

            for i, a in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else f"arg_{i}"
                _add_param(f"{func_name}:{param_name}", a)
            for k, v in kwargs.items():
                _add_param(f"{func_name}:{k}", v)
            # For bound methods (e.g. case_log.apply(...)), record the receiver object
            # so replay can reconstruct the method call without losing the DataFrame target.
            if hasattr(func, "__self__") and func.__self__ is not None:
                receiver = func.__self__
                if type(receiver).__name__ == "DataFrame" and self._lookup_artifact_state(receiver) is None:
                    _recv_state_id = _uid()
                    _recv_path = self.storage.save_artifact(_recv_state_id, receiver)
                    if _recv_path:
                        try:
                            _recv_recs = self._build_artifact_records(receiver, _recv_state_id, _recv_path)
                            _recv_art = _recv_recs.get("artifact_obj")
                            _recv_ast = _recv_recs.get("artifact_state_obj")
                            if _recv_ast:
                                self.storage.save_artifact_records_async(
                                    _recv_art, _recv_ast, self._history.history_id
                                )
                        except Exception as e:
                            log_trace_warning("receiver artifact record build failed",
                                              step="receiver_artifact", func_name=func_name, error=e)
                _add_param(f"{func_name}:__receiver__", receiver)
        except Exception as e:
            log_trace_warning("parameter value serialisation failed",
                              step="param_values", func_name=func_name, error=e)

        try:
            fp = _param_fingerprint(args, kwargs)
        except Exception as e:
            log_trace_warning("parameter fingerprinting failed, using random UUID",
                              step="fingerprint", func_name=func_name, error=e)
            fp = str(uuid.uuid4())
        try:
            input_state_id = self._resolve_input_state(func_name, fp)
        except Exception as e:
            log_trace_warning("post-checkout branch resolution failed, staying on current branch",
                              step="branch_detection", func_name=func_name, error=e)
            input_state_id = self._current_state_id

        # ---- 6. Pydantic model assembly -------------------------------------
        operation, op_type, step_cat = self._get_or_create_operation(func_name)

        step = AnalysisStep(
            step_id=step_id,
            input_state_id=input_state_id,
            output_state_id=output_state_id,
            agent_id=self._agent.agent_id,
            env_id=self._env.env_id,
            operation_id=operation.operation_id,
        )

        # ---- 7. Artifact persistence -----------------------------------------
        # One artifact per output: a tuple result bound to several names, or a
        # loop block that changed several frames, stores each separately.
        outputs = self._split_outputs(output, names, named_outputs)
        persisted: list[tuple[Optional[str], dict]] = []  # (name, artifact records)
        if should_snapshot(func_name, op_type.name, elapsed_seconds=_elapsed_seconds):
            for index, (out_name, out_obj) in enumerate(outputs):
                records = self._persist_output(
                    out_obj,
                    node_id=output_state_id if index == 0 else _uid(),
                    input_state_id=input_state_id,
                    parent_pre=lead_pre or pre_snaps.get("arg_0", {}),
                    parent_frame=next((a for a in args if type(a).__name__ == "DataFrame"), None),
                    func_name=func_name,
                )
                if records:
                    persisted.append((out_name, records))

        # ---- 8. Persist all entities ------------------------------------------------
        output_state = AnalysisState(
            state_id=output_state_id,
            history_id=self._history.history_id,
            branch_id=self._branch.branch_id,
            produced_by_step_id=step_id,
            derived_from_state_id=input_state_id,
        )
        self.storage.save_state_async(output_state)
        self.storage.save_step_async(step, func_name, raw_line, fp, self._history.history_id)
        self.storage.save_operation_async(operation, op_type, step_cat)
        if param_values:
            self.storage.save_param_values_async(param_values)
        if delta:
            self.storage.save_delta_async(delta, step_id)
        touched: set[str] = set()
        persisted_names: set[str] = set()
        for out_name, records in persisted:
            artifact_state_obj = records["artifact_state_obj"]
            self.storage.save_artifact_records_async(
                records.get("artifact_obj"), artifact_state_obj, self._history.history_id
            )
            touched.add(artifact_state_obj.artifact_id)
            label = out_name or f"{records['label']}_{artifact_state_obj.artifact_id[:8]}"
            self._bindings[label] = artifact_state_obj.artifact_state_id
            persisted_names.add(label)
        # A name rebound to something that was not stored must not keep
        # pointing at its previous value.
        for out_name, _ in outputs:
            if out_name and out_name not in persisted_names:
                self._bindings.pop(out_name, None)
        # Every state is a complete snapshot of the artifact set: link every
        # OTHER already-known artifact forward to this state too, aliased
        # rather than re-persisted (see _link_untouched_artifacts).
        self._link_untouched_artifacts(output_state_id, touched)
        self.storage.save_state_bindings_async(output_state_id, self._bindings)

        # Record this execution for future divergence checks on this func.
        self._cell_executions.setdefault(func_name, []).append({
            "input_state_id": input_state_id,
            "output_state_id": output_state_id,
            "param_fingerprint": fp,
            "branch_id": self._branch.branch_id,
        })

        self._current_state_id = output_state_id
        self._history.active_state_id = output_state_id
        self.storage.update_history_active_state_async(self._history.history_id, output_state_id)

        self._notify_step_observers(
            state_id=output_state_id, step_id=step_id, func_name=func_name,
            raw_line=raw_line, branch_name=self._branch.name,
            params=param_values, delta=delta,
        )

        return output

    def begin_block(self, raw_line: str, names: list[str], namespace: dict) -> None:
        """Remember the metadata of the frames a loop is about to change, for its delta."""
        pre = {}
        for name in names:
            value = namespace.get(name)
            if type(value).__name__ in _FRAME_TYPES:
                pre[name] = capture_snapshot(value)
        self._block_pre[raw_line] = pre

    def checkout(self, state_id: str, branch_name: Optional[str] = None) -> None:
        """
        Manually rewind to a previous analysis state, WITHOUT creating a
        branch yet.

        Branching in this build is manual-only: nothing forks automatically
        just because arguments differ from an earlier run (contrast the
        auto-divergence behaviour on the `feat/ui-interactions` branch).
        checkout() only moves "where the analyst currently is" back to
        state_id, staying on whichever branch state_id already belongs to.

        A branch is created lazily, only once a step actually diverges from
        something that was already executed from state_id (see
        _resolve_input_state()). This isn't limited to the single next step —
        as long as each step exactly repeats one that already happened from
        the current replay position, it rejoins and pmprov keeps checking
        the step after that too:

        - If nothing has ever run from state_id, the very next step forks
          immediately, starting at state_id.
        - If a step is an identical repeat of one that already ran from the
          current replay position (same func_name + param_fingerprint), it
          rejoins that existing branch and replay continues into the step
          after it — e.g. re-running a whole pipeline where only the third
          call actually has different arguments forks exactly there, with
          the first two calls rejoining rather than forking pointlessly.
        - The first step that doesn't match anything forks, starting at
          wherever the replay position had advanced to by then. That fork
          ends the replay — everything after it just appends normally.

        Parameters
        ----------
        state_id:
            The ``AnalysisState.state_id`` to resume from. Must already
            belong to a persisted branch.
        branch_name:
            Optional label to use IF a branch ends up being created by the
            next step. Ignored if that step doesn't diverge from anything.
        """
        branch_id = self.storage.load_state_branch_id(state_id)
        if branch_id is None:
            raise ValueError(f"checkout: unknown state_id {state_id!r}")
        branch_row = self.storage.load_branch(branch_id)
        self._branch = AnalysisBranch(
            branch_id=branch_row["branch_id"],
            history_id=branch_row["history_id"],
            name=branch_row["name"],
            starts_at_state_id=branch_row["starts_at_state_id"],
        )
        self._current_state_id = state_id
        self._bindings = self.storage.load_state_bindings(state_id)
        self._pending_checkout_state_id = state_id
        self._pending_checkout_branch_name = branch_name
        # active_state_id reflects where the analyst is now — the checkout point.
        self._history.active_state_id = state_id
        self.storage.update_history_active_state_async(self._history.history_id, state_id)

    def create_pipeline(self, name: str, step_ids: list[str]) -> str:
        """Record a named Pipeline covering the given step_ids in order. Returns pipeline_id."""
        pipeline_id = _uid()
        self.storage.save_pipeline_sync(pipeline_id, self._history.history_id, name)
        fragment_id = _uid()
        self.storage.save_fragment_sync(fragment_id, pipeline_id, step_ids, position=0)
        return pipeline_id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_input_state(self, func_name: str, fp: str) -> str:
        """
        Manual-only branching: never fork on its own during ordinary
        execution. Forking is only ever considered while "replaying" after a
        checkout() — see checkout()'s docstring.

        This replay isn't limited to a single step: as long as each step
        exactly repeats one that already ran from the current replay
        position (same func_name + fingerprint), it rejoins that existing
        lineage and the replay position advances to keep checking the *next*
        step too — e.g. re-running a whole multi-step pipeline after a
        restart, where only the third call actually uses different
        arguments, forks exactly at that third step, with the first two
        rejoining their original branch rather than forking pointlessly.
        The very first step that does NOT match anything (whether because
        nothing exists there yet, or because what exists differs) forks —
        and that fork ends the replay: every step after it just appends
        normally, like any ordinary execution.

        Returns
        -------
        str
            The ``input_state_id`` for the step about to be recorded.
        """
        if self._pending_checkout_state_id is None:
            return self._current_state_id

        checkout_state_id = self._pending_checkout_state_id
        branch_name = self._pending_checkout_branch_name

        children = self.storage.load_child_steps(checkout_state_id)
        # NOTE: "deep comparison" here is the same func_name + param_fingerprint
        # (truncated-string MD5) used elsewhere in this module, not a true
        # structural equality of the live argument objects. See the
        # conversation / PR description for why a full deep-equality
        # comparator was out of scope for this pass.
        match = next(
            (c for c in children if c["func_name"] == func_name and c["param_fingerprint"] == fp),
            None,
        )
        if match is not None:
            # Identical step already exists here — rejoin that lineage and
            # keep replaying: a later step might still diverge.
            branch_row = self.storage.load_branch(match["branch_id"])
            self._branch = AnalysisBranch(
                branch_id=branch_row["branch_id"],
                history_id=branch_row["history_id"],
                name=branch_row["name"],
                starts_at_state_id=branch_row["starts_at_state_id"],
            )
            self._pending_checkout_state_id = match["output_state_id"]
            # branch_name carried forward unchanged, in case a later step in
            # this same replay is what actually ends up diverging.
            return checkout_state_id

        # No match — the actual point of divergence. Fork here and stop
        # replaying; nothing after this is checked again.
        self._pending_checkout_state_id = None
        self._pending_checkout_branch_name = None

        # Genuine divergence: this is the one case that creates a branch.
        name = branch_name or f"branch-{func_name[:24]}-{checkout_state_id[:8]}"
        new_branch = self._create_branch(starts_at=checkout_state_id, name=name)
        return new_branch.starts_at_state_id

    def _create_branch(self, starts_at: str, name: str) -> AnalysisBranch:
        """
        Create a new AnalysisBranch starting at *starts_at*, persist it, and
        make it the active branch.
        """
        new_branch = AnalysisBranch(
            branch_id=_uid(),
            history_id=self._history.history_id,
            name=name,
            starts_at_state_id=starts_at,
        )
        self._branch = new_branch
        self._current_state_id = starts_at
        self.storage.save_branch_async(new_branch)
        return new_branch

    def _get_or_create_operation(
        self, func_name: str
    ) -> tuple[Operation, OperationType, Optional[StepCategory]]:
        """
        Return the (Operation, OperationType) pair for *func_name*, creating
        it on first encounter and caching it for the rest of the session.

        Operation represents the *definition* of what was called; OperationType
        is the semantic category resolved via the operation_registry (defaults
        to ``"unknown"`` when not registered). Operations are global reference
        data (no history_id), so before minting a new one this also checks
        storage for a row another session already created for the same
        func_name -- otherwise every session would mint its own duplicate
        Operation row for the same function.
        """
        if func_name in self._operation_cache:
            op, ot, _ = self._operation_cache[func_name]
            return op, ot, None  # step_cat already written on first encounter

        existing = self.storage.find_operation_by_name(func_name)
        if existing is not None:
            operation = Operation(
                operation_id=existing["operation_id"],
                name=func_name,
                operation_type_id=existing["operation_type_id"],
                step_category_id=existing["step_category_id"],
            )
            op_type = OperationType(
                type_id=existing["operation_type_id"],
                name=existing["operation_type_name"],
            )
            self._operation_cache[func_name] = (operation, op_type, None)
            return operation, op_type, None

        type_name = _lookup_operation_type(func_name)
        op_type = OperationType(type_id=_uid(), name=type_name)

        # Check for a registered StepCategory for this OperationType
        category_name = _lookup_category(type_name)
        step_cat: Optional[StepCategory] = None
        step_cat_id: Optional[str] = None
        if category_name:
            step_cat = StepCategory(category_id=_uid(), name=category_name)
            step_cat_id = step_cat.category_id

        operation = Operation(
            operation_id=_uid(),
            name=func_name,
            operation_type_id=op_type.type_id,
            step_category_id=step_cat_id,
        )
        self._operation_cache[func_name] = (operation, op_type, step_cat)
        return operation, op_type, step_cat

    def _lookup_artifact_state(self, obj: Any) -> Optional[str]:
        """artifact_state_id last recorded for *obj* itself (not for a dead object that had its id)."""
        ast_id = self._artifact_state_registry.get(id(obj))
        if ast_id is None:
            return None
        ref = self._registered_refs.get(id(obj))
        return ast_id if ref is not None and ref() is obj else None

    def _split_outputs(self, output: Any, names: list[str], named: bool) -> list[tuple[Optional[str], Any]]:
        """Pair a step's result with the names it was bound to."""
        if named:  # a loop block: {name: object}
            return [(n, o) for n, o in output.items()] if isinstance(output, dict) else []
        if len(names) > 1 and isinstance(output, (tuple, list)) and len(output) == len(names):
            return list(zip(names, output))
        return [(names[0] if len(names) == 1 else None, output)]

    def _persist_output(
        self, obj: Any, *, node_id: str, input_state_id: str, parent_pre: dict,
        parent_frame: Any, func_name: str,
    ) -> Optional[dict]:
        """Store *obj* as an artifact; return its records (plus a ``label``) or None."""
        if obj is None:
            return None
        try:
            kind = capture_snapshot(obj).get("kind")
            parent_ast_id = ""
            if kind == "dataframe":
                post = capture_snapshot(obj)
                if _is_row_filter(parent_pre, post) and parent_frame is not None and _is_row_subset(parent_frame, obj):
                    parent_ast_id = self._lookup_artifact_state(parent_frame) or ""
                    if parent_ast_id:
                        kind = "filter_index"
                label = "dataframe"
            elif kind == "figure":
                label = "dataframe"
            elif _is_storable(obj):
                kind, label = "pickle", "object"
            else:
                return None
            path = self.storage.save_artifact(
                node_id, obj, kind=kind,
                parent_state_id=input_state_id,
                parent_artifact_state_id=parent_ast_id,
            )
            if not path:
                return None
            records = self._build_artifact_records(obj, node_id, path, label=label)
            records["label"] = label
            return records if records.get("artifact_state_obj") else None
        except Exception as e:
            log_trace_warning("output artifact persistence failed",
                              step="output_artifact", func_name=func_name, error=e)
            return None

    def _build_artifact_records(
        self,
        obj: Any,
        output_state_id: str,
        artifact_path: str,
        label: str = "dataframe",
    ) -> dict:
        """
        Build Artifact + ArtifactState Pydantic models for an output.

        Returns a dict with keys ``artifact_obj`` (None if already seen) and
        ``artifact_state_obj``.
        Side effects: updates _artifact_registry and _artifact_state_registry
        for frames and figures (mutable objects that keep an identity across
        steps); other outputs are always a new artifact.
        """
        obj_python_id = id(obj)
        checksum_hex, size_bytes = _checksum_and_size(artifact_path)
        tracked = type(obj).__name__ in ("DataFrame", "Figure")

        artifact: Optional[Artifact] = None
        artifact_id = None
        if tracked:
            ref = self._registered_refs.get(obj_python_id)
            if ref is not None and ref() is obj:  # not a dead object whose id was reused
                artifact_id = self._artifact_registry.get(obj_python_id)
        if artifact_id is None:
            artifact_id = _uid()
            artifact = Artifact(
                artifact_id=artifact_id,
                name=f"{label}_{artifact_id[:8]}",
                artifact_type=_infer_artifact_type(obj),
            )
            if tracked:
                self._artifact_registry[obj_python_id] = artifact_id

        if artifact_path.endswith(".plotly.json"):
            mime_type = "application/vnd.plotly.v1+json"
        elif artifact_path.endswith(".filter.json"):
            mime_type = "application/x-pmprov-filter-index+json"
        elif artifact_path.endswith(".pkl"):
            mime_type = "application/x-pmprov-pickle"
        elif artifact_path.endswith(".ui_alias.json"):
            mime_type = "application/x-pmprov-ui-alias+json"
        else:
            mime_type = "application/vnd.apache.parquet"

        artifact_state = ArtifactState(
            artifact_state_id=_uid(),
            artifact_id=artifact_id,
            analysis_state_id=output_state_id,
            mime_type=mime_type,
            checksum=f"sha256:{checksum_hex}",
            content_ref=artifact_path,
            size_bytes=size_bytes,
        )
        if tracked:
            self._artifact_state_registry[obj_python_id] = artifact_state.artifact_state_id
            try:
                self._registered_refs[obj_python_id] = weakref.ref(obj)
            except TypeError:
                self._registered_refs.pop(obj_python_id, None)
        self._artifact_id_to_latest_state[artifact_id] = artifact_state.artifact_state_id
        return {"artifact_obj": artifact, "artifact_state_obj": artifact_state}

    def _link_untouched_artifacts(
        self, output_state_id: str, touched_artifact_id: Any = None,
    ) -> None:
        """
        Every AnalysisState is meant to be a complete snapshot of the current
        artifact set, not just whatever this one step happened to touch. For
        every artifact this step did NOT modify, link it forward to
        output_state_id via a cheap alias row (a tiny pointer file, same
        mechanism as UI-interaction states) instead of re-serializing it --
        a histogram/data_explorer render, or any step that isn't a data
        transform, should never trigger a duplicate Parquet write of
        artifacts it never touched.

        Known cost trade-off: this writes one small alias row per untouched
        artifact per step, so a session that accumulates many distinct
        snapshot-worthy artifacts (most pandas calls return a new object, so
        this grows with the number of DISTINCT tracked DataFrames/figures,
        not the number of steps) will see that cost grow accordingly. Still
        far cheaper than re-persisting full Parquet content, but not O(1).
        """
        for artifact_id, latest_state_id in list(self._artifact_id_to_latest_state.items()):
            if artifact_id == touched_artifact_id or (
                isinstance(touched_artifact_id, (set, frozenset)) and artifact_id in touched_artifact_id
            ):
                continue
            try:
                alias_path = self.storage.save_artifact(
                    output_state_id, None, kind="artifact_alias",
                    parent_artifact_state_id=latest_state_id,
                )
                if not alias_path:
                    continue
                checksum_hex, size_bytes = _checksum_and_size(alias_path)
                alias_state = ArtifactState(
                    artifact_state_id=_uid(),
                    artifact_id=artifact_id,
                    analysis_state_id=output_state_id,
                    mime_type="application/x-pmprov-ui-alias+json",
                    checksum=f"sha256:{checksum_hex}",
                    content_ref=alias_path,
                    size_bytes=size_bytes,
                )
                self.storage.save_artifact_records_async(None, alias_state, self._history.history_id)
                self._artifact_id_to_latest_state[artifact_id] = alias_state.artifact_state_id
            except Exception as e:
                log_trace_warning("untouched-artifact link failed",
                                  step="link_untouched_artifacts", artifact_id=artifact_id, error=e)

    def _make_param_value(self, param_id: str, step_id: str, value: Any):
        """Map a runtime Python value to the appropriate ParameterValue subclass."""
        if type(value).__name__ == "DataFrame":
            ast_id = self._lookup_artifact_state(value)
            if ast_id:
                return ArtifactStateParameterValue(
                    parameter_value_id=_uid(),
                    parameter_id=param_id,
                    step_id=step_id,
                    artifact_state_id=ast_id,
                )

        if isinstance(value, (int, float, str, bool)) or value is None:
            return ScalarParameterValue(
                parameter_value_id=_uid(),
                parameter_id=param_id,
                step_id=step_id,
                value=value,
            )
        if callable(value):
            return LambdaParameterValue(
                parameter_value_id=_uid(),
                parameter_id=param_id,
                step_id=step_id,
                source_code=getattr(value, "__qualname__", None) or str(value)[:150],
                function_name=getattr(value, "__qualname__", None),
            )
        if isinstance(value, list):
            return ListParameterValue(
                parameter_value_id=_uid(),
                parameter_id=param_id,
                step_id=step_id,
                value=_jsonable(value[:50]),
            )
        if isinstance(value, dict):
            return DictParameterValue(
                parameter_value_id=_uid(),
                parameter_id=param_id,
                step_id=step_id,
                value={str(k): _jsonable(v) for k, v in list(value.items())[:50]},
            )
        return ScalarParameterValue(
            parameter_value_id=_uid(),
            parameter_id=param_id,
            step_id=step_id,
            value=str(value)[:150],
        )


def _try_getuser() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
