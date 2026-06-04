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

import getpass
import hashlib
import json
import platform
import sys
import uuid
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
from tracker.operation_registry import lookup as _lookup_operation_type, lookup_category as _lookup_category
from tracker.snapshot_engine import capture_snapshot
from tracker.storage import StorageBackend


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
            except ImportError:
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
        self._agent = Agent(
            agent_id=agent_id or _try_getuser(),
            agent_type=AgentType.HUMAN,
        )

        # Artifact identity registries (scoped to this session)
        self._artifact_registry: dict[int, str] = {}
        self._artifact_state_registry: dict[int, str] = {}

        # Operation cache: func_name → (Operation, OperationType)
        self._operation_cache: dict[str, tuple[Operation, OperationType]] = {}

        # Branch-divergence detection: func_name → list of execution records
        self._cell_executions: dict[str, list[dict]] = {}

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
    ) -> Any:
        """
        Execute ``func(*args, **kwargs)``, capture provenance, and return the
        unmodified result.  Telemetry failures are isolated so they never
        prevent user code from completing.
        """
        # ---- 0. Runtime omit check ------------------------------------------
        # The AST rewriter filters omitted calls at compile time, but a stale
        # transformer instance (registered before an omit-list update or a
        # code reload) may still route calls here.  This gate ensures omitted
        # functions are NEVER tracked regardless of transformer state.
        if _is_omitted(func_name):
            return func(*args, **kwargs)

        # ---- 1. Pre-snapshots ------------------------------------------------
        pre_snaps: dict[str, dict] = {}
        for i, a in enumerate(args):
            pre_snaps[f"arg_{i}"] = capture_snapshot(a)
        for k, v in kwargs.items():
            pre_snaps[f"kwarg_{k}"] = capture_snapshot(v)

        # ---- 2. Execute (user exceptions always propagate) -------------------
        output = func(*args, **kwargs)

        # ---- 3. Post-snapshots -----------------------------------------------
        try:
            post_snap: dict = capture_snapshot(output) if output is not None else {}
            all_input_vals = list(args) + list(kwargs.values())
            post_input_snaps = {
                key: capture_snapshot(val)
                for key, val in zip(list(pre_snaps.keys()), all_input_vals)
            }
        except Exception:
            post_snap = {}
            post_input_snaps = {}

        # ---- 4. Delta --------------------------------------------------------
        delta: dict = {}
        try:
            first_pre = next(
                (v for v in pre_snaps.values() if v.get("kind") not in ("other", None)),
                None,
            )
            if first_pre and post_snap:
                delta = compute_delta(first_pre, post_snap)
            elif first_pre:
                first_key = next(iter(pre_snaps))
                delta = compute_delta(first_pre, post_input_snaps.get(first_key, {}))
        except Exception:
            delta = {"kind": "error"}

        # ---- 5. Parameter values + branch-divergence detection ---------------
        # Parameters must be built before divergence detection so we can
        # compute the fingerprint from the actual argument values.

        step_id = _uid()
        output_state_id = _uid()

        param_values: list[dict] = []
        try:
            for i, a in enumerate(args):
                pv = self._make_param_value(f"{func_name}:arg_{i}", step_id, a)
                param_values.append(pv.model_dump(mode="json"))
            for k, v in kwargs.items():
                pv = self._make_param_value(f"{func_name}:{k}", step_id, v)
                param_values.append(pv.model_dump(mode="json"))
        except Exception:
            pass

        try:
            fp = _param_fingerprint(args, kwargs)
        except Exception:
            fp = str(uuid.uuid4())
        try:
            input_state_id = self._detect_and_apply_branch(func_name, fp)
        except Exception:
            input_state_id = self._current_state_id

        # ---- 6. Pydantic model assembly -------------------------------------
        operation, op_type = self._get_or_create_operation(func_name)

        step = AnalysisStep(
            step_id=step_id,
            input_state_id=input_state_id,
            output_state_id=output_state_id,
            agent_id=self._agent.agent_id,
            env_id=self._env.env_id,
            operation_id=operation.operation_id,
        )

        # ---- 7. Artifact persistence -----------------------------------------
        artifact_records: dict = {}
        artifact_path: Optional[str] = None
        if post_snap.get("kind") == "dataframe" and output is not None:
            artifact_path = self.storage.save_artifact(output_state_id, output)
            if artifact_path:
                try:
                    artifact_records = self._build_artifact_records(
                        output, output_state_id, artifact_path
                    )
                except Exception:
                    pass

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
        self.storage.save_operation_async(operation, op_type)
        if param_values:
            self.storage.save_param_values_async(param_values)
        if delta:
            self.storage.save_delta_async(delta, step_id)
        if artifact_path:
            artifact_state_obj = artifact_records.get("artifact_state_obj")
            if artifact_state_obj:
                self.storage.save_artifact_records_async(
                    artifact_records.get("artifact_obj"), artifact_state_obj, self._history.history_id
                )

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

        return output

    def checkout(self, state_id: str, branch_name: Optional[str] = None) -> AnalysisBranch:
        """
        Case-2 branching: manually rewind to a previous analysis state and
        start a new branch from it.

        Call this when you want to explore a different analysis path from a
        state that was computed earlier in the session — for example, to try
        a different algorithm after an earlier pre-processing step.

        Parameters
        ----------
        state_id:
            The ``AnalysisState.state_id`` to resume from.  Must correspond to
            a node already persisted in the provenance DB.
        branch_name:
            Optional label for the new branch.  Defaults to
            ``"branch-<first-8-chars-of-state_id>"``.

        Returns
        -------
        AnalysisBranch
            The newly created branch, already persisted to storage.
        """
        name = branch_name or f"branch-{state_id[:8]}"
        branch = self._create_branch(starts_at=state_id, name=name)
        # active_state_id reflects where the analyst is now — the checkout point.
        self._history.active_state_id = state_id
        self.storage.update_history_active_state_async(self._history.history_id, state_id)
        return branch

    def create_pipeline(self, name: str, step_ids: list[str]) -> str:
        """Record a named Pipeline covering the given step_ids in order. Returns pipeline_id."""
        pipeline_id = _uid()
        self.storage.save_pipeline_async(pipeline_id, self._history.history_id, name)
        fragment_id = _uid()
        self.storage.save_fragment_async(fragment_id, pipeline_id, step_ids, position=0)
        return pipeline_id

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_and_apply_branch(self, func_name: str, fp: str) -> str:
        """
        Case-1 auto-branching: detect whether the current call diverges from
        a prior execution of the same function on the current branch.

        A divergence is detected when ``func_name`` has been executed on
        ``self._branch`` before AND the previous execution used different
        argument values (different ``param_fingerprint``).

        When a divergence is found:
        - A new AnalysisBranch is created, starting at the *input state* of
          the previous execution.  That input state is the true fork point in
          the analysis DAG.
        - ``self._branch`` and ``self._current_state_id`` are updated so that
          all subsequent steps are recorded on the new branch.

        Returns
        -------
        str
            The ``input_state_id`` for the step about to be recorded.  This is
            either ``self._current_state_id`` (no branch) or the prior
            execution's input state (branch created).
        """
        prior = self._cell_executions.get(func_name, [])

        # Find the most recent execution on the current branch with different params.
        divergence = next(
            (
                e for e in reversed(prior)
                if e["branch_id"] == self._branch.branch_id
                and e["param_fingerprint"] != fp
            ),
            None,
        )

        if divergence is None:
            return self._current_state_id

        # Auto-generate a branch name that captures the context.
        branch_name = f"branch-{func_name[:24]}-{divergence['input_state_id'][:8]}"
        new_branch = self._create_branch(
            starts_at=divergence["input_state_id"],
            name=branch_name,
        )
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
    ) -> tuple[Operation, OperationType]:
        """
        Return the (Operation, OperationType) pair for *func_name*, creating
        it on first encounter and caching it for the rest of the session.

        Operation represents the *definition* of what was called; OperationType
        is the semantic category resolved via the operation_registry (defaults
        to ``"unknown"`` when not registered).
        """
        if func_name in self._operation_cache:
            return self._operation_cache[func_name]

        type_name = _lookup_operation_type(func_name)
        op_type = OperationType(type_id=_uid(), name=type_name)

        # Check for a registered StepCategory for this OperationType
        category_name = _lookup_category(type_name)

        step_cat_id: Optional[str] = None
        if category_name:
            step_cat = StepCategory(category_id=_uid(), name=category_name)
            step_cat_id = step_cat.category_id
            self.storage.save_step_category_async(step_cat)

        operation = Operation(
            operation_id=_uid(),
            name=func_name,
            operation_type_id=op_type.type_id,
            step_category_id=step_cat_id,
        )
        self._operation_cache[func_name] = (operation, op_type)
        return operation, op_type

    def _build_artifact_records(
        self,
        obj: Any,
        output_state_id: str,
        artifact_path: str,
    ) -> dict:
        """
        Build Artifact + ArtifactState Pydantic models for a DataFrame output.

        Returns a dict with keys ``artifact_obj`` (None if already seen) and
        ``artifact_state_obj``.
        Side effects: updates _artifact_registry and _artifact_state_registry.
        """
        obj_python_id = id(obj)
        checksum_hex, size_bytes = _checksum_and_size(artifact_path)

        artifact: Optional[Artifact] = None
        artifact_id = self._artifact_registry.get(obj_python_id)
        if artifact_id is None:
            artifact_id = _uid()
            artifact = Artifact(
                artifact_id=artifact_id,
                name=f"dataframe_{artifact_id[:8]}",
                artifact_type=_infer_artifact_type(obj),
            )
            self._artifact_registry[obj_python_id] = artifact_id

        artifact_state = ArtifactState(
            artifact_state_id=_uid(),
            artifact_id=artifact_id,
            analysis_state_id=output_state_id,
            mime_type="application/vnd.apache.parquet",
            checksum=f"sha256:{checksum_hex}",
            content_ref=artifact_path,
            size_bytes=size_bytes,
        )
        self._artifact_state_registry[obj_python_id] = artifact_state.artifact_state_id
        return {"artifact_obj": artifact, "artifact_state_obj": artifact_state}

    def _make_param_value(self, param_id: str, step_id: str, value: Any):
        """Map a runtime Python value to the appropriate ParameterValue subclass."""
        if type(value).__name__ == "DataFrame":
            ast_id = self._artifact_state_registry.get(id(value))
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
                value=value[:50],
            )
        if isinstance(value, dict):
            return DictParameterValue(
                parameter_value_id=_uid(),
                parameter_id=param_id,
                step_id=step_id,
                value={k: v for k, v in list(value.items())[:50]},
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
