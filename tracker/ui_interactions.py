"""
UI interaction capture for RuntimeTracker.

Imported as a side-effect from tracker/__init__.py.
Patches trace_ui_step() onto RuntimeTracker.

Design
------
An AnalysisStep is the operation an analyst performed — not only a
DataFrame-transforming function call, but equally "changed the chart's mark
to bar" or "moved the threshold slider to 5". trace_ui_step() therefore
builds a real AnalysisStep + AnalysisState pair, through the same
operation-registry / parameter-value machinery trace_step() uses for
function calls (_get_or_create_operation, _make_param_value,
_cell_executions).

Branching (this branch = manual-only model): a UI interaction is resolved
through the exact same _resolve_input_state() every regular trace_step()
call goes through. In practice this means flipping through several widget
values in the ordinary course of exploring a chart never forks anything —
it just appends, one UI step after another, same as any other repeated
call with no checkout in between. A branch only appears if the analyst
explicitly checks out to an earlier state and then interacts with a widget
differently than whatever already happened from there — see
_resolve_input_state()'s docstring for the exact replay/divergence rule.
(Contrast the `feat/ui-interactions` branch, where every differing
interaction auto-forks — which is the DAG-noise problem manual-only
branching exists to avoid in the first place.)

What's deliberately different from trace_step():
  - No pre/post snapshot or delta computation: there is no "before/after
    DataFrame" to diff, since a UI interaction never transforms data.
  - The new AnalysisState's artifact (when the widget has a backing
    DataFrame) is a cheap alias pointing at the parent's ArtifactState — see
    storage.py's "ui_state_alias" kind — never a re-persisted Parquet copy.
    A chatty widget (a dragged slider) must not write megabytes of duplicate
    data per interaction.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Optional

from models import AnalysisState, AnalysisStep
from tracker.logger import log_trace_warning
from tracker.runtime import _param_fingerprint, _uid

if TYPE_CHECKING:
    from tracker.runtime import RuntimeTracker


def _trace_ui_step(
    self: "RuntimeTracker",
    *,
    widget_type: str,
    changed: dict[str, Any],
    widget_label: Optional[str] = None,
    df: Optional[Any] = None,
) -> str:
    """
    Record a UI-widget interaction as a real AnalysisStep + AnalysisState,
    anchored to whatever state was current before the interaction.

    Call this from a marimo `on_change` / anywidget `.observe()` callback.

    Parameters
    ----------
    widget_type:
        The marimo UIElement class name, e.g. "data_explorer", "slider".
    changed:
        Widget-specific readable diff, e.g. {"mark": "bar"}. Persisted as
        this step's ParameterValues, exactly like a function call's kwargs.
    widget_label:
        Analyst-facing label for the widget, if one was given. Folded into
        this interaction's func_name (see below) so distinct widgets of the
        same type get distinct Operations and are branch-detected separately.
    df:
        The DataFrame the widget is currently showing, if any. Only used to
        alias the new state's artifact onto its existing ArtifactState (no
        data is read or copied) — omit for widgets with no backing DataFrame.

    Returns
    -------
    The new output_state_id.
    """
    func_name = f"ui:{widget_type}:{widget_label}" if widget_label else f"ui:{widget_type}"

    try:
        fp = _param_fingerprint([], changed)
    except Exception as e:
        log_trace_warning("UI interaction fingerprinting failed, using random UUID",
                          step="fingerprint", func_name=func_name, error=e)
        fp = str(uuid.uuid4())
    try:
        input_state_id = self._resolve_input_state(func_name, fp)
    except Exception as e:
        log_trace_warning("post-checkout branch resolution failed, staying on current branch",
                          step="branch_detection", func_name=func_name, error=e)
        input_state_id = self._current_state_id

    step_id = _uid()
    output_state_id = _uid()
    operation, op_type, step_cat = self._get_or_create_operation(func_name)

    step = AnalysisStep(
        step_id=step_id,
        input_state_id=input_state_id,
        output_state_id=output_state_id,
        agent_id=self._agent.agent_id,
        env_id=self._env.env_id,
        operation_id=operation.operation_id,
    )

    param_values: list[dict] = []
    try:
        for key, value in changed.items():
            pv = self._make_param_value(f"{func_name}:{key}", step_id, value)
            param_values.append(pv.model_dump(mode="json"))
    except Exception as e:
        log_trace_warning("UI interaction parameter serialisation failed",
                          step="param_values", func_name=func_name, error=e)

    artifact_records: dict = {}
    artifact_path: Optional[str] = None
    if df is not None:
        parent_artifact_state_id = self._artifact_state_registry.get(id(df))
        if parent_artifact_state_id:
            try:
                artifact_path = self.storage.save_artifact(
                    output_state_id, df, kind="ui_state_alias",
                    parent_artifact_state_id=parent_artifact_state_id,
                )
                if artifact_path:
                    artifact_records = self._build_artifact_records(df, output_state_id, artifact_path)
            except Exception as e:
                log_trace_warning("UI interaction artifact alias failed",
                                  step="ui_artifact_alias", func_name=func_name, error=e)
                artifact_path = None

    output_state = AnalysisState(
        state_id=output_state_id,
        history_id=self._history.history_id,
        branch_id=self._branch.branch_id,
        produced_by_step_id=step_id,
        derived_from_state_id=input_state_id,
    )
    raw_line = f"# UI interaction: {widget_type}" + (f" [{widget_label}]" if widget_label else "") + f" -> {changed}"

    self.storage.save_state_async(output_state)
    self.storage.save_step_async(step, func_name, raw_line, fp, self._history.history_id)
    self.storage.save_operation_async(operation, op_type, step_cat)
    if param_values:
        self.storage.save_param_values_async(param_values)
    # No real data delta for a UI interaction — recorded so readers of the
    # deltas table (e.g. visualizations.py) see an explicit marker rather
    # than a missing row that looks like an omission.
    self.storage.save_delta_async({"kind": "ui_interaction"}, step_id)
    if artifact_path:
        artifact_state_obj = artifact_records.get("artifact_state_obj")
        if artifact_state_obj:
            self.storage.save_artifact_records_async(
                artifact_records.get("artifact_obj"), artifact_state_obj, self._history.history_id
            )

    self._cell_executions.setdefault(func_name, []).append({
        "input_state_id": input_state_id,
        "output_state_id": output_state_id,
        "param_fingerprint": fp,
        "branch_id": self._branch.branch_id,
    })

    self._current_state_id = output_state_id
    self._history.active_state_id = output_state_id
    self.storage.update_history_active_state_async(self._history.history_id, output_state_id)

    return output_state_id


from tracker.runtime import RuntimeTracker  # noqa: E402

RuntimeTracker.trace_ui_step = _trace_ui_step
