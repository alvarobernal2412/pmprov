"""AnalysisStep, AnalysisState, and StateAbstraction models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class StateAbstraction(BaseModel):
    """
    A comparable summary or aggregated representation of an AnalysisState,
    enabling state-level and history-level comparison (R5, R6).

    UML attributes : abstractionId, type, value, function
    UML relationship:
        - wasAbstractedBy → AnalysisState (via analysis_state_id)
    """

    abstraction_id: str = Field(..., description="Unique identifier for this abstraction.")
    analysis_state_id: str = Field(
        ...,
        description="FK → AnalysisState.state_id – the state this abstraction summarises.",
    )
    abstraction_type: str = Field(
        ...,
        description=(
            "Category of abstraction (e.g., 'trace_variant_distribution', "
            "'performance_metric', 'feature_vector')."
        ),
    )
    value: Any = Field(
        ...,
        description="The computed abstraction value (scalar, dict, list, …).",
    )
    function: str = Field(
        ...,
        description=(
            "Identifier or source-code reference of the function used to compute "
            "this abstraction (e.g., a qualified Python callable name)."
        ),
    )


class AnalysisState(BaseModel):
    """
    An immutable snapshot of the overall analysis at a specific point in time.
    Acts as a container for one or more ArtifactStates.

    UML attributes : stateId, createdAt
    UML relationships:
        - usedInput        → AnalysisStep    (via produced_by_step_id)
        - outputProducedBy → AnalysisStep    (via produced_by_step_id)
        - wasDerivedFrom   → AnalysisState   (via derived_from_state_id)
        - hasActiveState   → AnalysisBranch  (via branch_id)
        - includes         → ArtifactState   (1-to-N; represented in ArtifactState.analysis_state_id)
        - wasAbstractedBy  → StateAbstraction (1-to-N; represented in StateAbstraction.analysis_state_id)
    """

    state_id: str = Field(..., description="Unique identifier for this analysis state.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this state was created.",
    )
    branch_id: str = Field(
        ...,
        description="FK → AnalysisBranch.branch_id – the branch this state belongs to.",
    )
    produced_by_step_id: Optional[str] = Field(
        None,
        description=(
            "FK → AnalysisStep.step_id – the step whose output produced this state. "
            "None for the initial ROOT state."
        ),
    )
    derived_from_state_id: Optional[str] = Field(
        None,
        description=(
            "FK → AnalysisState.state_id – the predecessor state in the analysis chain. "
            "None for the ROOT state."
        ),
    )


class AnalysisStep(BaseModel):
    """
    A concrete, immutable execution of an Operation within an AnalysisBranch.
    Bridges two AnalysisStates (input → output) and records the full execution context.

    UML attributes: stepId, createdAt
    UML relationships:
        - usedInput        → AnalysisState      (via input_state_id)
        - outputProducedBy → AnalysisState      (via output_state_id)
        - performedBy      → Agent              (via agent_id)
        - executesIn       → RuntimeEnvironment (via env_id)
        - executes         → Operation          (via operation_id)
        - produces         → Delta              (1-to-N; represented in Delta.root/updated refs)
        - instantiates     → ParameterValue     (1-to-N; represented in ParameterValue.step_id)
    """

    step_id: str = Field(..., description="Unique identifier for this analysis step.")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this step was executed.",
    )
    input_state_id: str = Field(
        ...,
        description="FK → AnalysisState.state_id – the state consumed as input.",
    )
    output_state_id: str = Field(
        ...,
        description="FK → AnalysisState.state_id – the state produced as output.",
    )
    agent_id: str = Field(
        ...,
        description="FK → Agent.agent_id – the agent that performed this step.",
    )
    env_id: str = Field(
        ...,
        description="FK → RuntimeEnvironment.env_id – the runtime environment used.",
    )
    operation_id: str = Field(
        ...,
        description="FK → Operation.operation_id – the operation that was executed.",
    )


__all__ = [
    "StateAbstraction",
    "AnalysisState",
    "AnalysisStep",
]
