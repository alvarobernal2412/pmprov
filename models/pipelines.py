"""Pipeline and PipelineFragment models for reusable analysis workflows (R4)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PipelineFragment(BaseModel):
    """
    A named, reusable set of AnalysisSteps that act as templates within a Pipeline.
    Steps within a fragment are not bound to concrete AnalysisStates, making them
    context-independent and transferable across histories.

    UML attributes: fragmentId, name
    UML relationships:
        - defines → AnalysisStep (N-to-N; via step_ids)
        - used by → Pipeline     (via pipeline_id)
    """

    fragment_id: str = Field(..., description="Unique identifier for this pipeline fragment.")
    pipeline_id: str = Field(
        ...,
        description="FK → Pipeline.pipeline_id – the pipeline this fragment belongs to.",
    )
    name: str = Field(..., description="Descriptive name for this fragment (e.g., 'log_prep').")
    step_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of FK → AnalysisStep.step_id values. "
            "Steps are templates; their AnalysisState associations are optional within a fragment."
        ),
    )
    version_ids: list[str] = Field(
        default_factory=list,
        description="FK → Version.version_id values governing operations in this fragment.",
    )


class Pipeline(BaseModel):
    """
    An ordered composition of PipelineFragments that represents a complete,
    parameterizable reusable workflow (R4 – Reusability of Analysis Steps).

    UML attributes: pipelineId, name
    UML relationships:
        - aggregates (ordered) → PipelineFragment (1-to-N; represented in PipelineFragment.pipeline_id)
    """

    pipeline_id: str = Field(..., description="Unique identifier for this pipeline.")
    name: str = Field(..., description="Human-readable pipeline name.")
    fragment_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of FK → PipelineFragment.fragment_id values. "
            "Order defines the execution sequence of fragments."
        ),
    )


__all__ = [
    "PipelineFragment",
    "Pipeline",
]
