"""Operation definitions, their classification types, and versioning."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class OperationType(BaseModel):
    """
    Fine-grained classification of an operation (e.g., 'attribute_derivation',
    'case_filter', 'conformance_check').

    UML attributes: typeId, name
    """

    type_id: str = Field(..., description="Unique identifier for this operation type.")
    name: str = Field(..., description="Human-readable label (e.g., 'attribute_derivation').")


class StepCategory(BaseModel):
    """
    Broad grouping of analysis steps (e.g., 'log_enrichment', 'process_discovery',
    'conformance_checking').

    UML attributes: categoryId, name
    """

    category_id: str = Field(..., description="Unique identifier for this step category.")
    name: str = Field(..., description="Human-readable label (e.g., 'log_enrichment').")


class Operation(BaseModel):
    """
    The abstract, reusable definition of an analysis task (e.g., 'filter by threshold',
    'derive case-centric view').  Concrete executions are recorded as AnalysisSteps.

    UML attributes: operationId, name
    UML relationships:
        - hasType   → OperationType (via operation_type_id)
        - belongsTo → StepCategory  (via step_category_id)
        - aggregates → Parameter    (1-to-N; represented in Parameter.operation_id)
    """

    operation_id: str = Field(..., description="Unique identifier for this operation definition.")
    name: str = Field(..., description="Human-readable operation name (e.g., 'add_attribute').")
    operation_type_id: str = Field(
        ...,
        description="FK → OperationType.type_id – the fine-grained type of this operation.",
    )
    step_category_id: Optional[str] = Field(
        None,
        description=(
            "FK → StepCategory.category_id – the broad category this operation belongs to. "
            "Optional if not yet categorised."
        ),
    )


class Version(BaseModel):
    """
    Governs individual operations within a Pipeline, capturing creation timestamps
    and hierarchical lineage to maintain the integrity of reusable components.

    UML attributes: versionId, number, createdAt
    UML relationship:
        - originatedFrom → Version    (via originated_from_version_id – parent version)
        - references     → Operation  (via operation_id)
    """

    version_id: str = Field(..., description="Unique identifier for this version record.")
    operation_id: str = Field(
        ...,
        description="FK → Operation.operation_id – the operation this version tracks.",
    )
    number: str = Field(
        ..., description="Semantic version string (e.g., '1.0.0', '2.1.3')."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this version was created.",
    )
    originated_from_version_id: Optional[str] = Field(
        None,
        description=(
            "FK → Version.version_id – the parent version from which this one was derived. "
            "None for the initial version."
        ),
    )


__all__ = [
    "OperationType",
    "StepCategory",
    "Operation",
    "Version",
]
