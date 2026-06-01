"""Artifact, ArtifactState, and Delta models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    EVENT_LOG = "event_log"
    PROCESS_MODEL = "process_model"
    KPI_REPORT = "kpi_report"
    DATAFRAME = "dataframe"
    OTHER = "other"


class ModificationType(str, Enum):
    """High-level classification of what a Delta describes."""

    SCHEMA_CHANGE = "schema_change"
    ROW_FILTER = "row_filter"
    ROW_ADD = "row_add"
    VALUE_CHANGE = "value_change"
    TYPE_CHANGE = "type_change"
    OTHER = "other"


class Artifact(BaseModel):
    """
    Conceptual data object (e.g., an event log or process model) that persists
    across the analysis.  ArtifactStates are its versioned snapshots.

    UML attributes: artifactId, name, artifactType
    """

    artifact_id: str = Field(..., description="Unique identifier for the artifact.")
    name: str = Field(..., description="Human-readable name (e.g., 'RTFM event log').")
    artifact_type: ArtifactType = Field(..., description="Category of the artifact.")


class ArtifactState(BaseModel):
    """
    An immutable snapshot of an Artifact at a specific point in the analysis.
    Multiple ArtifactStates may be *included* in a single AnalysisState.

    UML attributes : artifactStateId, mimeType, checksum, contentRef, sizeBytes
    UML relationships:
        - includes  → AnalysisState  (via analysis_state_id)
        - roots     → Delta          (an ArtifactState is the root of a Delta chain)
        - updates   → Delta          (an ArtifactState is the updated end of a Delta)
        - snapshots → Artifact       (via artifact_id)
    """

    artifact_state_id: str = Field(..., description="Unique identifier for this snapshot.")
    artifact_id: str = Field(
        ..., description="FK → Artifact.artifact_id – the artifact being snapshotted."
    )
    analysis_state_id: str = Field(
        ...,
        description="FK → AnalysisState.state_id – the analysis state that includes this snapshot.",
    )
    mime_type: str = Field(
        ..., description="MIME type of the stored content (e.g., 'text/csv')."
    )
    checksum: str = Field(
        ..., description="Hash of the content for integrity verification (e.g., SHA-256)."
    )
    content_ref: str = Field(
        ...,
        description="Pointer to the actual stored content (file path, URI, object-store key, …).",
    )
    size_bytes: int = Field(..., description="Size of the stored content in bytes.")


class Delta(BaseModel):
    """
    Abstract base class that records the incremental structural change between
    two ArtifactStates, supporting Data Evolution Transparency (R7).

    UML attributes : deltaId, modificationType
    UML relationships:
        - roots   → ArtifactState  (via root_artifact_state_id)
        - updates → ArtifactState  (via updated_artifact_state_id)
    """

    delta_id: str = Field(..., description="Unique identifier for this delta.")
    modification_type: ModificationType = Field(
        ..., description="High-level classification of the change."
    )
    root_artifact_state_id: str = Field(
        ...,
        description="FK → ArtifactState.artifact_state_id – the state *before* the change.",
    )
    updated_artifact_state_id: str = Field(
        ...,
        description="FK → ArtifactState.artifact_state_id – the state *after* the change.",
    )


class DataFrameDelta(Delta):
    """
    Concrete Delta subclass for pandas DataFrame transformations.

    Extends Delta with DataFrame-specific change attributes used in the
    evaluation scenario (Sect. 6 of the paper).
    """

    columns_added: list[str] = Field(
        default_factory=list, description="Column names added in this step."
    )
    columns_removed: list[str] = Field(
        default_factory=list, description="Column names removed in this step."
    )
    dtype_changes: dict[str, str] = Field(
        default_factory=dict,
        description="Map of column name → new dtype string for columns whose type changed.",
    )
    rows_added: Optional[int] = Field(None, description="Number of rows added (if applicable).")
    rows_removed: Optional[int] = Field(
        None, description="Number of rows removed (if applicable)."
    )


__all__ = [
    "ArtifactType",
    "ModificationType",
    "Artifact",
    "ArtifactState",
    "Delta",
    "DataFrameDelta",
]
