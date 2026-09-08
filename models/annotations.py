"""Annotation, AnnotationTarget, Tag, and TagAssignment models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class AnnotatableType(str, Enum):
    """Kinds of objects that Annotations and Tags can be attached to."""

    ARTIFACT_STATE = "artifact_state"
    ANALYSIS_STATE = "analysis_state"
    BRANCH = "branch"
    ANALYSIS_HISTORY = "analysis_history"
    STEP = "step"
    PIPELINE = "pipeline"


class Annotation(BaseModel):
    """
    A free-text note attached to one or more Annotatable objects (e.g. a set of
    AnalysisStates and AnalysisSteps that together form a notable subtree).

    UML relationships:
        - authoredBy → Agent              (via agent_id)
        - targets    → AnnotationTarget   (1-to-N; represented in AnnotationTarget.annotation_id)
    """

    annotation_id: str = Field(..., description="Unique identifier for this annotation.")
    text: str = Field(..., description="The note's content.")
    agent_id: str = Field(
        ..., description="FK → Agent.agent_id – the agent that authored this annotation."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this annotation was created.",
    )


class AnnotationTarget(BaseModel):
    """
    Links a single Annotation to one Annotatable object. An Annotation with
    multiple AnnotationTargets is a single note shared across a set of objects.

    UML relationships:
        - belongs to → Annotation (via annotation_id)
    """

    annotation_id: str = Field(
        ..., description="FK → Annotation.annotation_id – the annotation being targeted."
    )
    target_type: AnnotatableType = Field(..., description="Kind of object being annotated.")
    target_id: str = Field(
        ..., description="ID of the annotated object, interpreted according to target_type."
    )


class Tag(BaseModel):
    """
    A short, reusable label. The same Tag may be assigned to many Annotatable
    objects via TagAssignment.

    UML relationships:
        - assignedTo → TagAssignment (1-to-N; represented in TagAssignment.tag_id)
    """

    tag_id: str = Field(..., description="Unique identifier for this tag.")
    name: str = Field(..., description="Short label text (e.g. 'reviewed', 'outlier').")


class TagAssignment(BaseModel):
    """
    Links a single Tag to one Annotatable object.

    UML relationships:
        - assigns → Tag (via tag_id)
    """

    assignment_id: str = Field(..., description="Unique identifier for this assignment.")
    tag_id: str = Field(..., description="FK → Tag.tag_id – the tag being assigned.")
    target_type: AnnotatableType = Field(..., description="Kind of object being tagged.")
    target_id: str = Field(
        ..., description="ID of the tagged object, interpreted according to target_type."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this tag was assigned.",
    )


__all__ = [
    "AnnotatableType",
    "Annotation",
    "AnnotationTarget",
    "Tag",
    "TagAssignment",
]
