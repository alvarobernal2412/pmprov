"""AnalysisBranch and AnalysisHistory models."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class AnalysisBranch(BaseModel):
    """
    An independent analysis path within an AnalysisHistory.  Multiple branches
    may share a common ancestor state (R3 – Branching Capability).

    UML attributes: branchId
    UML relationships:
        - belongs to     → AnalysisHistory (via history_id)
        - startsAt       → AnalysisState   (via starts_at_state_id – divergence point)
        - hasActiveState → AnalysisState   (via active_state_id – current tip of the branch)
    """

    branch_id: str = Field(..., description="Unique identifier for this branch.")
    history_id: str = Field(
        ...,
        description="FK → AnalysisHistory.history_id – the history this branch belongs to.",
    )
    name: str = Field(
        "main",
        description="Human-readable branch label (e.g., 'main', 'experiment-1'). Defaults to 'main'.",
    )
    starts_at_state_id: str = Field(
        ...,
        description=(
            "FK → AnalysisState.state_id – the state at which this branch diverges "
            "(i.e., its common origin with the parent branch)."
        ),
    )


class AnalysisHistory(BaseModel):
    """
    Root container that encapsulates the entire lifecycle of an analytical process,
    including all of its branches.

    UML attributes: historyId
    UML relationships:
        - aggregates → AnalysisBranch (1-to-N; represented in AnalysisBranch.history_id)
    """

    history_id: str = Field(..., description="Unique identifier for this analysis history.")
    name: Optional[str] = Field(
        None, description="Optional human-readable title for the analysis session."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which this history was initiated.",
    )
    active_state_id: Optional[str] = Field(
        None,
        description=(
            "FK → AnalysisState.state_id – the most recently produced state in this "
            "history, regardless of branch.  Updated after every step so it always "
            "reflects where the analyst currently is."
        ),
    )


__all__ = [
    "AnalysisBranch",
    "AnalysisHistory",
]
