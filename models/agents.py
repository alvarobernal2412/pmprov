"""Agent and execution-environment models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Distinguishes human analysts from automated scripts / systems."""

    HUMAN = "human"
    AUTOMATED = "automated"


class Agent(BaseModel):
    """
    Represents the entity (human or system) that performs an AnalysisStep.

    UML attributes: agentId, agentType
    """

    agent_id: str = Field(..., description="Unique identifier for the agent.")
    agent_type: AgentType = Field(
        ..., description="Whether the agent is a human analyst or an automated system."
    )
    username: Optional[str] = Field(
        None, description="OS username of the agent, captured at session start."
    )


class RuntimeEnvironment(BaseModel):
    """
    Captures the execution environment of an AnalysisStep to enable reproducibility (R2).

    UML attributes: envId, toolVersion, libraryVersions, runtime
    """

    env_id: str = Field(..., description="Unique identifier for this environment snapshot.")
    tool_version: str = Field(
        ..., description="Version of the primary analysis tool (e.g., Python 3.11)."
    )
    library_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of library name → version string (e.g., {'pandas': '2.2.1'}).",
    )
    runtime: Optional[str] = Field(
        None,
        description="Free-form description of the runtime platform (OS, hardware, etc.).",
    )


__all__ = [
    "AgentType",
    "Agent",
    "RuntimeEnvironment",
]
