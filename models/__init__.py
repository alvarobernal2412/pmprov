"""
Pydantic data model for Analytic Provenance in Exploratory Process Mining.

Derived from: "A Conceptual Model to Enhance Process Mining Analysis with Provenance"
(ER 2026, blinded for review)

Package layout
--------------
agents      – Agent, AgentType, RuntimeEnvironment
artifacts   – Artifact, ArtifactType, ArtifactState, Delta, DataFrameDelta, ModificationType
parameters  – Parameter, ParameterValueType, and all ParameterValue subclasses / union
operations  – OperationType, StepCategory, Operation, Version
analysis    – AnalysisStep, AnalysisState, StateAbstraction
history     – AnalysisBranch, AnalysisHistory
pipelines   – PipelineFragment, Pipeline
"""

from .agents import Agent, AgentType, RuntimeEnvironment
from .analysis import AnalysisState, AnalysisStep, StateAbstraction
from .artifacts import (
    Artifact,
    ArtifactState,
    ArtifactType,
    DataFrameDelta,
    Delta,
    ModificationType,
)
from .history import AnalysisBranch, AnalysisHistory
from .operations import Operation, OperationType, StepCategory, Version
from .parameters import (
    ArtifactStateParameterValue,
    DictParameterValue,
    LambdaParameterValue,
    ListParameterValue,
    Parameter,
    ParameterValue,
    ParameterValueType,
    ScalarParameterValue,
)
from .pipelines import Pipeline, PipelineFragment

__all__ = [
    # agents
    "AgentType",
    "Agent",
    "RuntimeEnvironment",
    # artifacts
    "ArtifactType",
    "ModificationType",
    "Artifact",
    "ArtifactState",
    "Delta",
    "DataFrameDelta",
    # parameters
    "ParameterValueType",
    "Parameter",
    "ScalarParameterValue",
    "ArtifactStateParameterValue",
    "LambdaParameterValue",
    "ListParameterValue",
    "DictParameterValue",
    "ParameterValue",
    # operations
    "OperationType",
    "StepCategory",
    "Operation",
    "Version",
    # analysis
    "StateAbstraction",
    "AnalysisState",
    "AnalysisStep",
    # history
    "AnalysisBranch",
    "AnalysisHistory",
    # pipelines
    "PipelineFragment",
    "Pipeline",
]
