"""Parameter definitions and their discriminated-union value types."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class ParameterValueType(str, Enum):
    """Discriminator tag for the concrete ParameterValue subclass."""

    SCALAR = "scalar"
    ARTIFACT_STATE_REF = "artifact_state_ref"
    LAMBDA_FUNCTION = "lambda_function"
    LIST = "list"
    DICT = "dict"


class Parameter(BaseModel):
    """
    Abstract definition of a configurable input of an Operation.

    UML attributes: name, type, required, default
    UML relationship:
        - belongs to → Operation (via operation_id)
    """

    parameter_id: str = Field(..., description="Unique identifier for this parameter definition.")
    operation_id: str = Field(
        ...,
        description="FK → Operation.operation_id – the operation this parameter belongs to.",
    )
    name: str = Field(..., description="Parameter name (e.g., 'threshold_days').")
    value_type: ParameterValueType = Field(
        ...,
        description=(
            "Expected kind of value for this parameter "
            "(e.g., 'scalar', 'artifact_state_ref', 'lambda_function', 'list', 'dict')."
        ),
    )
    required: bool = Field(..., description="Whether a value must be supplied at execution time.")
    default: Optional[Any] = Field(None, description="Default value if the parameter is optional.")


# ---------------------------------------------------------------------------
# ParameterValue – discriminated union
# ---------------------------------------------------------------------------


class _ParameterValueBase(BaseModel):
    """
    Shared identity fields for all ParameterValue subclasses.

    UML relationships:
        - instantiates → Parameter    (via parameter_id)
        - used in      → AnalysisStep (via step_id)
    """

    parameter_value_id: str = Field(..., description="Unique identifier for this value record.")
    parameter_id: str = Field(
        ...,
        description="FK → Parameter.parameter_id – the parameter definition being instantiated.",
    )
    step_id: str = Field(
        ...,
        description="FK → AnalysisStep.step_id – the step during which this value was used.",
    )


class ScalarParameterValue(_ParameterValueBase):
    """A primitive scalar value (int, float, str, bool, or None)."""

    value_type: Literal[ParameterValueType.SCALAR] = ParameterValueType.SCALAR
    value: Union[int, float, str, bool, None] = Field(
        ..., description="The concrete scalar value."
    )


class ArtifactStateParameterValue(_ParameterValueBase):
    """
    A reference to an ArtifactState used as a parameter (e.g., passing an event-log
    snapshot or a process model as input to an operation).
    """

    value_type: Literal[ParameterValueType.ARTIFACT_STATE_REF] = (
        ParameterValueType.ARTIFACT_STATE_REF
    )
    artifact_state_id: str = Field(
        ...,
        description=(
            "FK → ArtifactState.artifact_state_id – the snapshot being passed as the "
            "parameter value."
        ),
    )


class LambdaParameterValue(_ParameterValueBase):
    """
    A callable (lambda expression or named function) captured as source code.
    This preserves the exact transformation logic for reproducibility (R2).
    """

    value_type: Literal[ParameterValueType.LAMBDA_FUNCTION] = ParameterValueType.LAMBDA_FUNCTION
    source_code: str = Field(
        ...,
        description=(
            "Raw Python source of the callable "
            "(e.g., 'lambda x: x[\"concept:name\"] == \"Pay\"')."
        ),
    )
    function_name: Optional[str] = Field(
        None,
        description="Qualified name of the callable if it is a named function rather than a lambda.",
    )


class ListParameterValue(_ParameterValueBase):
    """An ordered sequence of values passed as a single parameter."""

    value_type: Literal[ParameterValueType.LIST] = ParameterValueType.LIST
    value: list[Any] = Field(..., description="The list of values.")


class DictParameterValue(_ParameterValueBase):
    """A key-value mapping passed as a single parameter."""

    value_type: Literal[ParameterValueType.DICT] = ParameterValueType.DICT
    value: dict[str, Any] = Field(..., description="The mapping of string keys to arbitrary values.")


ParameterValue = Annotated[
    Union[
        ScalarParameterValue,
        ArtifactStateParameterValue,
        LambdaParameterValue,
        ListParameterValue,
        DictParameterValue,
    ],
    Field(discriminator="value_type"),
]
"""
Discriminated union of all concrete ParameterValue types.
Use this type annotation wherever a parameter value is stored or validated.
Pydantic selects the correct subclass automatically based on the ``value_type`` field.
"""


__all__ = [
    "ParameterValueType",
    "Parameter",
    "ScalarParameterValue",
    "ArtifactStateParameterValue",
    "LambdaParameterValue",
    "ListParameterValue",
    "DictParameterValue",
    "ParameterValue",
]
