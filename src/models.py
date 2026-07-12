"""Pydantic models for the data that flows in and out of the pipeline.

These mirror the on-disk JSON formats: the function catalog the
model is allowed to call, the list of prompts to run through it, and
the function-call results produced for each prompt.
"""

from pydantic import BaseModel, RootModel


class FunctionParameterSchema(BaseModel):
    """The declared type of a single function parameter or return value."""

    type: str


class FunctionDefinition(BaseModel):
    """One callable function, as described in functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, FunctionParameterSchema]
    returns: FunctionParameterSchema


class PromptItem(BaseModel):
    """One entry from the input prompts file."""

    prompt: str


class FunctionCallResult(BaseModel):
    """The function call decoded for a single prompt."""

    prompt: str
    name: str
    parameters: dict[str, str | float | int | bool]


class FunctionDefinitionList(RootModel[list[FunctionDefinition]]):
    """The full function catalog loaded from functions_definition.json."""

    pass


class PromptList(RootModel[list[PromptItem]]):
    """The full list of prompts loaded from the input prompts file."""

    pass
