from pydantic import BaseModel, RootModel


class FunctionParameterSchema(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FunctionParameterSchema]
    returns: FunctionParameterSchema


class PromptItem(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, str | float | bool]


class FunctionDefinitionList(RootModel[list[FunctionDefinition]]):
    pass


class PromptList(RootModel[list[PromptItem]]):
    pass
