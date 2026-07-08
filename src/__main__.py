import json

from pydantic import ValidationError

from .io_utils import load_json
from .models import FunctionDefinitionList, PromptList

"""CLI entrypoint scaffold (placeholder, replaced during implementation)."""

if __name__ == "__main__":
    defintion_str = load_json("./data/input/functions_definition.json")
    prompt_str = load_json("./data/input/function_calling_tests.json")
    try:
        definitions = FunctionDefinitionList.model_validate(defintion_str)
        prompts = PromptList.model_validate(prompt_str)
        print(f"Model:{definitions.model_dump_json(indent=2)}")
        print(f"Valisdssdsdsddator:{prompts.model_dump_json(indent=2)}")
    except ValidationError as e:
        print(e)
