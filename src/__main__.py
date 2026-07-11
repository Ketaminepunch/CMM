"""CLI entry point: load inputs, run the pipeline, save the results.

Run with ``uv run python -m src`` (see the Makefile's ``run`` target).
"""

import argparse

from pydantic import ValidationError

from src.pipeline import build_context, run_pipeline

from .io_utils import load_json, save_json
from .models import FunctionDefinitionList, PromptList


def main() -> None:
    """Parse CLI args, run the function-calling pipeline, save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output", default="data/output/function_calling_results.json"
    )
    args = parser.parse_args()

    defintion_str = load_json(args.functions_definition)
    prompt_str = load_json(args.input)
    try:
        definitions = FunctionDefinitionList.model_validate(defintion_str)
        prompts = PromptList.model_validate(prompt_str)
    except ValidationError as e:
        print(e)
        return
    ctx = build_context(definitions.root)
    results = run_pipeline(ctx, prompts.root)
    formatted = [r.model_dump() for r in results]
    save_json(formatted, args.output)


if __name__ == "__main__":
    main()
