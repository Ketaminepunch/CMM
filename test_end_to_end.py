"""
End-to-end smoke test for decode_function_call against the real example
prompts/functions. Not part of the graded submission (per subject p.8,
throwaway test programs are expected, not graded).

Since prompts.py doesn't exist yet, this hand-writes a small few-shot
preamble so the model has *some* grounding for both function selection
and argument extraction - real prompt engineering is a separate,
later piece of work. The point here is only to confirm decode_function_call
itself produces valid, parseable JSON end to end.

Run with: uv run python test_end_to_end.py
"""

import json

from llm_sdk import Small_LLM_Model

from src.decoder import (
    TooManyTokens,
    build_bool_trie,
    build_function_trie,
    decode_function_call,
)
from src.grammar import TokenSets
from src.io_utils import load_json
from src.models import FunctionDefinitionList
from src.tokenizer_vocab import decode_ids, load_vocab

FEW_SHOT_PREAMBLE = (
    "Functions available:\n"
    "fn_add_numbers(a: number, b: number): Add two numbers together and "
    "return their sum.\n"
    "fn_greet(name: string): Generate a greeting message for a person by "
    "name.\n"
    "fn_reverse_string(s: string): Reverse a string and return the "
    "reversed result.\n\n"
    "Question: What is the sum of 7 and 8?\n"
    'Answer: {"name": "fn_add_numbers", "parameters": {"a": 7, "b": 8}}\n\n'
    "Question: Greet mario.\n"
    'Answer: {"name": "fn_greet", "parameters": {"name": "mario"}}\n\n'
)


def main() -> None:
    sdk = Small_LLM_Model()
    vocab = load_vocab(sdk.get_path_to_vocab_file())
    sets = TokenSets(vocab)

    raw_defs = load_json("./data/input/functions_definition.json")
    definitions = FunctionDefinitionList.model_validate(raw_defs).root
    name_trie, name_to_def = build_function_trie(sdk, definitions)
    bool_trie = build_bool_trie(sdk)

    raw_prompts = load_json("./data/input/function_calling_tests.json")
    prompts = [item["prompt"] for item in raw_prompts]

    for prompt in prompts:
        print("\n" + "=" * 60)
        print(f"PROMPT: {prompt}")

        full_prompt = FEW_SHOT_PREAMBLE + f"Question: {prompt}\nAnswer: "
        ids: list[int] = sdk.encode(full_prompt).tolist()[0]
        start = len(ids)

        try:
            name, params = decode_function_call(
                sdk, ids, name_trie, name_to_def, sets, vocab, bool_trie
            )
        except TooManyTokens:
            print("FAILED: hit the generation cap (TooManyTokens)")
            continue

        generated_text = decode_ids(ids[start:], vocab)
        print(f"raw generated JSON text: {generated_text}")
        print(f"decode_function_call returned: name={name!r}, params={params!r}")

        try:
            parsed = json.loads(generated_text)
        except json.JSONDecodeError as e:
            print(f"FAILED: generated text is not valid JSON: {e}")
            continue

        if parsed.get("name") != name or parsed.get("parameters") != params:
            print("FAILED: parsed JSON doesn't match the returned (name, params)")
            continue

        print("OK: valid JSON, matches returned (name, params)")


if __name__ == "__main__":
    main()
