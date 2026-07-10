"""
End-to-end smoke test for decode_function_call against the real example
prompts/functions. Not part of the graded submission (per subject p.8,
throwaway test programs are expected, not graded).

Now drives the real src/prompts.py (build_preamble/build_prompt) instead
of a hand-written preamble, so this also exercises prompts.py's dynamic,
schema-agnostic preamble construction end to end, not just decoder.py.

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
from src.prompts import build_preamble, build_prompt
from src.tokenizer_vocab import decode_ids, load_vocab


def main() -> None:
    sdk = Small_LLM_Model()
    vocab = load_vocab(sdk.get_path_to_vocab_file())
    sets = TokenSets(vocab)

    raw_defs = load_json("./data/input/functions_definition.json")
    definitions = FunctionDefinitionList.model_validate(raw_defs).root
    name_trie, name_to_def = build_function_trie(sdk, definitions)
    bool_trie = build_bool_trie(sdk)

    preamble = build_preamble(definitions)
    print("=== PREAMBLE (built once) ===")
    print(preamble)

    raw_prompts = load_json("./data/input/function_calling_tests.json")
    prompts = [item["prompt"] for item in raw_prompts]

    for prompt in prompts:
        print("\n" + "=" * 60)
        print(f"PROMPT: {prompt}")

        full_prompt = build_prompt(preamble, prompt)
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
