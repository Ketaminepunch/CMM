"""
Quick smoke test for src/decoder.py. Not part of the graded submission —
just a manual sanity check that everything wired up so far actually runs
against the real model, per the subject's suggestion (p.8) to write
throwaway test programs.

Run with: uv run python test_decoder.py
"""

from llm_sdk import Small_LLM_Model
from src.decoder import (
    build_bool_trie,
    build_function_trie,
    decode_function_call,
    gen_bool,
    gen_number,
    gen_string,
    masked_argmax,
    walk_trie,
    write_literal,
)
from src.grammar import TokenSets
from src.io_utils import load_json
from src.models import FunctionDefinitionList
from src.tokenizer_vocab import load_vocab


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def test_masked_argmax() -> None:
    section("masked_argmax (pure logic, no model)")
    scores = [1.0, 5.0, 2.0, 9.0, 0.0]
    legal_ids = [0, 2, 4]  # index 3 (the real max) is NOT legal
    chosen = masked_argmax(scores, legal_ids)
    assert chosen == 2, (
        f"expected 2 (score 2.0 is the max among legal ids), got {chosen}"
    )
    print("OK - picks the best-scoring token among only the legal ones")


def main() -> None:
    test_masked_argmax()

    section("loading model (downloads Qwen3-0.6B on first run, be patient)")
    sdk = Small_LLM_Model()
    print("OK - model loaded")

    section("loading vocab + building TokenSets")
    vocab_path = sdk.get_path_to_vocab_file()
    vocab = load_vocab(vocab_path)
    sets = TokenSets(vocab)
    print(
        f"OK - vocab has {len(vocab)} tokens, {len(sets.digits)} digit tokens, "
        f"{len(sets.string_body)} string-safe tokens"
    )

    section("write_literal")
    ids: list[int] = sdk.encode("hello").tolist()[0]
    before = len(ids)
    write_literal(sdk, ids, ", ")
    assert len(ids) > before, "write_literal did not append anything"
    print(f"OK - ids grew from {before} to {len(ids)} tokens")

    section("loading real function definitions")
    raw = load_json("./data/input/functions_definition.json")
    definitions = FunctionDefinitionList.model_validate(raw).root
    print(
        f"OK - loaded {len(definitions)} functions: {[d.name for d in definitions]}"
    )

    section("build_function_trie")
    name_trie, name_to_def = build_function_trie(sdk, definitions)
    assert set(name_to_def.keys()) == {d.name for d in definitions}
    print("OK - trie + lookup dict built for all function names")

    section("build_bool_trie")
    bool_trie = build_bool_trie(sdk)
    print("OK - bool trie built")

    section(
        "walk_trie (function selection) - not yet driven by a real prompt template"
    )
    prompt_ids: list[int] = sdk.encode(
        "What is the sum of 2 and 3?\nFunction:"
    ).tolist()[0]
    chosen_name = walk_trie(sdk, list(prompt_ids), name_trie)
    assert chosen_name in name_to_def, (
        f"{chosen_name!r} is not a known function name"
    )
    print(
        f"OK - model picked {chosen_name!r} (structurally valid; accuracy is prompts.py's job, not decoder.py's)"
    )

    # NOTE: gen_number/gen_string/gen_bool are primed here with small
    # hand-written few-shot contexts, not real prompts.py output (which
    # doesn't exist yet). Without *some* grounding the model has no signal
    # for what value to produce or when to stop, and run_fsm's safety cap
    # will (correctly) fire - this bit us once already testing gen_number
    # against a bare '{"a": ' with no question attached.

    section("gen_number")
    num_prompt = (
        "Question: What is the sum of 7 and 8?\n"
        'Answer: {"a": 7, "b": 8}\n'
        "Question: What is the sum of 2 and 3?\n"
        'Answer: {"a": '
    )
    num_ids: list[int] = sdk.encode(num_prompt).tolist()[0]
    value = gen_number(sdk, num_ids, sets, vocab, is_last=False)
    assert isinstance(value, float)
    print(f"OK - generated number: {value}")

    section("gen_string")
    str_prompt = (
        "Question: Greet john.\n"
        'Answer: {"name": "john"}\n'
        "Question: Greet shrek.\n"
        'Answer: {"name": '
    )
    str_ids: list[int] = sdk.encode(str_prompt).tolist()[0]
    text = gen_string(sdk, str_ids, sets, vocab, is_last=True)
    assert isinstance(text, str)
    print(f"OK - generated string: {text!r}")

    section("gen_bool")
    bool_prompt = (
        "Question: Is 4 an even number?\n"
        'Answer: {"flag": true}\n'
        "Question: Is 7 an even number?\n"
        'Answer: {"flag": '
    )
    bool_ids: list[int] = sdk.encode(bool_prompt).tolist()[0]
    flag = gen_bool(sdk, bool_ids, bool_trie, is_last=True)
    assert isinstance(flag, bool)
    print(f"OK - generated bool: {flag}")

    section("gen_number - is a 'random number' prompt actually random?")
    # masked_argmax always takes the single highest-scoring legal token, so
    # decoding here is fully deterministic: the exact same prompt run twice
    # must produce the exact same number both times. No dice-rolling
    # anywhere in this pipeline - this demonstrates that directly.
    random_prompt = 'Question: Generate a random number between 32.233 and -212333.2 and .\nAnswer: {"a": '
    first_ids: list[int] = sdk.encode(random_prompt).tolist()[0]
    second_ids: list[int] = sdk.encode(random_prompt).tolist()[0]
    first_value = gen_number(sdk, first_ids, sets, vocab, is_last=True)
    second_value = gen_number(sdk, second_ids, sets, vocab, is_last=True)
    print(f"run 1: {first_value}, run 2: {second_value}")
    assert first_value == second_value, (
        "expected the same prompt to deterministically produce the same value"
    )
    print(
        "OK - confirmed deterministic: identical prompt gives the identical 'random' number every time"
    )

    section("decode_function_call (full generation - name + parameters)")
    call_prompt = (
        "Question: What is the sum of 7 and 8?\n"
        'Answer: {"name": "fn_add_numbers", "parameters": {"a": 7, "b": 8}}\n'
        "Question: What is the sum of 2 and 3?\n"
        "Answer: "
    )
    call_ids: list[int] = sdk.encode(call_prompt).tolist()[0]
    name, params = decode_function_call(
        sdk, call_ids, name_trie, name_to_def, sets, vocab, bool_trie
    )
    print(f"OK - selected function {name!r}, parameters: {params!r}")

    section("all checks passed")


if __name__ == "__main__":
    main()
