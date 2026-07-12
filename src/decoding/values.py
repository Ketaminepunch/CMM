"""Decoding logic for individual JSON values and full function calls.

Builds on the generic machinery in :mod:`src.decoding.primitives` to
generate each JSON value type (number, string, boolean) under its
matching grammar, and to assemble a full ``{"name": ..., "parameters":
{...}}`` function call one field at a time.
"""

import json

from llm_sdk import Small_LLM_Model
from src.decoding.primitives import (
    masked_argmax,
    run_fsm,
    walk_trie,
    write_literal,
)
from src.grammar import NumberFSM, StringFSM, TokenSets, Trie, build_trie
from src.models import FunctionDefinition
from src.tokenizer_vocab import decode_ids, token_to_bytes


def build_function_trie(
    sdk: Small_LLM_Model, definitions: list[FunctionDefinition]
) -> tuple[Trie, dict[str, FunctionDefinition]]:
    """
    Builds a trie that maps function names to their
    corresponding tokens.
    """
    pairs_list: list[tuple[list[int], str]] = []
    name_def: dict = {}
    for defn in definitions:
        tokenized_defn = sdk.encode(defn.name).tolist()
        pairs_list.append((tokenized_defn[0], defn.name))
        name_def[defn.name] = defn
    trie = build_trie(pairs_list)
    return trie, name_def


def build_bool_trie(sdk: Small_LLM_Model) -> Trie:
    """
    Builds a trie that maps boolean literals to their
    corresponding tokens.
    """
    bool_lst = ["true", "false"]
    pairs_list = []
    for item in bool_lst:
        tokenized_bool = sdk.encode(item).tolist()
        pairs_list.append((tokenized_bool[0], item))
    trie = build_trie(pairs_list)
    return trie


def gen_number(
    sdk: Small_LLM_Model,
    ids: list[int],
    sets: TokenSets,
    vocab: dict[int, str],
    is_last: bool,
) -> float:
    """
    Generates a number from the given vocabulary.
    """
    terminator_id = sets.close_brace if is_last else sets.comma
    fsm = NumberFSM(sets, terminator_id)
    sliced = run_fsm(sdk, ids, fsm)
    return float(decode_ids(sliced[:-1], vocab))


def gen_string(
    sdk: Small_LLM_Model,
    ids: list[int],
    sets: TokenSets,
    vocab: dict[int, str],
    is_last: bool,
) -> str:
    """
    Generates a string from the given vocabulary.
    """
    # Opening quote: let the model pick any token that starts with '"'
    # (token healing) — whatever trails the quote begins the value.
    scores = sdk.get_logits_from_input_ids(ids)
    opener = masked_argmax(scores, sets.quote_openers)
    ids.append(opener)
    raw = token_to_bytes(vocab[opener])
    prefix = raw[raw.index(b'"') + 1:].decode("utf-8")
    string_terminator = "}" if is_last else ","
    fsm = StringFSM(sets)
    sliced = run_fsm(sdk, ids, fsm)[:-1]
    # The closer may also be merged (e.g. '",'); keep the context a
    # cleanly closed string before the decoder writes what follows.
    ids[-1] = sets.quote
    decoded = prefix + decode_ids(sliced, vocab)
    decoded = json.loads('"' + decoded + '"').removeprefix(" ")
    write_literal(sdk, ids, string_terminator)
    return decoded


def gen_bool(
    sdk: Small_LLM_Model, ids: list[int], trie: Trie, is_last: bool
) -> bool:
    """
    Generates a boolean from the given vocabulary.
    """
    trie_result = walk_trie(sdk, ids, trie)
    bool_terminator = "}" if is_last else ","
    write_literal(sdk, ids, bool_terminator)
    return trie_result == "true"


def decode_function_call(
    sdk: Small_LLM_Model,
    ids: list[int],
    name_tries: Trie,
    name_to_def: dict[str, FunctionDefinition],
    sets: TokenSets,
    vocab: dict[int, str],
    bool_trie: Trie,
) -> tuple[str, dict[str, str | float | int | bool]]:
    """Decode a function call from a sequence of tokens."""
    write_literal(sdk, ids, '{"name": "')
    trie_result = walk_trie(sdk, ids, name_tries)
    write_literal(sdk, ids, '", "parameters": {')
    func_defn = name_to_def[trie_result]
    defn_lst = list(func_defn.parameters.items())
    params: dict[str, str | float | int | bool] = {}
    for i, (key, schema) in enumerate(defn_lst):
        if i == len(defn_lst) - 1:
            is_last = True
        else:
            is_last = False
        # Strings get no trailing space: the opener token supplies
        # it (' "'), keeping the space+quote boundary natural.
        if schema.type == "string":
            write_literal(sdk, ids, f'"{key}":')
        else:
            write_literal(sdk, ids, f'"{key}": ')
        match schema.type:
            case "number":
                params[key] = gen_number(sdk, ids, sets, vocab, is_last)
            case "string":
                params[key] = gen_string(sdk, ids, sets, vocab, is_last)
            case "boolean":
                params[key] = gen_bool(sdk, ids, bool_trie, is_last)
            case "integer":
                params[key] = int(gen_number(sdk, ids, sets, vocab, is_last))
            case _:
                raise ValueError(f"Unsupported parameter type: {schema.type}")
    if len(defn_lst) == 0:
        write_literal(sdk, ids, "}}")
    else:
        write_literal(sdk, ids, "}")
    return trie_result, params
