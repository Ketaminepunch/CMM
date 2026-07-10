from typing import Protocol

import numpy as np

from llm_sdk import Small_LLM_Model
from src.grammar import NumberFSM, StringFSM, TokenSets, Trie, build_trie
from src.models import FunctionDefinition
from src.tokenizer_vocab import decode_ids


class TooManyTokens(Exception):
    """Raised when the decoder tries to consume too many tokens."""

    pass


class TokenFSM(Protocol):
    """
    A finite state machine that maps a sequence of tokens to a
    sequence of actions.
    """

    @property
    def done(self) -> bool: ...

    def allowed(self) -> list[int]: ...
    def advance(self, token_id: int) -> None: ...


def masked_argmax(scores: list[float], legal_ids: list[int]) -> int:
    """
    Masks the scores of legal_ids and returns the index of the
    maximum score.
    """
    scores_array = np.array(scores)
    masked_arr = np.full(len(scores), -np.inf)
    masked_arr[legal_ids] = scores_array[legal_ids]
    return int(np.argmax(masked_arr))


def walk_trie(sdk: Small_LLM_Model, ids: list[int], trie: Trie) -> str:
    """
    Walks the trie to find the name of the node that
    corresponds to the given ids.
    """
    current_node = trie.root
    while current_node.children:
        legal_ids = list(current_node.children.keys())
        scores = sdk.get_logits_from_input_ids(ids)
        chosen_id = masked_argmax(scores, legal_ids)
        ids.append(chosen_id)
        current_node = current_node.children[chosen_id]
    assert current_node.name is not None
    return current_node.name


def run_fsm(
    sdk: Small_LLM_Model, ids: list[int], fsm: TokenFSM, max_tokens: int = 120
) -> list[int]:
    """
    Runs the FSM on the given ids and returns the resulting ids.
    """
    startlen = len(ids)
    while not fsm.done:
        if len(ids) - startlen > max_tokens:
            raise TooManyTokens()
        scores = sdk.get_logits_from_input_ids(ids)
        legal_ids = fsm.allowed()
        chosen_id = masked_argmax(scores, legal_ids)
        ids.append(chosen_id)
        fsm.advance(chosen_id)
    return ids[startlen:]


def write_literal(sdk: Small_LLM_Model, ids: list[int], text: str) -> None:
    """
    Writes a literal to the ids list.
    """
    text_lst = sdk.encode(text).tolist()
    ids.extend(text_lst[0])


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
    write_literal(sdk, ids, '"')
    string_terminator = "}" if is_last else ","
    fsm = StringFSM(sets)
    sliced = run_fsm(sdk, ids, fsm)[:-1]
    decoded = decode_ids(sliced, vocab)
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
) -> tuple[str, dict[str, str | float | bool]]:
    """Decode a function call from a sequence of tokens."""
    write_literal(sdk, ids, '{"name": "')
    trie_result = walk_trie(sdk, ids, name_tries)
    write_literal(sdk, ids, '", "parameters": {')
    func_defn = name_to_def[trie_result]
    defn_lst = list(func_defn.parameters.items())
    params: dict[str, str | float | bool] = {}
    for i, (key, schema) in enumerate(defn_lst):
        if i == len(defn_lst) - 1:
            is_last = True
        else:
            is_last = False
        write_literal(sdk, ids, f'"{key}": ')
        match schema.type:
            case "number":
                params[key] = gen_number(sdk, ids, sets, vocab, is_last)
            case "string":
                params[key] = gen_string(sdk, ids, sets, vocab, is_last)
            case "boolean":
                params[key] = gen_bool(sdk, ids, bool_trie, is_last)
            case _:
                raise ValueError(f"Unsupported parameter type: {schema.type}")
    if len(defn_lst) == 0:
        write_literal(sdk, ids, "}}")
    else:
        write_literal(sdk, ids, "}")
    return trie_result, params
