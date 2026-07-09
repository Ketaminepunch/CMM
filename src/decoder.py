from typing import Protocol

import numpy as np

from grammar import NumberFSM, TokenSets, Trie, build_trie
from llm_sdk import Small_LLM_Model
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
    sdk: Small_LLM_Model, ids: list[int], fsm: TokenFSM, max_tokens: int = 50
) -> list[int]:
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
    text_lst = sdk.encode(text).tolist()
    ids.extend(text_lst[0])


def build_function_trie(
    sdk: Small_LLM_Model, definitions: list[FunctionDefinition]
) -> tuple[Trie, dict[str, FunctionDefinition]]:
    pairs_list: list[tuple[list[int], str]] = []
    name_def: dict = {}
    for defn in definitions:
        tokenized_defn = sdk.encode(defn.name).tolist()
        pairs_list.append((tokenized_defn[0], defn.name))
        name_def[defn.name] = defn
    trie = build_trie(pairs_list)
    return trie, name_def


def build_bool_trie(sdk: Small_LLM_Model) -> Trie:
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
    terminator_id = sets.close_brace if is_last else sets.comma
    fsm = NumberFSM(sets, terminator_id)
    sliced = run_fsm(sdk, ids, fsm)
    return float(decode_ids(sliced[:-1], vocab))
