"""Generic, value-type-agnostic building blocks for constrained decoding.

These operate purely in terms of token ids and grammar objects
(:class:`~src.grammar.Trie`, any FSM satisfying :class:`TokenFSM`);
they have no idea whether they're producing a function name, a
number, or a string. That knowledge lives in
:mod:`src.decoding.values`.
"""

from typing import Protocol

import numpy as np

from llm_sdk import Small_LLM_Model
from src.grammar import Trie


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
