import numpy as np

from grammar import Trie, TrieNode
from llm_sdk.llm_sdk import Small_LLM_Model


def masked_argmax(scores: list[float], legal_ids: list[int]) -> int:
    scores_array = np.array(scores)
    masked_arr = np.full(len(scores), -np.inf)
    masked_arr[legal_ids] = scores_array[legal_ids]
    return int(np.argmax(masked_arr))


def walk_trie(sdk: Small_LLM_Model, ids: list[int], trie: Trie) -> str:
    current_node = trie.root
    while current_node.children:
        legal_ids = list(current_node.children.keys())
        scores = sdk.get_logits_from_input_ids(ids)
        chosen_id = masked_argmax(scores, legal_ids)
        ids.append(chosen_id)
        current_node = current_node.children[chosen_id]
    assert current_node.name is not None
    return current_node.name
