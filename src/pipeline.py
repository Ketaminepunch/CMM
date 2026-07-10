from dataclasses import dataclass

from llm_sdk import Small_LLM_Model
from src.grammar import TokenSets, Trie
from src.models import FunctionDefinition
from src.tokenizer_vocab import load_vocab

from .decoder import build_bool_trie, build_function_trie, decode_function_call
from .prompts import build_preamble, build_prompt


@dataclass
class PipelineContext:
    sdk: Small_LLM_Model
    vocab: dict[int, str]
    sets: TokenSets
    name_trie: Trie
    name_to_def: dict[str, FunctionDefinition]
    bool_trie: Trie
    preamble: str


def build_context(definitions: list[FunctionDefinition]) -> PipelineContext:
    sdk = Small_LLM_Model()
    vocab = load_vocab(sdk.get_path_to_vocab_file())
    sets = TokenSets(vocab)
    name_trie, name_to_def = build_function_trie(sdk, definitions)
    bool_trie = build_bool_trie(sdk)
    preamble = build_preamble(definitions)
    return PipelineContext(
        sdk=sdk,
        vocab=vocab,
        sets=sets,
        name_trie=name_trie,
        name_to_def=name_to_def,
        bool_trie=bool_trie,
        preamble=preamble,
    )
