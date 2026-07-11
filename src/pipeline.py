"""Wires the model, grammar, and prompts together to run function calling
end to end: load everything once into a :class:`PipelineContext`, then
decode a function call for each prompt.
"""

from dataclasses import dataclass

from llm_sdk import Small_LLM_Model
from src.grammar import TokenSets, Trie
from src.models import FunctionCallResult, FunctionDefinition, PromptItem
from src.tokenizer_vocab import load_vocab

from .decoding import (
    TooManyTokens,
    build_bool_trie,
    build_function_trie,
    decode_function_call,
)
from .prompts import build_preamble, build_prompt


@dataclass
class PipelineContext:
    """Everything decoding needs, built once and reused per prompt."""

    sdk: Small_LLM_Model
    vocab: dict[int, str]
    sets: TokenSets
    name_trie: Trie
    name_to_def: dict[str, FunctionDefinition]
    bool_trie: Trie
    preamble: str


def build_context(definitions: list[FunctionDefinition]) -> PipelineContext:
    """Load the model and vocab, and precompute everything decoding needs.

    Args:
        definitions: The function catalog available for calling.

    Returns:
        A :class:`PipelineContext` ready to be passed to
        :func:`run_single` / :func:`run_pipeline`.
    """
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


def run_single(ctx: PipelineContext, prompt: str) -> FunctionCallResult | None:
    """Decode a single prompt into a function call.

    Args:
        ctx: The shared pipeline context from :func:`build_context`.
        prompt: The natural-language question to answer.

    Returns:
        The decoded :class:`FunctionCallResult`, or ``None`` if
        decoding hit the token cap before finishing.
    """
    prompt_txt = build_prompt(ctx.preamble, prompt)
    ids: list[int] = ctx.sdk.encode(prompt_txt).tolist()[0]
    try:
        name, params = decode_function_call(
            ctx.sdk,
            ids,
            ctx.name_trie,
            ctx.name_to_def,
            ctx.sets,
            ctx.vocab,
            ctx.bool_trie,
        )
        return FunctionCallResult(prompt=prompt, name=name, parameters=params)
    except TooManyTokens:
        print("Too many tokens used")
        return None


def run_pipeline(
    ctx: PipelineContext, prompts: list[PromptItem]
) -> list[FunctionCallResult]:
    """Run every prompt through :func:`run_single`, logging progress.

    Args:
        ctx: The shared pipeline context from :func:`build_context`.
        prompts: The prompts to decode, in order.

    Returns:
        The successfully decoded results, in the same order as the
        input prompts (failed prompts are skipped, not padded).
    """
    result_list = []
    total = len(prompts)
    for index, item in enumerate(prompts, start=1):
        result = run_single(ctx, item.prompt)
        if result is not None:
            result_list.append(result)
            print(
                f"[{index}/{total}] {item.prompt}: \n {result.name} {result.parameters}"
            )
        else:
            print("Question failed")
    return result_list
