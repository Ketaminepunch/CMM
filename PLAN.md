# "call me maybe" — Function Calling via Constrained Decoding

## Context

This is the 42 School "call me maybe" project (subject at `/sgoinfre/vsack/CMM/en.subject.pdf`). Goal: build a CLI tool that reads natural-language prompts, picks the right function from a provided schema, and extracts correctly-typed arguments — using a small local LLM (Qwen3-0.6B) — while **guaranteeing** 100% valid, schema-compliant JSON via **hand-built constrained decoding** (token-by-token logit masking). Prompting the model and hoping for valid JSON is explicitly forbidden by the subject.

Repo root for the submission: **`/sgoinfre/vsack/CMM/CMM`** — solo project, 42 login `vsack`. The provided `llm_sdk.zip` exposes `Small_LLM_Model` with only: `encode(text) -> Tensor`, `decode(ids) -> str`, `get_logits_from_input_ids(ids) -> list[float]`, `get_path_to_vocab_file() -> str`, plus `get_path_to_merges_file()` / `get_path_to_tokenizer_file()` (public, undocumented, useful for the bonus). Verified facts about `Qwen/Qwen3-0.6B`'s `vocab.json`: standard GPT-2-style byte-level BPE map (`{token_str: id}`, 151,643 entries, `Ġ` = leading space), and **numbers are always split into single-digit tokens** — the vocab contains exactly 10 all-digit tokens (`0`–`9`), so numeric values are generated one digit per step.

## Core design: token-level constrained decoding

The output the model contributes to is always `{"name": "<fn>", "parameters": {...}}` (compact JSON). `prompt` is known already and injected when the output file is assembled.

**Key idea (`src/grammar.py`, implemented): the grammar is token-granular, not character-granular.** Instead of walking each candidate token character-by-character through a pushdown automaton, every decoding state only allows tokens that are valid *in full* — so a token can never straddle a grammar-state boundary, and the character/token mismatch problem never occurs.

Three mechanisms, all model-free and unit-testable with fake logits:

1. **Structural JSON is never generated.** Braces, `"name":`, `"paramX":`, quotes, separators are tokenized via `sdk.encode()` and appended directly by the decoder — zero forward passes for structure. The model is only invoked at genuine decision points (function name, argument values, number termination).
2. **`Trie` over token-id sequences** for closed choices: function names, and `true`/`false` for booleans. At each step only the children of the current trie node are unmasked; a node carrying a name marks completion. Invalid or hallucinated function names are structurally impossible. Once the name completes, the pipeline commits to that function's parameter schema for the rest of generation.
3. **`TokenSets` + per-type FSMs** for open values. `TokenSets` is built once from `vocab.json` by content-filtering the whole vocab:
   - `digits` — tokens consisting only of `0-9`;
   - `string_body` — tokens whose decoded bytes contain no `"`, no `\`, and no control bytes (< 0x20), and are not special tokens (`<|...|>`). The byte check reuses the GPT-2 byte-decoder in `src/tokenizer_vocab.py`, so raw newlines inside a JSON string (invalid JSON) are impossible.

   `NumberFSM` implements `-?digits(.digits)?`: leading minus allowed once, one dot only, at least one digit required before termination and after a dot. Since a number has no closing delimiter, the terminator token (`,` or `}`) is part of the allowed set and chosen by the model. `StringFSM` covers the string body (opening quote is inserted by the decoder); after `max_tokens` (default 40) the only allowed token is the closing quote, so strings terminate deterministically — no logit-boost hacks. Decoding stops when the structure is complete; we never rely on the model's EOS token, so validity is structural, not probabilistic.

**Prompting strategy (`src/prompts.py`, not yet built) carries the semantic load.** Decided against the original per-parameter-type re-prompting idea in favor of a single combined preamble: one prompt built once per run, listing every function's `name`/`parameters`/`description` (rendered dynamically from whatever `FunctionDefinitionList` is loaded — never hardcoded to the example functions, since the subject tests against different schemas) plus 2-3 few-shot examples showing the full `{"name": ..., "parameters": {...}}` output shape, followed by the real question. A hardcoded version of exactly this shape was validated end-to-end (see the Decoder loop entry below) and got all 5 example prompts fully correct, so this is the starting point — only add per-parameter re-prompting later if accuracy testing against harder/ambiguous prompts shows the single-preamble approach isn't reliable enough. Simpler and not yet proven insufficient beats building the more elaborate version pre-emptively.

**Documented limitations (accepted trade-offs of the token-level approach):**
- String values cannot contain `"` or `\` (no escape sequences) or control characters.
- No exponent notation for numbers.
- Strings are capped at `max_tokens` tokens.

**Performance:** `get_logits_from_input_ids` has no KV-cache and reprocesses the full sequence every call, and the subject requires the full run in under 5 minutes. Mitigations: structure is forced (no model calls), `TokenSets` filtering runs once at startup, masking/argmax uses numpy, and the model is instantiated once and reused across all prompts.

## Repo layout (`/sgoinfre/vsack/CMM/CMM`)

```
CMM/
├── .gitignore
├── Makefile
├── pyproject.toml          # deps: pydantic, numpy, llm-sdk (path dep); tool.uv.package = false
├── uv.lock
├── README.md
├── llm_sdk/                # unzipped verbatim from llm_sdk.zip
├── src/
│   ├── __init__.py
│   ├── __main__.py          # argparse CLI: --functions_definition/--input/--output, defaults per spec
│   ├── models.py            # pydantic: FunctionParameterSchema, FunctionDefinition, PromptItem, FunctionCallResult
│   ├── io_utils.py          # robust JSON load/save, clear errors on missing/malformed files
│   ├── prompts.py           # NOT STARTED: single combined preamble (function descriptions + few-shot examples), built once per run
│   ├── tokenizer_vocab.py   # vocab.json loader, GPT-2 byte-decoder, decode_ids()
│   ├── grammar.py           # DONE: Trie, TokenSets, NumberFSM, StringFSM (model-free)
│   ├── decoder.py           # DONE: forced literals + masked-argmax/FSM-driven generation; mypy-clean; verified end-to-end
│   └── pipeline.py          # NOT STARTED: load -> per-prompt generate (per-item try/except) -> write output
├── data/
│   └── input/
│       ├── functions_definition.json   # subject's example schema, committed for demo
│       └── function_calling_tests.json # subject's example prompts, committed for demo
└── tests/                   # dev-only pytest sanity checks for grammar.py with fake logits
```

`pyproject.toml` uses `[tool.uv.sources] llm-sdk = { path = "llm_sdk", editable = true }` and `[tool.uv] package = false` so `uv sync` alone pulls everything. `Makefile` targets: `install`, `run`, `debug`, `clean`, `lint` (flake8 + the subject's mypy flags), `lint-strict` (flake8 + `mypy . --strict`); configs exclude `llm_sdk/`.

## Build order

1. ~~**Scaffold**: repo, pyproject/Makefile/.gitignore, module stubs, example JSON committed.~~ Done.
2. ~~**Models + IO**: pydantic schemas, safe JSON load/save with clear errors.~~ Done.
3. ~~**Tokenizer layer**: vocab loader + byte-decoder + `decode_ids`.~~ Done.
4. ~~**Grammar**: Trie, TokenSets, NumberFSM, StringFSM.~~ Done — flake8 + mypy --strict clean; sanity-checked against the real vocab.json (digit/string-body set sizes, `-3.14}` walk, dot/minus guards, string cap, trie shape).
5. ~~**Decoder loop** (`decoder.py`)~~ Done — flake8 + mypy (mandatory flags) clean. Built: `masked_argmax` (numpy `-inf` mask); `walk_trie` (closed-choice selection, used for both function names and booleans); `TokenFSM` (a `Protocol`, not a base class, so `NumberFSM`/`StringFSM` needed no changes — matched structurally once `done` was declared as a read-only property, not a plain mutable attribute); `run_fsm` (generic FSM-driving loop shared by numbers and strings, with a `max_tokens` cap that raises `TooManyTokens` instead of looping forever); `write_literal` (forced-literal fast path — encode + append, no forward pass); `build_function_trie` / `build_bool_trie` (trie construction from real `FunctionDefinition`s / the `true`/`false` literals); `gen_number` / `gen_string` / `gen_bool` (per-type value generators, each responsible for its own trailing separator except `gen_number`, whose terminator is folded into the FSM itself since a number has no delimiter of its own); and `decode_function_call` (the full orchestrator — name selection, then a parameter loop with `match`-based type dispatch, raising on unsupported types, with correct brace-balancing including the zero-parameter edge case). Verified end-to-end against the real Qwen3-0.6B model with a throwaway hand-primed prompt (not real `prompts.py` output): all 5 of the subject's example prompts produced valid, correctly-typed, round-trippable JSON (see `test_decoder.py` / `test_end_to_end.py` at repo root — dev-only, not graded, not part of the deliverable). Also fixed along the way: a duplicate stray `grammar.py` at the repo root (pre-restructure leftover) was shadowing `src/grammar.py` for `decoder.py`'s imports — deleted, and `decoder.py` now imports consistently via `src.grammar`.
6. **Prompts + pipeline + CLI** (not started): `prompts.py` — single combined preamble per the updated prompting strategy above, built once from the loaded `FunctionDefinitionList`, not rebuilt per-prompt. `pipeline.py` / `__main__.py` — instantiate the model, vocab, `TokenSets`, and both tries exactly once; loop over every prompt calling `decode_function_call`; catch `TooManyTokens` (and any other decode failure) per-prompt so one bad case can't crash the whole run; assemble `{prompt, name, parameters}` via `FunctionCallResult` and write `function_calling_results.json`.
7. **README.md**: italic first line (`*This project has been created as part of the 42 curriculum by vsack.*`), Description, Instructions, Resources (+ AI-usage disclosure), plus: Algorithm explanation, Design decisions (token-level grammar and its trade-offs), Performance analysis, Challenges faced, Testing strategy, Example usage.
8. **End-to-end verification**: `make install && make lint && make run` against the real Qwen3-0.6B on the subject's example data; check output validity/schema/timing (<5 min). Stress-test beyond the examples: floats, negative numbers, numbers written as words, strings with apostrophes/special characters, ambiguous prompts, multi-parameter functions with similar parameters.
9. **Bonus** (only after mandatory verified): from-scratch BPE encoder via `get_path_to_merges_file()` to drop `sdk.encode()` (the vocab-based `decode_ids` already avoids `sdk.decode()` in the main path), caching/batching, nested-argument support.

## Verification plan

- `uv run flake8 .` and `uv run mypy .` (both lint targets) clean.
- pytest sanity tests for grammar.py with fake logits (no model download needed).
- `make run` on the committed example `data/input/*.json` (2+3, 265+345, greet shrek, greet john, reverse 'hello') → inspect `data/output/function_calling_results.json`: valid JSON, correct `name`/`parameters`, matches the subject's example output shape.
- Deliberately corrupt/remove an input file and confirm a clear error instead of a crash.
- Time the full run to confirm well under 5 minutes.
