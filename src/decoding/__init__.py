"""Grammar-constrained decoding of function calls, token by token.

This package is split by concern:

- :mod:`src.decoding.primitives` -- generic, value-type-agnostic
  machinery (masked argmax, running an FSM to completion, walking a
  trie, writing literal text straight into the id stream).
- :mod:`src.decoding.values` -- decoding logic for each JSON value
  type (number, string, boolean) and for a full function call, built
  on top of those primitives.

Everything below is re-exported here so callers can keep writing
``from src.decoding import decode_function_call`` without caring how
the package is internally organized.
"""

from src.decoding.primitives import (
    TokenFSM,
    TooManyTokens,
    masked_argmax,
    run_fsm,
    walk_trie,
    write_literal,
)
from src.decoding.values import (
    build_bool_trie,
    build_function_trie,
    decode_function_call,
    gen_bool,
    gen_number,
    gen_string,
)

__all__ = [
    "TokenFSM",
    "TooManyTokens",
    "build_bool_trie",
    "build_function_trie",
    "decode_function_call",
    "gen_bool",
    "gen_number",
    "gen_string",
    "masked_argmax",
    "run_fsm",
    "walk_trie",
    "write_literal",
]
