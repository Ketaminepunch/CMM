"""Precomputed sets of token ids used to constrain decoding.

Grouping tokens by what they mean structurally (a digit, a quote, a
token that is safe inside a JSON string, ...) lets the rest of the
grammar code work in terms of "the allowed token ids" instead of
re-scanning the vocabulary on every decoding step.
"""

from src.tokenizer_vocab import token_to_bytes

DIGITS = set("0123456789")


class TokenSets:
    """Precomputed allowed-token-id sets, built once from the vocab."""

    def __init__(self, vocab: dict[int, str]) -> None:
        """Derive every token-id set from the tokenizer vocabulary.

        Args:
            vocab: Mapping of token id to its decoded string form.
        """
        by_str = {text: tid for tid, text in vocab.items()}
        self.quote = by_str['"']
        self.minus = by_str["-"]
        self.dot = by_str["."]
        self.comma = by_str[","]
        self.close_brace = by_str["}"]
        self.digits = [
            tid
            for tid, text in vocab.items()
            if all(ch in DIGITS for ch in text)
        ]
        self.string_body = [
            tid for tid, text in vocab.items() if _string_safe(text)
        ]


def _string_safe(text: str) -> bool:
    """True if the token can appear inside a JSON string body."""
    if text.startswith("<|"):
        return False
    raw = token_to_bytes(text)
    if b'"' in raw or b"\\" in raw:
        return False
    return all(byte >= 0x20 for byte in raw)
