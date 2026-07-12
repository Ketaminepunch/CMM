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
        self.quote_starts = [
            tid
            for tid, text in vocab.items()
            if token_to_bytes(text).startswith(b'"')
        ]
        self.quote_starts_set = set(self.quote_starts)
        self.quote_openers = [
            tid for tid, text in vocab.items() if _opens_string(text)
        ]


def _opens_string(text: str) -> bool:
    """True if the token can open a JSON string value.

    An opener is an optional space, the quote, then any body-safe
    payload — matching how tokenizers merge ``: "`` boundaries.
    """
    raw = token_to_bytes(text)
    if raw.startswith(b' "'):
        return _bytes_safe(raw[2:])
    if raw.startswith(b'"'):
        return _bytes_safe(raw[1:])
    return False


def _string_safe(text: str) -> bool:
    """True if the token can appear inside a JSON string body."""
    if text.startswith("<|"):
        return False
    return _bytes_safe(token_to_bytes(text))


def _bytes_safe(raw: bytes) -> bool:
    """True if these raw bytes are valid inside a JSON string body."""
    unsave_chars = [
        b"\xe2\x80\x9c",
        b"\xe2\x80\x9d",
        b"\xe2\x80\x98",
        b"\xe2\x80\x99",
    ]
    if any(seq in raw for seq in unsave_chars):
        return False
    i = 0
    while i < len(raw):
        if raw[i] == 0x5C:
            if i + 1 == len(raw):
                return False
            if raw[i + 1] not in b'"\\':
                return False
            else:
                i += 2
        elif raw[i] == 0x22:
            return False
        elif raw[i] < 0x20:
            return False
        else:
            i += 1
    return True
