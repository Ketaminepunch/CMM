"""Loading the tokenizer vocabulary and turning token ids back into text.

The tokenizer's vocab file maps each token to a string made of
"visible" printable characters (GPT-2-style byte-level BPE), where
each character stands for one raw byte. :data:`BYTE_DECODER` inverts
that mapping so token strings can be turned back into the original
bytes and decoded as UTF-8.
"""

import json
import sys


def _build_byte_decoder() -> dict[str, int]:
    """Build the character -> byte-value map used to undo byte-level BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256+n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DECODER = _build_byte_decoder()


def load_vocab(path: str) -> dict[int, str]:
    """Load a vocab file and invert it to a token-id -> token-string map.

    Args:
        path: Path to the tokenizer's vocab JSON file (token string
            to token id).

    Returns:
        The inverted mapping: token id to token string.
    """
    try:
        with open(path) as f:
            original: dict[str, int] = json.load(f)
            reverse = {v: k for k, v in original.items()}
    except FileNotFoundError:
        sys.exit(f"Error file not found at: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error invalid JSON in {path}: {e}")

    return reverse


def token_to_bytes(token_str: str) -> bytes:
    """Convert a token's visible-character string back to raw bytes."""
    return bytes([BYTE_DECODER[c] for c in token_str])


def decode_ids(ids: list[int], vocab: dict[int, str]) -> str:
    """Decode a sequence of token ids back into a UTF-8 string."""
    final = b""
    for id in ids:
        final += token_to_bytes(vocab[id])
    return final.decode("utf-8")
