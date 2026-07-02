import json
import sys


def _build_byte_decoder() -> dict[str, int]:
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
    return bytes(BYTE_DECODER[c] for c in token_str)


def decode_ids(ids: list[int], vocab: dict[int, str]) -> str:
