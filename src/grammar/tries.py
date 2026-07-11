"""Trie over token-id sequences, used to constrain generation to a
closed set of literal choices (function names, ``true``/``false``).
"""

from dataclasses import dataclass, field


@dataclass
class TrieNode:
    """One node in a :class:`Trie`.

    ``children`` maps a token id to the node reached by consuming
    that token. ``name`` is set only on nodes that terminate a
    complete entry (e.g. a full function name), and is ``None`` for
    every other node.
    """

    children: dict[int, "TrieNode"] = field(default_factory=dict)
    name: str | None = None


class Trie:
    """Trie over token-id sequences (function names, true/false)."""

    def __init__(self) -> None:
        """Create an empty trie with just a root node."""
        self.root = TrieNode()

    def insert(self, ids: list[int], name: str) -> None:
        """Insert a token-id sequence, labelling its terminal node.

        Args:
            ids: The sequence of token ids that spells out ``name``.
            name: The label to attach to the node reached after
                consuming every id in ``ids``.
        """
        node = self.root
        for tid in ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.name = name


def build_trie(entries: list[tuple[list[int], str]]) -> Trie:
    """Build a :class:`Trie` from a list of (token ids, label) pairs."""
    trie = Trie()
    for ids, name in entries:
        trie.insert(ids, name)
    return trie
