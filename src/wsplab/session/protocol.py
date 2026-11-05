"""The reply protocol of the paper's harness: plain text with at most one fenced block. No tool calling, so any chat
model can be an editor."""

from __future__ import annotations

import re
from dataclasses import dataclass

_OPEN = re.compile(r"^```(\w+)[ \t]*$", re.M)
_CLOSE = re.compile(r"^```[ \t]*$", re.M)


@dataclass
class Block:
    kind: str
    body: str


@dataclass
class Parsed:
    explanation: str
    blocks: list[Block]


def parse(text: str, kinds: set[str]) -> Parsed:
    """Split a reply into the text before its first block and its blocks. Fences of other kinds are prose."""
    blocks: list[Block] = []
    explanation = None
    pos = 0
    while True:
        m = _OPEN.search(text, pos)
        if m is None:
            break
        if m.group(1) not in kinds:
            close = _CLOSE.search(text, m.end())
            pos = close.end() if close else m.end()
            continue
        close = _CLOSE.search(text, m.end())
        end = close.start() if close else len(text)
        if explanation is None:
            explanation = text[:m.start()].strip()
        blocks.append(Block(m.group(1), text[m.end():end].strip("\n")))
        pos = close.end() if close else len(text)
    return Parsed(text.strip() if explanation is None else explanation, blocks)
