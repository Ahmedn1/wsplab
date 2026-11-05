"""An editor is whatever writes the replies: an LLM, a person, a script, or a council of LLMs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class EditorRefused(Exception):
    """The provider refused the last message, for example through a content filter."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_tokens += other.cached_tokens


@dataclass
class Reply:
    """`truncated` means the reply hit the output limit, so any block in it is incomplete and must not run."""

    text: str
    usage: Usage = field(default_factory=Usage)
    truncated: bool = False


@runtime_checkable
class Editor(Protocol):
    name: str

    def reply(self, system: str, messages: list[dict]) -> Reply:
        """The next reply. `messages` is the conversation so far, ending with the user's message."""
        ...


class ScriptedEditor:
    """Replays fixed replies: for tests, and for running the whole pipeline with no LLM."""

    def __init__(self, replies: list[str], name: str = "scripted"):
        self.replies = list(replies)
        self.name = name
        self.seen: list[list[dict]] = []

    def reply(self, system: str, messages: list[dict]) -> Reply:
        self.seen.append(list(messages))
        if not self.replies:
            raise RuntimeError("the script has no more replies")
        return Reply(self.replies.pop(0))
