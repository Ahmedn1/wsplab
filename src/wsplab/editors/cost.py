"""A shared spending limit. Councils and searches multiply LLM calls, so they cannot run without one."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .base import Reply, Usage


class BudgetExceeded(Exception):
    pass


@dataclass
class Spending:
    """Stops once `max_tokens` total tokens or `max_usd` is reached. `prices` maps an editor name to dollars per million
    (input, output) tokens; a limit in dollars needs a price for every editor, so a typo cannot disable it."""

    max_tokens: int | None = None
    max_usd: float | None = None
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    tokens: int = 0
    usd: float = 0.0

    def __post_init__(self):
        if self.max_tokens is None and self.max_usd is None:
            raise ValueError("set max_tokens or max_usd")
        self._lock = threading.Lock()

    def add(self, editor: str, usage: Usage) -> None:
        with self._lock:
            self.tokens += usage.input_tokens + usage.output_tokens
            if self.max_usd is not None:
                if editor not in self.prices:
                    raise ValueError(f"max_usd needs a price for {editor!r}: pass prices={{{editor!r}: (input, output)}}")
                price_in, price_out = self.prices[editor]
                self.usd += (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1e6

    def check(self) -> None:
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"token limit reached: {self.tokens:,} of {self.max_tokens:,}")
        if self.max_usd is not None and self.usd >= self.max_usd:
            raise BudgetExceeded(f"spending limit reached: ${self.usd:.2f} of ${self.max_usd:.2f}")


class Metered:
    """An editor whose calls count against a Spending limit."""

    def __init__(self, editor, spending: Spending):
        self.editor = editor
        self.spending = spending
        self.name = editor.name

    def reply(self, system: str, messages: list[dict]) -> Reply:
        self.spending.check()
        reply = self.editor.reply(system, messages)
        self.spending.add(self.name, reply.usage)
        return reply
