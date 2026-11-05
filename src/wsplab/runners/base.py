"""Runners execute the network. One interface serves the editor's forward passes, the evaluation and user inference, so
a score and a pass can never disagree about how the network runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class Generation:
    text: str
    finish_reason: str
    token_ids: list[int] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return len(self.token_ids)


@dataclass
class Prompt:
    """One item to answer: its rendered text and token ids, and the sampling seed fixed by the item."""

    id: str
    text: str
    seed: int


class LMRunner(Protocol):
    """A language model. `weights` is a checkpoint folder: the base model or a built edited model."""

    def render(self, user_message: str) -> str:
        """The evaluation's exact prompt for a user message, chat template included."""
        ...

    def generate(self, weights: Path, prompts: list[Prompt], n: int, sampling: dict) -> list[list[Generation]]:
        ...

    def read(self, weights: Path, texts: list[str], request: dict) -> dict[str, np.ndarray]:
        """Teacher forcing: the arrays asked for in `request`, right-padded to the longest text."""
        ...

    def activations(self, weights: Path, prompts: list[Prompt], answers: list[Generation], layers: list[int], mlp: bool,
                    max_tokens: int) -> dict[str, np.ndarray]:
        """Residual stream (and MLP inputs) at the last prompt token and averaged over the answer's first tokens."""
        ...
