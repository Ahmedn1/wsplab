"""A language model with no model: deterministic answers and arrays of the right shapes, for tests and dry runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Generation, Prompt


class FakeLM:
    """Answers `\\boxed{<answer>}` when `correct(weights, prompt)` says so, otherwise loops up to the token limit."""

    def __init__(self, hidden: int = 8, mlp: int = 16, correct=None, answers: dict[str, str] | None = None):
        self.hidden = hidden
        self.mlp = mlp
        self.correct = correct or (lambda weights, prompt: False)
        self.answers = answers or {}
        self.calls: list[tuple[str, Path]] = []
        self.fail_with: Exception | None = None

    def render(self, user_message: str) -> str:
        return f"<user>{user_message}</user>"

    def generate(self, weights, prompts: list[Prompt], n: int, sampling: dict) -> list[list[Generation]]:
        self.calls.append(("generate", Path(weights)))
        if self.fail_with:
            raise self.fail_with
        out = []
        for p in prompts:
            if self.correct(weights, p):
                out.append([Generation(f"\\boxed{{{self.answers.get(p.id, '1')}}}", "stop", [1, 2, 3]) for _ in range(n)])
            else:
                out.append([Generation("Wait, " * 3, "length", [1] * sampling.get("max_tokens", 8)) for _ in range(n)])
        return out

    def read(self, weights, texts: list[str], request: dict) -> dict[str, np.ndarray]:
        self.calls.append(("read", Path(weights)))
        if self.fail_with:
            raise self.fail_with
        lengths = np.array([len(t.split()) + 1 for t in texts])
        n, t = len(texts), int(lengths.max())
        out: dict[str, np.ndarray] = {"lengths": lengths, "input_ids": np.zeros((n, t), np.int64)}
        if request.get("token_logprobs"):
            out["token_logprobs"] = -np.ones((n, t - 1), np.float32)
        if request.get("topk"):
            out["topk_logprobs"] = np.zeros((n, t, request["topk"]), np.float32)
            out["topk_ids"] = np.zeros((n, t, request["topk"]), np.int64)
        for layer in request.get("hidden_states", []):
            out[f"hidden_{layer}"] = np.zeros((n, t, self.hidden), np.float32)
        for layer in request.get("mlp_act", []):
            out[f"mlp_act_{layer}"] = np.zeros((n, t, self.mlp), np.float32)
        for layer in request.get("attn_layers", []):
            out[f"attn_{layer}"] = np.zeros((n, 2, t, t), np.float32)
        return out

    def activations(self, weights, prompts, answers, layers, mlp, max_tokens) -> dict[str, np.ndarray]:
        n, k = len(answers), len(layers)
        out = {"hidden_last_prompt": np.zeros((n, k, self.hidden), np.float32),
               "hidden_mean_answer": np.zeros((n, k, self.hidden), np.float32)}
        if mlp:
            out["mlp_last_prompt"] = np.zeros((n, k, self.mlp), np.float32)
            out["mlp_mean_answer"] = np.zeros((n, k, self.mlp), np.float32)
        return out
