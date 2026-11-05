"""A suite scores a network: per-item scores, higher is better. The same object scores practice passes, the baseline and
the final evaluation, so every comparison is like for like. An item's score is a list of samples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from collections.abc import Callable

import numpy as np


@dataclass
class Scores:
    per_item: dict[str, list[float]]
    extra: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        values = [sum(v) / len(v) for v in self.per_item.values()]
        return sum(values) / len(values) if values else 0.0


class Scorer(Protocol):
    """Scores a callable network on items. `call(inputs)` runs the network on a batch."""

    def ids(self) -> list[str]: ...
    def score(self, call: Callable, ids: list[str]) -> Scores: ...
    def describe(self) -> str: ...
    def export(self, folder: Path) -> None: ...


def _accuracy(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return (outputs.argmax(-1) == targets).astype(np.float64)


def _neg_mse(outputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return -((outputs - targets) ** 2).reshape(len(outputs), -1).mean(1)


METRICS = {"accuracy": _accuracy, "neg_mse": _neg_mse}


class ArrayScorer:
    """Labelled data: inputs and targets as arrays. `metric` is "accuracy", "neg_mse", or f(outputs, targets) returning
    one score per item, where higher is better."""

    def __init__(self, inputs, targets, metric="accuracy", name: str = "data", batch: int = 1024, prefix: str = "item"):
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(targets)
        if len(self.inputs) != len(self.targets):
            raise ValueError("inputs and targets differ in length")
        self.metric = METRICS[metric] if isinstance(metric, str) else metric
        self.metric_name = metric if isinstance(metric, str) else getattr(metric, "__name__", "custom")
        self.name = name
        self.batch = batch
        self.prefix = prefix

    def ids(self) -> list[str]:
        return [f"{self.prefix}_{i}" for i in range(len(self.inputs))]

    def _index(self, ids: list[str]) -> np.ndarray:
        return np.array([int(i.rsplit("_", 1)[1]) for i in ids], dtype=np.int64)

    def outputs(self, call: Callable, ids: list[str]) -> np.ndarray:
        idx = self._index(ids)
        parts = [np.asarray(call(self.inputs[idx[i:i + self.batch]])) for i in range(0, len(idx), self.batch)]
        return np.concatenate(parts)

    def score(self, call: Callable, ids: list[str]) -> Scores:
        out = self.outputs(call, ids)
        values = self.metric(out, self.targets[self._index(ids)])
        return Scores({i: [float(v)] for i, v in zip(ids, values)}, {"metric": self.metric_name})

    def describe(self) -> str:
        return (f"Practice data (none of it is in the evaluation): /practice/inputs.npy {list(self.inputs.shape)} and "
                f"/practice/targets.npy {list(self.targets.shape)}; item i has id {self.prefix}_i; the score is "
                f"{self.metric_name}.")

    def export(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "inputs.npy", self.inputs)
        np.save(folder / "targets.npy", self.targets)

    def fingerprints(self) -> set[bytes]:
        return {np.ascontiguousarray(x).tobytes() for x in self.inputs}

    def split(self, fraction: float, seed: int = 0) -> tuple[ArrayScorer, ArrayScorer]:
        """Two disjoint scorers: (first with `fraction` of the items, the rest)."""
        order = np.random.default_rng(seed).permutation(len(self.inputs))
        cut = max(1, min(len(order) - 1, int(len(order) * fraction)))
        a, b = order[:cut], order[cut:]
        make = lambda i, p: ArrayScorer(self.inputs[i], self.targets[i], self.metric, self.name, self.batch, p)  # noqa: E731
        return make(a, self.prefix), make(b, "eval")


class EpisodeScorer:
    """Policies: one item per episode, with the episode's seed fixed by the item, so two networks meet the same episodes.
    `env` builds a Gymnasium-style environment, or give a Gymnasium id. `act` turns network outputs into an action."""

    def __init__(self, env, seeds, act: Callable | None = None, name: str = "episodes", max_steps: int = 10_000):
        self.env = env
        self.seeds = list(seeds)
        self.act = act or (lambda out: int(np.asarray(out).argmax()))
        self.name = name
        self.max_steps = max_steps

    def _make(self):
        if callable(self.env):
            return self.env()
        import gymnasium

        return gymnasium.make(self.env)

    def ids(self) -> list[str]:
        return [f"episode_{s}" for s in self.seeds]

    def score(self, call: Callable, ids: list[str]) -> Scores:
        env = self._make()
        out: dict[str, list[float]] = {}
        for item in ids:
            obs, _ = env.reset(seed=int(item.rsplit("_", 1)[1]))
            total = 0.0
            for _ in range(self.max_steps):
                action = self.act(np.asarray(call(np.asarray(obs, dtype=np.float32)[None]))[0])
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            out[item] = [total]
        return Scores(out, {"metric": "episode_return"})

    def describe(self) -> str:
        return (f"Practice episodes (none is in the evaluation): {len(self.seeds)} episodes of {self.name}, ids "
                "episode_<seed> listed in /practice/seeds.json; the score of an episode is its total reward.")

    def export(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "seeds.json").write_text(json.dumps(self.seeds))

    def fingerprints(self) -> set[bytes]:
        return {str(s).encode() for s in self.seeds}


def check_disjoint(practice, evaluation) -> None:
    """Refuse a practice set that overlaps the evaluation: an editor must never see what it is measured on."""
    a, b = getattr(practice, "fingerprints", set), getattr(evaluation, "fingerprints", set)
    overlap = (a() if callable(a) else set()) & (b() if callable(b) else set())
    if overlap:
        raise ValueError(f"{len(overlap)} items are in both the practice data and the evaluation")
