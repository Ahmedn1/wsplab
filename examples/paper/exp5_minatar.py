"""Experiment 5: write a MinAtar policy from scratch, in one reply. Write mode, weights in the prompt.

MinAtar is GPL-3.0, so nothing from it is shipped here: install it yourself (`pip install minatar==1.0.15`) to score the
result. The prompt quotes MinAtar's source, so it is fetched from the authors' release at run time.

    python exp5_minatar.py breakout anthropic:<model>
"""

import sys

import numpy as np

from _common import exact_prompt

NAMES = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel", "b2": "actor/dense2/bias",
         "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}


def zeros() -> dict:
    import torch

    return {k: torch.zeros(s) for k, s in SHAPES.items()}


def config(game: str) -> dict:
    return dict(description="a policy network", mode="write", access="inline", budget=0, no_eval=True,
                prompt=exact_prompt(f"exp5_write_from_scratch/runs/{game}/message.txt"))


def policy(weights: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h = np.tanh(x @ weights["W1"] + weights["b1"])
    h = np.tanh(h @ weights["W2"] + weights["b2"])
    return h @ weights["W3"] + weights["b3"]


if __name__ == "__main__":
    import wsplab

    result = wsplab.run(zeros(), editor=sys.argv[2], **config(sys.argv[1]))
    print(result.verdict, result.patch)
