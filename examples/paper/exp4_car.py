"""Experiment 4: fix a failing car network in one reply. Edit mode, weights in the prompt.

Scoring needs the Kinetix environment (`pip install kinetix-env`), which `score` imports. The network shown to the editor
is rounded, and edits apply to the rounded values, so `shown_network` rebuilds it from the prompt.

    python exp4_car.py anthropic:<model>
"""

import re
import sys

import numpy as np

from _common import exact_prompt
from wsplab.upstream import fetch

PROMPT = "exp4_fix_car/runs/oneshot_repair/reply/prompt.txt"
_HEADER = re.compile(r"^([\w.]+) \(([\d x]+)\), scale ([^:\s]+):")


def shown_network(prompt: str) -> dict[str, np.ndarray]:
    """The weights exactly as the prompt shows them: integers times one scale per tensor."""
    tensors: dict[str, np.ndarray] = {}
    lines = prompt.splitlines()
    i = 0
    while i < len(lines):
        m = _HEADER.match(lines[i].strip())
        i += 1
        if m is None:
            continue
        name, shape, scale = m.group(1), [int(x) for x in m.group(2).split("x")], float(m.group(3))
        rows: list[list[int]] = []
        while i < len(lines) and lines[i].strip() and not _HEADER.match(lines[i].strip()):
            line = lines[i].split(":", 1)[-1]
            rows.append([int(x) for x in line.split()])
            i += 1
        tensors[name] = np.array(rows, dtype=np.float64).reshape(shape) * scale
    return tensors


def config() -> dict:
    return dict(description="a car policy", access="inline", budget=0, prompt=exact_prompt(PROMPT))


def network_state() -> dict:
    import torch

    prompt = fetch(PROMPT).read_text()
    return {k: torch.from_numpy(v.astype(np.float32)) for k, v in shown_network(prompt).items()}


if __name__ == "__main__":
    import wsplab

    result = wsplab.run(network_state(), editor=sys.argv[1], no_eval=True, **config())
    print(result.verdict, result.patch)
