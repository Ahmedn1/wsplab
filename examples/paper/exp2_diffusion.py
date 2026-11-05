"""Experiment 2: what does this diffusion model generate? The same path as experiment 1: read mode, weights in the prompt.
A diffusion model needs no type of its own, because reading it needs nothing a game policy does not.

    python exp2_diffusion.py circle anthropic:<model>
"""

import sys

import wsplab
from _common import exact_prompt, section
from wsplab.upstream import fetch

NETWORKS = {"circle": "d88aba69", "square": "a6a9dfb7", "triangle": "246a5b58"}
KEYWORDS = {"circle": ("circle", "disc", "round", "oval"), "square": ("square", "rectangle", "box", "rectangular"),
            "triangle": ("triangle", "pyramid", "wedge")}


def config(shape: str) -> dict:
    return dict(model=fetch(f"exp2_diffusion/networks/{NETWORKS[shape]}.npz"), description="a diffusion model", mode="read",
                access="inline", budget=0, prompt=exact_prompt(f"exp2_diffusion/runs/{shape}/prompt.txt"))


def score(answer: str) -> str | None:
    line = section(answer, "Images").lower()
    return next((s for s, words in KEYWORDS.items() if any(w in line for w in words)), None)


if __name__ == "__main__":
    result = wsplab.run(editor=sys.argv[2], **config(sys.argv[1]))
    print(result.answer, "\n\nscored as:", score(result.answer))
