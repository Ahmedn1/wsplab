"""Experiment 3: which language did this model learn, from its middle blocks only? Read mode through the kernel: the weights
do not fit in a prompt, so the editor writes its own scripts, with no forward passes.

The paper ran each script in a fresh process; wsplab's kernel keeps variables between runs, which only makes this easier.

    python exp3_language.py python anthropic:<model>
"""

import re
import sys

import wsplab
from _common import exact_prompt
from wsplab.kernel import Limits
from wsplab.upstream import fetch


def config(language: str, conversation: int = 0) -> dict:
    return dict(model=fetch(f"exp3_language/networks/{language}/blocks.npz"), description="a small language model", mode="read",
                budget=0, max_runs=100, prompt=exact_prompt(f"exp3_language/runs/{language}_r{conversation}/message.txt"),
                limits=Limits(seconds=600, output_chars=20_000), prelude="W = {n: load(n).numpy() for n in NAMES}\n")


def score(answer: str) -> str | None:
    found = re.findall(r"^\W*corpus\W*:\W*(python|romaji|english)", answer, re.M | re.I)
    return found[-1].lower() if found else None


if __name__ == "__main__":
    result = wsplab.run(editor=sys.argv[2], **config(sys.argv[1]))
    print(result.answer, "\n\nscored as:", score(result.answer))
