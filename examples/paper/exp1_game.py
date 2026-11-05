"""Experiment 1: which game was this network trained on? A generic network, read mode, weights in the prompt.

    python exp1_game.py breakout anthropic:<model>
"""

import sys

import wsplab
from _common import exact_prompt, section
from wsplab.upstream import fetch

NETWORKS = {"breakout": "a9f535cc", "pong": "dfc6cb65", "pacman": "bdf99279", "freeway": "0bbcc3ea"}
KEYWORDS = {"breakout": ("breakout", "brick"), "pong": ("pong",), "pacman": ("pac-man", "pacman", "pac man"),
            "freeway": ("freeway", "frogger", "road", "traffic")}


def config(game: str) -> dict:
    return dict(model=fetch(f"exp1_which_game/networks/{NETWORKS[game]}.npz"), description="a policy network", mode="read",
                access="inline", budget=0, prompt=exact_prompt(f"exp1_which_game/runs/{game}/prompt.txt"))


def score(answer: str) -> str | None:
    """The answer's Game line, mapped to a game by keywords, as in the paper."""
    line = section(answer, "Game").lower()
    return next((g for g, words in KEYWORDS.items() if any(w in line for w in words)), None)


if __name__ == "__main__":
    result = wsplab.run(editor=sys.argv[2], **config(sys.argv[1]))
    print(result.answer, "\n\nscored as:", score(result.answer))
