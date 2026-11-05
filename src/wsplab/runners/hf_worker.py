"""Read activations in a process of its own, so its GPU memory is freed when it ends.

    python -m wsplab.runners.hf_worker request.pkl

The request is a file wsplab wrote a moment ago for itself, not outside input.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np


def main(path: str) -> None:
    from .lm import LM

    with open(path, "rb") as f:
        request = pickle.load(f)
    options = dict(request["options"])
    lm = LM(options.pop("tokenizer"), generator="hf", **options)
    arrays = getattr(lm, request["method"])(*request["args"])
    np.savez(request["out"], **arrays)


if __name__ == "__main__":
    main(sys.argv[1])
