"""Experiment 7: which of two language models is sycophantic? Read mode on two checkpoints, with the ```attention tool, which
runs both models on texts the editor chooses and returns only their attention.

The fine-tuned checkpoints are not in the release. Rebuild them with the authors' exp7_sycophancy/code/train.py, about 25 s per
model on one GPU; they will not match the published hashes bit for bit. Then pass two folders holding config.json,
model.safetensors, tokenizer.json and tokenizer_config.json.

    python exp7_sycophancy.py path/to/A path/to/B anthropic:<model>
"""

import json
import sys
from pathlib import Path

import numpy as np

from _common import exact_prompt
from wsplab.passes import PassFailed, PassInvalid, PassResult
from wsplab.upstream import fetch

MAX_TEXTS, MAX_TOKENS, BOS = 16, 128, "<bos>"


class AttentionTool:
    """Runs both checkpoints on each text and saves /work/attn/r<k>/t<i>_<X>.npy, and the tokens as tokens.json."""

    fence = "attention"

    def __init__(self, runner, work: Path, checkpoints: dict[str, Path], layers: int):
        self.runner, self.work, self.checkpoints, self.layers = runner, Path(work), checkpoints, layers

    def run(self, body: str, request_number: int) -> PassResult:
        try:
            texts = json.loads(body)
        except json.JSONDecodeError as e:
            raise PassInvalid(f"JSONDecodeError: {e}") from e
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_TEXTS or not all(isinstance(t, str) for t in texts):
            raise PassInvalid(f"give a JSON list of 1 to {MAX_TEXTS} texts")
        folder = self.work / "attn" / f"r{request_number}"
        folder.mkdir(parents=True, exist_ok=True)
        tokens = None
        request = {"attn_layers": list(range(self.layers))}
        for side, weights in self.checkpoints.items():
            for i, text in enumerate(texts):
                try:
                    out = self.runner.read(weights, [BOS + text], request)
                except Exception as e:  # noqa: BLE001 - a failed run must not cost a pass
                    raise PassFailed(f"{type(e).__name__}: {e}") from e
                n = int(out["lengths"][0])
                if n > MAX_TOKENS + 1:
                    raise PassInvalid(f"text {i} has {n - 1} tokens, more than {MAX_TOKENS}")
                stack = np.stack([out[f"attn_{layer}"][0, :, :n, :n] for layer in range(self.layers)]).astype(np.float16)
                np.save(folder / f"t{i}_{side}.npy", stack)
                if side == "A":
                    tokens = (tokens or []) + [out["input_ids"][0, :n].tolist()]
        (folder / "tokens.json").write_text(json.dumps(tokens))
        return PassResult("attention", f"/work/attn/r{request_number}/", {"tokens": tokens})


def config(a: str, b: str, runs: int = 20, runner=None, layers: int = 18) -> dict:
    checkpoints = {"A": Path(a), "B": Path(b)}

    def tools(work, base, store, edited):
        from wsplab.runners import LM

        return {"attention": AttentionTool(runner or LM(a), work, checkpoints, layers)}

    prelude = fetch("exp7_sycophancy/runs/attention_tool/sandbox/prelude.py").read_text()
    return dict(description="two language models", mode="read", budget=runs, max_runs=runs, tools=tools, prelude=prelude,
                binds={a: "/case/A", b: "/case/B"}, prompt=exact_prompt("exp7_sycophancy/runs/attention_tool/prompt.txt"))


if __name__ == "__main__":
    import wsplab

    result = wsplab.run(sys.argv[1], editor=sys.argv[3], **config(sys.argv[1], sys.argv[2]))
    print(result.answer)
