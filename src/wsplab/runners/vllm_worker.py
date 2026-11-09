"""Generate with vLLM in a process of its own, so its GPU memory is freed when it ends.

    python -m wsplab.runners.vllm_worker request.json

If the request names a `progress` file, the worker keeps it updated with {"done", "total", "out_tokens_per_s"} as answers
finish, for the parent to report. The engine settings follow exp8_math/code/evaluate.py (MIT).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time


def progress_bar(path: str, total: int, every: float = 2.0):
    """A tqdm class for vLLM's `use_tqdm`, which writes progress to `path` instead of drawing a bar."""
    from tqdm import tqdm

    speed = re.compile(r"output: ([0-9.]+) toks/s")

    class FileProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = open(os.devnull, "w")  # noqa: SIM115 - closed with the bar
            super().__init__(*args, **kwargs)
            self._written = 0.0
            self._write(force=True)

        def _write(self, force: bool = False) -> None:
            now = time.monotonic()
            if not force and now - self._written < every:
                return
            self._written = now
            found = speed.search(str(self.postfix or ""))
            row = {"done": min(int(self.n), total), "total": total, "out_tokens_per_s": float(found[1]) if found else None}
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump(row, f)
            os.replace(tmp, path)

        def update(self, n=1):
            shown = super().update(n)
            self._write()
            return shown

        def close(self):
            if not self.disable:
                self._write(force=True)
            super().close()
            self.fp.close()

    return FileProgress


def main(path: str) -> None:
    from vllm import LLM, SamplingParams

    request = json.load(open(path))
    sampling = dict(request["sampling"])
    max_tokens = sampling["max_tokens"]
    llm = LLM(model=request["weights"], tokenizer=request["tokenizer"], dtype="bfloat16",
              gpu_memory_utilization=request.get("gpu_memory", 0.85), max_model_len=request.get("max_model_len") or max_tokens + 2048,
              seed=0, enforce_eager=request.get("enforce_eager", False), **request.get("engine", {}))
    params = [SamplingParams(n=request["n"], **{**sampling, "seed": seed}) for seed in request["seeds"]]
    progress = request.get("progress")
    bar = progress_bar(progress, len(request["prompts"]) * request["n"]) if progress else False
    outputs = llm.generate(request["prompts"], params, use_tqdm=bar)
    with open(request["out"], "w") as f:
        for o in outputs:
            f.write(json.dumps([{"text": x.text, "finish": x.finish_reason, "ids": list(x.token_ids)} for x in o.outputs]) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
