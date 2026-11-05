"""Generate with vLLM in a process of its own, so its GPU memory is freed when it ends.

    python -m wsplab.runners.vllm_worker request.json

The engine settings follow exp8_math/code/evaluate.py (MIT).
"""

from __future__ import annotations

import json
import sys


def main(path: str) -> None:
    from vllm import LLM, SamplingParams

    request = json.load(open(path))
    sampling = dict(request["sampling"])
    max_tokens = sampling["max_tokens"]
    llm = LLM(model=request["weights"], tokenizer=request["tokenizer"], dtype="bfloat16",
              gpu_memory_utilization=request.get("gpu_memory", 0.85), max_model_len=request.get("max_model_len") or max_tokens + 2048,
              seed=0, enforce_eager=request.get("enforce_eager", False), **request.get("engine", {}))
    params = [SamplingParams(n=request["n"], **{**sampling, "seed": seed}) for seed in request["seeds"]]
    outputs = llm.generate(request["prompts"], params)
    with open(request["out"], "w") as f:
        for o in outputs:
            f.write(json.dumps([{"text": x.text, "finish": x.finish_reason, "ids": list(x.token_ids)} for x in o.outputs]) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
