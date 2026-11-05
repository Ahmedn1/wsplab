"""Experiment 8: make a language model better at math. The whole job in one call.

    python exp8_math.py anthropic:<model> 25

A full evaluation is MATH-500 with 4 samples of up to 32,768 tokens, which needs a large GPU: pass a smaller suite to try
it on a laptop, as `small_suite` does.
"""

import sys

import wsplab
from wsplab.eval.lm_suite import math_suite

MODEL = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def small_suite():
    sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192}
    return math_suite(sampling, samples=1, limit=100)


if __name__ == "__main__":
    result = wsplab.run_lm(MODEL, revision=REVISION, thinking=True, eval=small_suite(), editor=sys.argv[1],
                           budget=int(sys.argv[2]), note="This model really sucks at math")
    print(result.verdict, result.reasons, result.patch)
    print(result.generate("What is 17 * 23?"))
