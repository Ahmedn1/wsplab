"""The whole language model path with real components and no LLM spend: real Qwen3.5-0.8B, vLLM, the sandboxed kernel, a real
forward pass, patch, evaluation and verdict. Only the editor is scripted: it applies the paper's published 25-pass edit.

    python scripts/smoke_lm.py <run folder>
"""

import json
import os
import sys
from pathlib import Path

import wsplab
from wsplab.editors import ScriptedEditor
from wsplab.eval import MathGrader
from wsplab.eval.graders import INSTRUCTION
from wsplab.eval.lm_suite import LMSuite, read_jsonl
from wsplab.kernel import Limits
from wsplab.passes import ForwardLimits
from wsplab.upstream import fetch

REV = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN = Path(os.environ["HF_HOME"]) / f"hub/models--Qwen--Qwen3.5-0.8B/snapshots/{REV}"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 1024}


def main(folder: str) -> None:
    rows = read_jsonl(fetch("exp8_math/practice/gsm8k_train.jsonl"))
    suite = LMSuite(rows[:32], rows[32:96], MathGrader(), SAMPLING, benchmark="64 GSM8K problems",
                    user_message=lambda row: f"{row['problem']}\n{INSTRUCTION}")
    edit = ("name = 'model.language_model.layers.15.self_attn.o_proj.weight'\n"
            "w = load(name); w[:, 1792:2048] = 0\nprint(save_edit(name, w))")
    request = json.dumps({"type": "generate", "model": "both", "problems": [r["id"] for r in rows[:16]], "n": 1})
    read = json.dumps({"type": "teacher_forced", "model": "edited", "texts": ["2 + 2 = 4"],
                       "return": {"token_logprobs": True, "hidden_states": [15, 24], "mlp_act": [15], "attn_layers": [15]}})
    replies = ["Listing tensors.\n```python\nprint(len(NAMES), NAMES[0])\nprint(load(NAMES[5]).shape)\n```",
               f"Switching off head 7 of layer 15, the paper's edit.\n```python\n{edit}\n```",
               f"Measuring it.\n```forward\n{request}\n```",
               f"Reading activations.\n```forward\n{read}\n```",
               "I switched off one attention head."]
    result = wsplab.run(str(QWEN), eval=suite, editor=ScriptedEditor(replies), budget=2, folder=folder, name="Qwen3.5-0.8B",
                        note="This model really sucks at math", limits=Limits(seconds=120, memory_gb=8), control=True,
                        forward_limits=ForwardLimits(n_layers=24, full_attention=(3, 7, 11, 15, 19, 23), gen_max=64, max_tokens=1024,
                                                     tf_max_tokens=1024))
    print(json.dumps({"verdict": result.verdict, "reasons": result.reasons, "baseline": result.baseline, "final": result.final,
                      "passes": result.state.passes, "runs": result.state.runs, "patch": str(result.patch)}, indent=1))
    print(result.generate("What is 17 * 23? Answer with the number only.", max_tokens=32))


if __name__ == "__main__":
    main(sys.argv[1])
