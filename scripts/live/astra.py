"""The main live run: the paper's editor on the paper's model, scaled to a laptop. Answers are capped at 8,192 tokens rather
than 32,768 and the evaluation is 200 MATH-500 problems, for the baseline and the result alike."""
import json, sys
import wsplab
from wsplab.editors import Metered
from wsplab.eval.lm_suite import math_suite
from wsplab.kernel import Limits
from wsplab.passes import ForwardLimits
import common

REV = "2fc06364715b967f1860aea9cf38778875588b17"
model, cap, folder, budget = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
resume = "--resume" in sys.argv
sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 8192}
suite = math_suite(sampling, samples=1, limit=200)
s = common.spending(cap)
r = wsplab.run("Qwen/Qwen3.5-0.8B", revision=REV, name="Qwen3.5-0.8B", thinking=True, eval=suite, resume=resume,
               editor=Metered(common.editor("openai:" + model, "high"), s), note="This model really sucks at math", budget=budget, max_runs=300,
               folder=folder, limits=Limits(seconds=900, memory_gb=16), pass_time="about ten minutes",
               forward_limits=ForwardLimits(n_layers=24, full_attention=(3, 7, 11, 15, 19, 23), gen_max=256, max_tokens=8192,
                                            tf_max_tokens=8192, tf_max_texts=256, tf_total_tokens=262144))
print(json.dumps({"verdict": r.verdict, "baseline": r.baseline, "final": r.final, "reasons": r.reasons, "turns": r.state.turn,
                  "passes": r.state.passes, "runs": r.state.runs, "stop": r.state.stop_reason, "run_usd": round(s.usd, 4),
                  "tokens": s.tokens, "patch": str(r.patch)}, indent=1))
print(r.answer)
