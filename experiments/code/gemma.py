"""Milestone 10: a real editor on Gemma 3 270M with 5 passes, with the paper's sampling and lenient grading. MATH-500 is cut to
200 problems and 2 samples to suit a laptop. The whole path is the default one: no runner given."""
import json, sys
import wsplab
from wsplab.editors import Metered
from wsplab.eval.lm_suite import math_suite
from wsplab.kernel import Limits
from wsplab.passes import ForwardLimits
import common

REV = "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
model, cap, folder = sys.argv[1], float(sys.argv[2]), sys.argv[3]
sampling = {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 8192}
suite = math_suite(sampling, samples=2, limit=200, lenient=True)
s = common.spending(cap)
r = wsplab.run("google/gemma-3-270m-it", revision=REV, name="Gemma 3 270M IT", eval=suite, editor=Metered(common.editor(model), s),
               note="This model really sucks at math", budget=5, max_runs=60, folder=folder,
               limits=Limits(seconds=600, memory_gb=12), pass_time="a few minutes",
               forward_limits=ForwardLimits(n_layers=18, full_attention=(5, 11, 17), gen_max=1024, max_tokens=8192, tf_max_tokens=8192))
print(json.dumps({"verdict": r.verdict, "baseline": r.baseline, "final": r.final, "reasons": r.reasons, "turns": r.state.turn,
                  "passes": r.state.passes, "runs": r.state.runs, "stop": r.state.stop_reason, "run_usd": round(s.usd, 4),
                  "tokens": s.tokens, "patch": str(r.patch)}, indent=1))
print(r.answer)
