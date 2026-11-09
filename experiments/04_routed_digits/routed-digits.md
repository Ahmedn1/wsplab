# Routed digit repair benchmark

Continuation of the final Claude Code decision: build a task with observable partial
progress before buying another council comparison. No paid editor calls were made
for this calibration. Councils remain experimental.

## Why the proposed task was rejected

The recovered `experiments/code/hard2_net.py` prototype used ten branches followed by
an argmax over their logits. Separate weights did not make accuracy independent:
a damaged branch could dominate predictions for every digit. Claude's final tool
output, before the conversation stopped, already showed the failure:

| Repair digit | Gain in percentage points |
|---|---:|
| 0 | 23.7 |
| 1 | 0.0 |
| 2 | 0.0 |
| 3 | 2.8 |
| 4 | 0.2 |
| 5 | 0.0 |
| 6 | 0.0 |
| 7 | 0.0 |

The original clean score was 94.16%, with a damaged score of 3.82%. The recovered
prototype and check script are kept for provenance, not as the accepted benchmark.

## Replacement and calibration

`experiments/code/modular_net.py` trains ten binary digit detectors on CPU solely to
construct the fixture. Editors may not train or access those clean weights.
An input has 64 image pixels plus a one-hot digit query. Only the queried branch's
two-logit difference determines its binary prediction. A repair to one branch therefore cannot
alter any other branch's queried outputs.

Images are split into training (900), calibration (250), practice (300), and
evaluation (347) before making queries. Each image is queried at most once, with
equal positives and negatives and equal query counts per digit. This also keeps
`search`'s practice/selection split free of repeated images under different queries.
The normal search splits practice in half for editor access and candidate selection;
only the selected candidate reaches final evaluation.

Eight branches are damaged; two are healthy regression controls. Faults include
an output-row swap, suppressed hidden biases, an output offset, hidden sign
flips, rewiring, mirrored inputs, shifted inputs, and a compound rewiring/sign fault.
"Repair" here means restoring a whole damaged branch. It does not mean that every
elementary correction inside the compound fault must independently help.

The first routed version also failed calibration: attenuating an output weight
did not change classifications, and alternating hidden bias shifts sometimes
helped. Those faults were replaced with a fixed output offset and hidden bias
suppression. The rejected results are preserved in
[routed-digits-rejected-faults.json](routed-digits-rejected-faults.json).
This is benchmark development on calibration data, not a preregistered LLM result.

Integration also rejected the initial scalar-output design: reversing a whole
output tensor violates the framework's normal rescaling restriction, and a scalar
bias cannot be freely edited under that rule. Each branch now has two output rows;
swapping them is a legal repair. Calibration verifies every oracle tensor repair
through `WeightStore` with its normal restrictions enabled. Alternating hidden-unit
flips were also too weak after the architecture change, so the final fault flips
the seven strongest output contributors, leaving one unchanged. The rejected
two-logit calibration is preserved in
[routed-digits-rejected-weak-flips.json](routed-digits-rejected-weak-flips.json).

Acceptance required clean accuracy of at least 90%, each complete branch repair
adding at least 0.5 percentage points, and exactly unchanged outputs for other
routes. All three construction seeds pass:

| Seed | Damaged | Clean | Smallest isolated repair gain |
|---|---:|---:|---:|
| 21 | 53.33% | 95.83% | 2.08 pp |
| 22 | 47.92% | 97.08% | 2.92 pp |
| 23 | 52.08% | 96.25% | 2.92 pp |

There are 240 calibration queries per seed. These are observed finite-sample gains,
not claims that every individual gain is statistically significant. Single-repair
models were evaluated directly; all 256 complete-branch repair subsets were checked
by composing their independent logits. Scores were additive to numerical precision.
Full results and clean-weight hashes are in
[routed-digits-calibration.json](routed-digits-calibration.json).

Reproduce with the project's CPU environment:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/wsplab/bin/python \
  experiments/code/modular_check.py --out .wsplab/calibration/recheck.json
```

`modular_job.run_benchmark` accepts editor objects and an explicit `Spending`
instance. It records the seed, prices, budgets and members before calls begin.
`shared=True` enables peers; `shared=False` uses the same fixture, selection and
evaluation process without sharing. Its CLI runs only a scripted smoke test with
a known repair; that tests integration, not model problem-solving:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/wsplab/bin/python \
  experiments/code/modular_job.py --out .wsplab/smoke/new-run
```

The scripted two-peer smoke test applied the known output-row swap and improved
held-out accuracy from 53.53% to 63.53%, with a `confirmed` gate verdict. Both
candidates contained real edits. This is a plumbing check with an oracle-provided
repair, not a council result.

## Comparison design and remaining evidence

The original comparison proposal was to freeze this task before paid runs and compare the peer trio with the same three
members searching independently, and with three independent attempts by the best
solo model. Use the same construction seeds, practice/selection split, per-session
Python/forward limits, selection method, and total spending allowance. Record
actual cost, turns, passes, missing edits and provider errors alongside scores.
Do not call a lower score a reasoning failure when the provider failed or the run
was cut off; still include that failure in the operational outcome.

The independent trio is essential: it separates the value of sharing from the
value of having three candidates and selecting the best. Choose the strongest
solo model using prior development results, not this comparison's evaluation.
Fix the run count and analysis before starting; do not repeat until a council wins.
Use paired differences across runs/construction seeds, not a new arbitrary
"2 of 5" success threshold. The old 70% threshold belongs to a different task.

This routed, decomposable task deliberately makes partial progress observable.
Even a council advantage here would establish a limited result on modular repair,
not general superiority on complex tasks. A second task with genuine interactions
would still be needed.

Live testing has now produced one positive two-provider case: on seed 22 the
selected peer repair scored 88.82%, versus 83.53% for the selected independent
candidate, on 340 held-out queries. All four sessions finished normally. The
peers cost more, and repeated strongest-solo attempts have not been tested.
See the [confirmation results and audit](peer-confirmation-results.md), including
why an earlier seed-21 pilot was disqualified. This is limited evidence for a
peer advantage on this fixture, not completion of the broader comparison above.

## Astra monitoring handoff

The existing run is `~/.wsplab/live/astra50`, PID 3683356 at handoff.
`experiments/code/monitor_run.py` is a separate read-only watcher: every 60 seconds it
writes `.wsplab/monitor/astra50/status.json` and appends `history.jsonl`. It watches
for up to 48 hours, then stops; it also stops on a final report or process exit.
It never restarts the experiment, changes its budgets, or makes API calls.
The watcher PID is recorded in `watcher.pid`.

A saved editor state with `done=true` is not completion: final evaluation can still
be running. Completion requires `report.json`, whose verdict is preserved even if
negative. Missing activity for an hour means "needs_inspection", not a proven hang.
The watcher checks process start identity to avoid mistaking a reused PID for Astra.
Its output is a local log, not an automatic chat notification.
