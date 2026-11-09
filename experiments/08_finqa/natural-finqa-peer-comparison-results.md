# FinQA council versus Claude solo: no fix found

The corrected run completed all four editor sessions in about 67 minutes.
OpenAI, Claude and Gemini participated in the council, and Claude completed the
independent solo arm. Every session returned `no_fix_found`. All four submitted
patches are empty, and their resulting safetensors files hash identically to
the original Qwen3-0.6B checkpoint.

**No editor attempted a weight change.** This is a failure to produce a candidate,
not a measured failure of candidate weight edits, and not evidence that no useful
edit exists. There is no council advantage in the returned models: both selected
arms are the unchanged original model.

| Session | Forward passes used / allowed | Python runs | Changed weights | Estimated spend |
|---|---:|---:|---:|---:|
| Council: OpenAI | 2 / 6 | 9 | 0 | $0.2445 |
| Council: Claude | 2 / 6 | 7 | 0 | $0.8622 |
| Council: Gemini | 1 / 6 | 19 | 0 | $1.2838 |
| Solo: Claude | 2 / 18 | 10 | 0 | $0.4303 |

The corrected attempt cost $2.82085510. Including the archived tokenizer-failure
attempt ($2.32187765), total estimated provider spending is **$5.14273275**.
All reservations are cleared. No new paid run is started by this report.

## What the editors did

The shared original-model DEV baseline scored 23/64 responses across 32 questions
with two samples each (35.94%). These are repeated samples, not 64 independent
questions or independent replications by each council member. All recorded
failures were wrong answers, with no truncation or missing-answer failures.

The editors examined arithmetic, table selection, formula choice and unit-scale
errors. OpenAI and Claude used official teacher-forced probes on synthetic scale
and division examples to investigate possible mechanisms. These probes were
diagnostic only; they did not replace the genuine FinQA task or its frozen grader.
Gemini prepared additional DEV probes but did not execute another forward pass.
Solo Claude collected four samples on 16 DEV questions and compared recorded
activations within questions.

They found no sufficiently convincing activation correlate and chose to stop.
Their statements that the failures cannot be repaired, or that a mechanism does
not exist, are not established by these small observational analyses. In
particular, a failed correlation screen does not rule out a useful intervention.

## Audit and closure

Review covered the executed Python, official forward requests, notes and final
submissions, supplemented by source/data checks, patch validation and model-file
hashes. No training, unauthorized model replay, reference changes, or access to
private selection/external data was found. Internal HOLDOUT was not used for
analysis or inference. All three council providers made paid calls and shared
DEV findings; solo received no council findings. No `save_edit` call occurred.

The private 32-question selection baseline was 16/32 (50%). Each arm retained
that original model. This is not an external-test accuracy claim.

**The 95-question external set remains unused.** As an explicit departure from
the planned external inference stage, the experiment was closed by verified
weight identity: evaluating identical checkpoints cannot distinguish the arms.
No external accuracy, p-value or random-edit comparison is reported.

Machine-readable evidence is in `.wsplab/finqa-peer-comparison/`: `outcome.json`,
`protocol-audit.json`, `selections-frozen.json`, the four submitted patches,
per-session logs, and both spending ledgers. `monitor.json` and `status.json`
now read `completed_no_fix_found`.

## Implication for another experiment

The search stopped at diagnosis. Before spending on another run, the experimental
procedure should require a small predeclared set of reversible interventions to
be tested on DEV, with a pass budget reserved for that purpose. It should still
allow the original model to win final selection. Requiring causal evidence before
performing any intervention creates a circular stopping rule. This run does not
justify abandoning weight editing or claiming a council win.
