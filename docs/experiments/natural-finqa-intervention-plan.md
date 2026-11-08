# FinQA follow-up: require causal intervention tests

The user authorized this follow-up with "ok go ahead" after the previous council
and solo run returned unchanged weights without testing any edits. This is a
separate exploratory experiment, not a replacement for that negative result or
an independent replication. The original FinQA prompt, model, numeric grader,
sampling settings, report partitions and external-test questions remain fixed.

## Required experiment

OpenAI, Claude and Gemini each test exactly **two** predeclared edited checkpoints.
Claude solo tests **six**. Both arms therefore test six candidates in total.
Each editor writes a frozen list identifying the tensor names, operation,
strength and rationale before its first edited forward pass. Two strengths of
one mechanistic hypothesis can count as two candidates, provided their actual
BF16 weights differ. Each candidate starts from the original checkpoint.

The harness requires paired original-versus-edited generation on all 32 DEV
questions, with one sample per question and unchanged full-context prompts.
It verifies the materialized patch is nonzero after dtype rounding and rejects
duplicate checkpoints, changed proposal lists and undeclared tensor names.
Diagnostics are limited to original-model DEV generation: one pass per peer,
three for solo. No synthetic or teacher-forced diagnostic screens are used here.
Saving the first edit by Python run 6 is an advisory target; completing all
required candidate tests is mandatory. Candidate backups must be saved so the
editor can restore the exact winner.

Choose the highest positive DEV gain, breaking ties by fewer changed entries
and a deterministic candidate ID. If no gain is positive, restore the original.
Otherwise restore and freeze that winner and run exactly one paired comparison
on the 32 internal HOLDOUT questions. Retain the exact edit only for a positive
HOLDOUT gain; otherwise restore the original. No further inference or tuning
is allowed after confirmation. This internal rule is exploratory, not a claim
of statistical significance.

The editor cannot successfully finish with an untested no-change result. The
session receives progress notices and rejects premature final answers; an
incomplete experiment at a budget or turn limit is recorded as a failure, not
`no_fix_found`. A valid no-change result remains possible after real tests.

## Fairness and isolation

Peers retain independent weights and share DEV findings. Messages from a peer
entering HOLDOUT confirmation are withheld thereafter. Solo receives no live
peer findings. All editors receive the same brief history: the earlier local
neuron edit did not improve DEV, and the previous council/solo run performed
diagnostics but no interventions. No prior editor transcripts, private selection
scores or patches are given to the new editors.

Each peer retains ceilings of 6 passes, 20 Python runs and 40 turns; solo has
18 passes, 60 runs and 120 turns. Actual consumption may differ. The council
submits three final candidates to private selection, while solo submits one,
so this is not an ablation isolating communication from portfolio diversity.
The retained original model remains eligible in private selection.

All four sessions require an audit before any use of the still-unused
95-question external set. The existing paired tests, random-edit controls and
joint win criterion remain unchanged. If both arms return the identical original
checkpoint, they can be closed as no fix found by weight identity, without
consuming external questions or claiming an external score.

## Spending and launch

Prior FinQA attempts spent $5.14273275. Their three immutable ledgers are
referenced by SHA256 in the new `budget-history.json`. Remaining ceilings are:

| Arm | OpenAI | Claude | Gemini | Total |
|---|---:|---:|---:|---:|
| Council | $1.44584635 | $4.13779840 | $1.70395270 | $7.28759745 |
| Solo | — | $7.28759745 | — | $7.28759745 |

Maximum new spending is $14.57519490; including all previous FinQA attempts,
the maximum is $19.71792765. This remains within the existing $24 authorization
and cumulative provider limits of OpenAI $22, Anthropic $47, Gemini $30.
There is no new spending-limit increase.

Artifacts: `.wsplab/finqa-intervention-comparison/`. Scratch checkpoints:
`/tmp/wsplab-finqa-intervention-comparison/`. The separate trial/guard/watch
scripts preserve the earlier experiment's frozen source files and artifacts.
Tokenizer warmup, startup synchronization, GPU serialization, spending guards
and failure cancellation are retained from the repaired runner.

Validation before launch: 31 focused tests passed, including real BF16 patch
rounding, duplicate edits, frozen proposal lists, premature stopping, positive
and negative HOLDOUT outcomes, startup cancellation and historical spending.
Host GPU and sandbox preflight passed. Live provider-backed completion remains
to be demonstrated by this experiment.
