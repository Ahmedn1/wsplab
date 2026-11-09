# FinQA direct weight-edit pilot: no retained improvement

The bounded local editing attempt completed with framework verdict
**`no_fix_found`**. Actual weight edits were tested with the original prompt,
context, thinking mode, sampling and scorer fixed. Neither of the two
precommitted strengths met the development retention rule, so the final
submission restores the original weights.

This is a negative result for one neuron intervention, not a test of every
weight-edit method and not a peer-council comparison. No paid provider calls,
training, manual model replay or external evaluation were used.

## Measured development results

Same 16 development questions and item seeds throughout; paired original
responses were reused from the official forward-tool cache.

| Weights | Correct / 16 | Accuracy | Fixes | Regressions | Gain vs original | Paired SE |
|---|---:|---:|---:|---:|---:|---:|
| Original Qwen3-0.6B | 7 | 43.75% | — | — | — | — |
| Zero selected output column | 6 | 37.50% | 1 | 2 | -6.25 points | 11.06 points |
| Halve selected output column | 7 | 43.75% | 2 | 2 | 0 points | 12.91 points |

Exact two-sided paired binomial p = 1 for each intervention. These small-sample,
single-seed exploratory results do not establish equivalence or absence of an
effect. No generated answer was truncated. The original and ablated models
averaged 1,183 and 1,480 output tokens respectively; shortening reasoning was
neither the metric nor an observed benefit of ablation.

The predeclared retention rule required DEV gain greater than one paired
standard error. Neither candidate qualified. All edits were dropped before
submission; the framework reports no changed entries in the final patch. Its
generic reason, "no weight differs from the base model after rounding," refers
to that final submission: the two measured candidates each really changed
**1,024 BF16 entries**, not changes that disappeared in rounding.

## Intervention and its evidential limits

Initial weight inspection found a large final-block normalization coefficient:
channel 1 of `model.layers.27.post_attention_layernorm.weight` is 192 versus
median 3.125. Recorded activation summaries were similar for correct and failed
answers, so this coefficient was not edited merely for being large.

Within the sampled layers and the top 5% of neurons by mean absolute recorded
activity, layer 21 neuron 377 had the largest standardized failed-versus-correct
activity difference (1.43). Its mean absolute activity was 1.048 on the nine
failed traces and 0.686 on the seven successful traces. This association was
selected after inspecting many neurons on only 16 questions; it may reflect
question difficulty or chance and does not establish a semantic function.

The causal probe changed only column 377 of
`model.layers.21.mlp.down_proj.weight`: first zeroing it, then multiplying its
original values by 0.5. Both strengths were declared before the first was scored.
The second was not an additional search direction chosen after a disappointing
result. No gradients, optimizer, loss fitting, or modified-model simulation were
used. All inference used the normal wsplab forward tool; sandbox analysis used
weights and recorded activations only.

The full ablation fixed MAS/2018 but broke AWK/2013 and JPM/2014. Half attenuation
fixed MAS/2018 and PNC/2009, while breaking the same two original successes.
MAS's match is sensitive to the frozen display-precision tolerance (original
prediction 1.46%, edited predictions 1.43 and 1.45 against reference 1.4%). PNC
changes from 33.33 to the reference -1. Neither produces a net accuracy advantage.
Per-item predictions and paired statistics are retained in
[the machine-readable results](natural-finqa-weight-edit-results.json).

## Held-out data and integrity

The private original baseline on the 32 selection reports was **16/32 (50%)**.
Its scores were not inspected during editing. Because no candidate qualified
for submission, no edited candidate was evaluated on selection and no random
control was necessary. The final score field is therefore null, not a newly
measured 50% edited score. The 16 internal HOLDOUT questions and the separate
96-report external evaluation were not evaluated.

Completed three official forward passes and seven sandbox Python runs. There
were 80 new generations: 32 private original-baseline answers, 16 DEV original
answers, and 16 answers for each candidate. Original model and fixture hashes
verified after the run; completion-controller source hashes also verified.
Seven targeted fixture/grader tests and lint passed before launch. The daemon,
completion controller and GPU worker exited; the final GPU process query was empty.

The completion controller applies the precommitted freeze rule and finishes the
session without needing another user message. It permits no extra candidate or
paid call. The [frozen plan](natural-finqa-weight-edit-plan.md), session commands,
raw DEV responses, activations, candidate backups and final no-change patch
remain under `.wsplab/fq-edit1/`.

## What this leaves open

The actual OpenAI/Claude/Gemini peer-council versus matched-budget solo comparison
remains unrun. This small local attempt does not qualify or disqualify FinQA for
that experiment. It establishes that the direct-edit path works and that this
particular failure-associated neuron attenuation did not deliver a net gain.
Any next attempt must be reported as a new experiment, retain a strong unedited
baseline, use explicit spending and compute limits, and preserve the external
test until candidate selection is frozen.
