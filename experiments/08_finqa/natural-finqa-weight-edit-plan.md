# FinQA: direct weight-edit pilot

The user redirected the experiment toward its actual hypothesis: can direct weight
edits improve a genuine weak task, and eventually can a peer council edit better
than a solo editor? Prompt improvements are not an eligibility requirement. The
previous controls remain reported, but are not a gate for this experiment.

This first bounded attempt uses the coding agent as editor through normal wsplab
session tools. It incurs no new provider API calls and is not a council comparison.
No claim about council superiority can follow from this pilot alone.

## Frozen task and partitions

Use the original, unmodified Qwen3-0.6B checkpoint and original full-context FinQA
prompt with thinking enabled. No abbreviated context, annotated evidence, prompt
changes, injected faults, training, or optimizer. Keep the corrected scalar scorer
against the original display answers. Do not substitute execution answers or
silently adjudicate labels based on model outputs. This remains a custom numerical
answer metric, not official FinQA execution accuracy.

Reuse the previously frozen report partitions. Select one question per report:
64 practice reports and 32 selection reports. For practice, take the first
source-order question with an existing numeric display reference. Five reports'
first questions have unsupported references (three empty, two nonempty), so this
rule chooses a later question in each; the skipped IDs are recorded before any
new inference. No report is dropped. The selection reports' first questions all
have supported scalar references. No question is chosen based on model failure.

The first 16 practice records are DEV; the next 16 are internal HOLDOUT. Do not
use the remaining 32 practice records in this pilot. Evaluate the frozen final
patch on the 32 separate selection records, whose content and baseline outcomes
remain hidden from the editor. These selection results will inform later work,
so they are not the final external test. The reserved 96-report external
partition is neither materialized nor evaluated by this run.

Sample one response per question using wsplab's SHA256-derived item seed, identical
for original, edited and random-control weights. This differs from the earlier
pilot's index-derived seeds and uses different questions, so compare paired scores
within this run rather than comparing directly to the previous 17/32.

## Editing and stopping

- At most six official forward passes and 20 sandbox Python runs.
- Inspect weights and recorded activations; choose at most two nonzero candidates.
- Never simulate the model or evaluate a modified layer outside the forward tool.
- Preserve the default frozen embedding/output and final normalization tensors.
- Use paired DEV measurements. Keep the best candidate only if its DEV gain
  exceeds one paired standard error; otherwise restore original weights.
- Freeze before one internal HOLDOUT comparison. No tuning after HOLDOUT.
- Finish with unused passes when the decision is made. A no-change result is valid.

The final wsplab evaluation compares original and submitted weights on selection.
If the submission changes weights, it also measures a randomized edit of the same
size and emits the ordinary framework verdict. Report that verdict unchanged,
with paired counts, uncertainty and its small-sample limitations. Any later
external confirmation must use the untouched partition after the experiment is
frozen. A weak result is not an instruction to retry until something wins.

## Execution and integrity

Local BF16 vLLM, eager execution, two concurrent sequences, 70% GPU memory,
32,768 total context and the full remaining context allowance per prompt. No input
truncation; require at least 8,192 available output tokens. Temperature .6, top-p
.95, top-k 20, presence penalty zero. The same runner serves practice and final
evaluation. Host checks confirm GPU and bubblewrap work.

The runner is `experiments/code/finqa_edit.py`. Its manifest freezes model, data,
scoring source, runner source, selected IDs and instructions. Final verification
checks the original checkpoint and fixtures again. Artifacts live in
`.wsplab/fq-edit1/`; the editor sees only its practice mount and session outputs.
Seven targeted grading/fixture tests pass before launch.

## Subsequent council experiment

Prepare a separate matched-budget comparison with OpenAI, Claude and Gemini
against a solo editor. Distinguish a single solo editor from a portfolio of
independent editors. Freeze provider spending, forward-pass allowance, candidate
selection and the external evaluation before launching. Existing spending approval
was specific to CIFAR; this local pilot does not spend that budget on FinQA.

## Completed outcome

The [completed results](natural-finqa-weight-edit-results.md) report 7/16 original,
6/16 full ablation, and 7/16 half attenuation on DEV. Neither candidate qualified;
the final submission restores the original weights, with verdict `no_fix_found`.
Three forward passes were used. Internal HOLDOUT and external evaluation remained
unused. The private selection baseline was 16/32, inspected only after submission.
