# FinQA continuation: audit, cheap prompt control, evidence diagnostic

Authorized scope: local inference on the existing 32 pilot questions only. No
paid provider calls, weight edits, new samples, or external evaluation. Preserve
all previous results and the previous screening-rule outcome. This continuation
investigates the mechanism; it does not lower the earlier 40% cutoff to claim a
retrospective pass.

## Audit frozen before the new generations

The [32-item audit](natural-finqa-audit.json) checks source tables, supporting
text, reference calculations, original thinking outputs and scoring conventions.
It is assistant-reviewed, not independent or blinded.

Eight cases are flagged for a secondary sensitivity analysis: unspecified US/UK
pension trust; an approximately phrased sum; signed versus magnitude reduction;
thousands-of-dollars scale; unspecified backlog year; a decrease modeled as an
increase; mutual funds confused with money-market funds; and truncation versus
rounding of 22.555... . Keep these examples in the main 32-item scores and in all
new generations. Do not silently correct labels or drop difficult questions.

All eight happened to be failures in the original thinking run. Thus the other
24 examples give 17/24 (70.8%) rather than 17/32 (53.1%), but this is **post hoc
sensitivity analysis, not a new unbiased benchmark score**. Nonthinking is 4/24
rather than 4/32. Actual errors remain in evidence selection, table reading,
interpreting temporal boundaries, arithmetic and cash-flow signs.

The Mexico-revenue case has a bad execution answer but a supported displayed
answer: 2100 / 21963 * 100 = 9.5615%, rounding to the displayed 9.6%. Retain it
in the 24-item analysis because the primary scorer uses the displayed answer.

## Three fixed conditions, 32 questions each

1. `clean_nonthinking`: full original excerpt, thinking off, replace only the
   percentage example sentence with “For a percentage, give percentage units
   rather than a fraction.”
2. `clean_thinking`: identical prompt change, thinking on.
3. `evidence_thinking`: same simpler instruction and thinking mode, retaining
   annotated source rows/sentences plus conservative table-header prefixes.

Compare the two clean conditions to their original matching modes, and compare
evidence thinking to clean full-context thinking. Gold evidence is a diagnostic
of answering with selected inputs, not an available deployment baseline or a
weight-edit gain. It does not isolate a single neural mechanism.

Map `gold_inds` keys to the original `table_ori` rows using the difference in
header-row counts, and require matching row labels after stripping HTML and
punctuation. Map text indices to pre_text + post_text and require matching source
text. Copy original cells and sentences exactly, retaining original order and
all conservative header-prefix rows. Some such prefixes include an extra data
row; retain it rather than guess a header boundary. Never feed the normalized
gold-rendered sentences, programs, answers or explanations. All 32 mappings were
validated before inference. Wrong original evidence annotations, notably the
mutual-funds case, remain flagged and excluded only from the secondary analysis;
they are not secretly repaired to make the oracle look stronger.

## Execution and scoring

Reuse original Qwen3-0.6B checkpoint, frozen pilot, seed 20261108 plus item index,
BF16 vLLM built-in sampler, top-k 20, no presence penalty. Nonthinking uses
.7 temperature/.8 top-p; thinking uses .6/.95. Same 32,768 total context and full
remaining output allowance, with all prompts preflighted to leave at least
8,192 output tokens. No input truncation. Two concurrent sequences, eager mode,
70% GPU memory, 3-hour active cap checked between batches. A partial run remains
partial. Original files and model hashes verified before and after.

Use the already corrected numeric parser, unchanged across all three conditions.
Retain original-label full-sample accuracy, parsing failures and truncation.
Report the fixed 24-item sensitivity separately. Compute paired changes and
standard errors over questions, not tokens or individual reasoning steps. These
are exploratory single-seed prompt/input comparisons, not confirmation of a
council effect and not official FinQA program/execution accuracy.

Source snapshots, condition plan, audit hash, baseline hashes, mappings, responses,
status and summary are stored in `.wsplab/natural-finqa/controls/`. The
[results report](natural-finqa-controls-results.md) updates each completed batch.
Nine targeted tests pass for grading units/rounding, final-only scoring,
annotation isolation, source-row alignment, and paired completeness; lint passes.

## Decision after review

Any successful cheap prompt change becomes part of the baseline. Interpret an
evidence-only gain only on unambiguous, correctly annotated cases, retaining the
all-item result alongside it. If neither control exposes a sizeable, credible
recoverable gap, do not infer that a council can repair the remaining failures.
If a gap appears, identify specific errors worth an editing attempt and prepare
a bounded comparison with OpenAI, Claude and Gemini versus an equal-budget solo
arm. No paid comparison is launched by this continuation.
