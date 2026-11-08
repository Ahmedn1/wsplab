# Candidate: numerical reasoning over real financial reports

Screen original **Qwen3-0.6B** on **FinQA**, after the completed HotpotQA pilot
failed its predeclared qualification rule. This remains an unpaid task screen,
not a weight edit, training run or council result.

[FinQA](https://github.com/czyssrs/FinQA) contains expert-written questions over
financial reports, with numerical answers and reasoning programs. The
[paper](https://arxiv.org/abs/2109.00122) documents the underlying task. These are
published questions; we inject no faults, adversarial text or fabricated data.

## Frozen pilot

Dataset revision `0f16e2867befa6840783e58be38c9efb9229d742`, public development
file, 883 questions in 224 company/year reports. Sort report IDs and shuffle
with seed 20261108, before any model inference. Reserve 32 reports for the pilot,
64 for practice, 32 for selection, and 96 for external evaluation. Use the first
source-order question in each pilot report, 32 total. All questions from the
same report stay together. This samples reports, not uniformly all questions;
report the scope honestly. Remaining report text/questions are not inspected
or evaluated by the editor. Different years of a company may appear in different
partitions. Public benchmark training contamination remains possible.

Use the existing, hash-verified Qwen3-0.6B original checkpoint at revision
`c1899de289a04d12100db370d81485cdf75e47ca`. Input hashes, report IDs and model
hashes are frozen in `.wsplab/natural-finqa/plan.json`.

## Inputs and controls

Supply all provided `pre_text`, the original `table_ori`, all `post_text`, and
the question. Preserve original multi-row table headers. The dataset's normalized
table can lose year distinctions; one pilot example duplicates a 2016 header
where the original table distinguishes 2016 and 2015. Do not use normalized
`table`, gold programs/explanations, `gold_inds`, `model_input`, or supplied
retrieval outputs in the prompt. The authors also document a historical
retrieval label-leakage bug; this pilot uses neither their learned retriever nor
that row-conversion implementation.

Run all 32 examples with thinking off, then thinking on. Use .7/.8 temperature/
top-p off, .6/.95 on, top-k 20, presence penalty zero, and seed 20261108 plus pilot
index. Each prompt uses the same instruction to calculate and give a boxed
number in the question's units; percentage answers must be in percentage units,
not fractions. No generated intermediate calculation is executed externally.

Original BF16 checkpoint, local vLLM built-in sampler, two concurrent sequences,
70% GPU memory, eager execution, 32,768 total context. No passage truncation;
preflight every prompt for at least 8,192 available output tokens and actually
allow the full remaining context. Abort if any prompt is too long. Three-hour
active cap checked between batches. A partial run is not a completed screen.
If repetition/truncation dominates, test decoding remedies before editor spend.

## Honest metric and data-quality limits

This is a **custom direct-numeric-answer task over FinQA data**, not its official
program-generation/execution leaderboard metric. We are not claiming official
execution or program accuracy. Grade against published `qa.answer` in its display
units, retaining `qa.exe_ans` separately for audit. Official execution answers
may be fractions while displayed answers are percentages; some are inconsistent.
One pilot item (`UNP/2013/page_54.pdf-2`) has a displayed answer 9.6% but execution
answer .0001, so target quality needs manual review. No item is dropped or label
changed based on model behavior.

Parse only one boxed final scalar, or a complete standalone scalar final response
optionally preceded by `Answer:`. Strip commas/currency symbols; preserve sign
and percentage-unit interpretation. Do not rescale by 100 to get a match. Never
search reasoning for a correct number, and never grade unfinished thinking.
Use half a unit of the displayed decimal precision as rounding tolerance;
whole-number percentage answers allow half a percentage point. Plain integer
answers use absolute tolerance 1e-6. This custom tolerance accommodates the
published rounded answers and must be reported rather than equated to the
strict official execution evaluator. Preserve predictions and tolerances.

Qualification rule before inference: both conditions complete, best accuracy
<=40%, and each condition has <=10% truncation and <=10% unparseable final answers.
Then manually inspect numerical reasoning errors and questionable labels. Passing
only establishes a genuine weak task; evidence of a recoverable mechanism is
still needed before claiming weight editability or starting a paid comparison.
The earlier English MGSM result shows general arithmetic ability but does not
prove financial-table competence.

Ten combined HotpotQA/FinQA targeted tests pass, covering gold-label isolation,
original-table preservation, final-answer-only grading, numerical scale and
rounding, and resume integrity. Lint passes. No paid provider calls are made.

An eventual council still includes OpenAI, Claude and Gemini, with an equal-total-
budget solo control and untouched external evaluation. Existing CIFAR-specific
spending authorization is not transferred to this task.

## Parser correction and initial audit

The initial parser failed to normalize the LaTeX escape `\%`, which incorrectly
marked some numerical outputs unparseable. This was discovered by inspecting
actual generated responses. A presentation-only correction normalizes escaped
percent/currency symbols and thin spaces; regression checks ensure it does not
rescale percentages or credit reasoning-only numbers. No prompts, seeds, weights,
gold labels or generated responses were changed. The original inference script
and original scores remain frozen under `run/`; a separate scorer replays the
same outputs and writes [corrected results](natural-finqa-rescored-results.md).
Its watcher only reads generations and writes reports; it performs no inference.

The complete nonthinking condition has 4/32 corrected strict numeric matches,
3/32 unparseable final outputs, and no truncation. Manual inspection found two
of the three unparseable outputs are correct prose answers (2234 and 688), while
the third gives conflicting boxed answers. Thus 12.5% strict accuracy should not
be equated to semantic accuracy; those two prose answers would make 6/32 (18.75%).
The conflicting response also copies the illustrative percentage in the prompt;
if this candidate proceeds, remove that example in a cheap prompting control
before claiming a purely weight-level defect. No selective answer correction is
applied to the official reported custom metric.

Examples of substantive errors: adding two quarterly prices instead of
subtracting when calculating growth; answering a three-year average with a
single year's change; returning accrued interest rather than its requested
percentage; and treating annual derivative values as identical. The thinking
condition remains necessary before deciding whether this is a suitable candidate.

## Completed pilot and authorized continuation

Both original conditions finished: corrected numeric scoring is 4/32 (12.5%)
nonthinking and 17/32 (53.125%) thinking, with no truncation. This fails the
original <=40% best-condition screening rule. That historical result remains
unchanged. The subsequent [source/reference audit](natural-finqa-audit.md) and
[fixed prompt/evidence controls](natural-finqa-controls-plan.md) investigate why
the model fails, instead of changing the original cutoff or claiming a pass.
