# Next candidate: English multi-document question answering

Screen unmodified **Qwen/Qwen3-0.6B** on **HotpotQA**, using the published
`distractor` configuration unchanged. This is a candidate, not an established
hard case for this particular checkpoint or a demonstrated council win.

## Why this candidate

[HotpotQA](https://hotpotqa.github.io/) contains human-written questions requiring
facts from Wikipedia passages. The standard distractor setting supplies ten
paragraphs, including the benchmark authors' distractors. We do not add
adversarial text, damage weights, reorder context to induce failure, or choose
questions based on model failures. The setting is curated rather than a raw
production distribution; its distractors are part of the original benchmark.

Unlike the previous multilingual candidate, this tests English reading. Gold
supporting-paragraph annotations permit a diagnostic comparison: can the same
model answer when evidence selection is supplied? A large paired gap would
suggest difficulty using context, without proving that a weight edit can fix
it. Oracle success also makes the task easier and cannot uniquely identify an
attention defect or separate every reasoning demand.

Other routes considered: instruction-format benchmarks have cheap constrained
output controls that may dominate a weight edit; another small model's low
math score alone gives little evidence of repairability; deliberately generated
syntactic traps do not fit the requested task. This pilot does not claim that
HotpotQA is uniquely optimal.

## Frozen resources and split

- Model: existing original Qwen3-0.6B checkpoint, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`; no new model download or modification.
- Dataset: [hotpotqa/hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa),
  revision `1908d6afbbead072334abe2965f91bd2709910ab`, distractor validation,
  7,405 rows. Dataset card declares CC BY-SA 4.0.
- Scorer: unchanged answer-normalization, EM and F1 functions from the
  [official evaluator](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py).
  Only these reviewed functions are loaded; no CLI or unnecessary dependencies.
- Seed 20261107, chosen before inference. Normalize question case/whitespace,
  group duplicate questions, sort then shuffle groups. Reserve 32 pilot groups,
  128 practice, 128 selection, 1,000 external evaluation, and leave the remainder
  unused. Pilot uses one representative per duplicate group. No stratification
  or selection on answers, outcomes, length or question type.
- Partition IDs, input hashes and model hashes are in
  `.wsplab/natural-hotpotqa/plan.json`. Only pilot questions are used for inference.
  The source file is read programmatically to establish partitions, but external
  examples are not inspected, prompted or scored.
- Titles/documents may overlap partitions; this is question-disjoint, not
  document-disjoint. Public-benchmark pretraining contamination is possible.

## Fixed six-condition screen

Run in the order below, on the same 32 questions, one sample each:

| Condition | Passages | Thinking |
|---|---|---|
| full_nonthinking | All original paragraphs | off |
| oracle_nonthinking | All sentences in gold supporting paragraphs | off |
| bm25_nonthinking | Top four paragraphs by fixed lexical BM25 | off |
| full_thinking | All original paragraphs | on |
| oracle_thinking | All sentences in gold supporting paragraphs | on |
| bm25_thinking | Top four paragraphs by fixed lexical BM25 | on |

Oracle uses gold support labels **only as a capability diagnostic**. It is not
available to the deployed system and must never be reported as a fair baseline
or a weight-edit improvement. BM25 uses the question and paragraph text/title
only, with fixed k1=1.5, b=0.75 and a fixed stopword list. Selected paragraphs
retain original relative order. No learned retriever, provider call, answer
labels or support labels enter BM25 ranking.

No context truncation. Before inference check all prompts leave at least 8,192
output tokens within the 32,768-token total context; abort if not. Actual output
allowance is the entire remaining context, not 8,192. Use the model-card sampling
settings: temperature .7/top-p .8 nonthinking; .6/.95 thinking; top-k 20; presence
penalty zero; seed 20261107 plus pilot index. Hardware: local BF16 vLLM, two
concurrent sequences, 70% GPU memory, eager execution. The model must start only
with at least 14,000 MiB free. Three-hour active limit checked between batches;
partial runs remain incomplete. No training, weight edits or paid calls.

## Grading, interpretation and stopping

Primary measure: official normalized answer exact match. Report answer token
F1 as secondary; this is not the benchmark's joint answer/support-fact metric.
Request `<answer>...</answer>`, parse only final output after thinking ends,
and score a plain final response as a whole if tags are absent. Never search
reasoning for a correct substring. Record missing tags, truncation, unfinished
thinking, token counts and whether BM25 retains all supporting paragraphs.
Check grading/formatting failures manually before claiming the task is hard.

Exploratory qualification rule, fixed before inference:

1. In at least one matched thinking mode, full-context EM <=50%, oracle EM >=65%,
   and oracle-minus-full EM >=20 percentage points.
2. The best full-context or BM25 condition across modes has EM <=60%.
3. Every condition has <=10% truncation and <=10% missing answer tags.

Report paired differences and standard errors across the 32 questions. Passing
these thresholds only justifies further review; it does not prove an editable
mechanism or reliable model improvement. Comparing multiple modes makes this
exploratory. If either thinking or a cheap lexical filter solves the problem,
retain that as a strong baseline rather than choosing the weaker condition.
If exhaustion dominates, check documented decoding remedies before editor spend.
Do not enlarge or reroll the pilot just to pass the rule.

## Eventual council comparison

Keep OpenAI, Claude and Gemini in the peer council. Use an equal-total-budget
solo arm and a selection-matched independent-search control if comparing against
parallel search. Freeze edits and selection before external evaluation, account
for all provider use, and prevent unauthorized network simulation before peer
outputs are shared. This screen launches no council and incurs no provider cost.
The previous paid authorization was specific to CIFAR-100.

## Execution validation

Six targeted tests pass for deployable-prompt independence from gold labels,
whole-paragraph oracle selection, unfinished-thinking and multiple-answer
handling, and duplicate/foreign resume records. Input hashes, disjoint IDs,
all ten original paragraphs and oracle support coverage were checked for all
32 pilot items. The frozen random sample contains 22 bridge and 10 comparison
questions; no resampling was performed. Lint and whitespace checks pass.

The first engine startup failed before generating any answers because the
FlashInfer sampler required unavailable nvcc. Its source/settings/logs are
preserved under `startup-failed-no-responses`. The active attempt uses the same
vLLM built-in sampler as the repository LM runner and the completed MGSM run.
Its exact source and settings are copied into the run directory. No examples,
conditions or statistical rules changed in response to this startup failure.

[Live results](natural-hotpotqa-results.md) update after each completed batch.
The run is local and makes no paid provider calls.
