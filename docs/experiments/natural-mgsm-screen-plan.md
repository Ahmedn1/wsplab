# Next genuine task candidate: multilingual arithmetic

Status: the initial local screen is complete; see the [results](natural-mgsm-screen-results.md).
This is not a paid council experiment or a measured council win. The public
checkpoint and dataset are frozen. The full qualification screen and controls
are now running under the [continuation plan](natural-mgsm-controls-plan.md);
their review is still required before committing editor spend.

## Choice and rationale

Use unmodified **Qwen/Qwen3-0.6B** on **MGSM-Rev2**, initially English, Bengali
and Swahili. These are ordinary grade-school word problems in a published
multilingual benchmark. We do not inject weight damage, corrupt examples,
choose examples from model failures, or manufacture adversarial inputs.

The hypothesis is a language-dependent obstacle to mathematical reasoning:
the same model may solve an English question but fail its translated version.
Paired English questions help distinguish that from simply lacking the needed
mathematical ability. A language gap would not itself prove that weight editing
can remove the obstacle; language knowledge can also be missing.

The [Qwen3 technical report](https://arxiv.org/html/2505.09388v1#S4.T19)
reports 77.6% MATH-500 in thinking mode for this model. That makes it a poor
choice if our only aim is a very low English-math baseline, but provides evidence
of some mathematical capability. It does **not** supply a measured MGSM baseline
for this proposed setup. The [model card](https://huggingface.co/Qwen/Qwen3-0.6B)
documents thinking/non-thinking modes and sampling controls for repetition.

The corrected data matters. [Peter et al.](https://arxiv.org/abs/2511.05162)
found that translation and answer-extraction errors inflated apparent language
gaps, with much of the gap disappearing after correction in their experiments.
We therefore use their [MGSM-Rev2 release](https://github.com/google-research-datasets/MGSM-Rev2),
which also discloses use of Gemini for revised translations and verification.
This is a reason to test the hypothesis carefully, not evidence that this small
model necessarily has a large gap.

Other candidates considered:

| Candidate | Decision |
|---|---|
| HANS with an MNLI classifier | Excluded as the main test: deliberately constructed syntactic traps do not fit the user's preference |
| Another small image classifier | Low priority: another low score alone gives no reason to expect a removable mechanism |
| SmolLM2-360M-Instruct on GSM8K | Its card reports 7.43% in a five-shot setup, but that may primarily reflect missing reasoning ability |
| Qwen3-0.6B on English MATH-500 | Published 77.6% thinking result; not the severe natural weakness initially sought |
| Qwen3-0.6B on corrected multilingual math | Selected for baseline screening; same-problem English control and deterministic grading |

Sources for alternatives: [HANS paper](https://aclanthology.org/P19-1334/),
[SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).

## Frozen resources and partitions

- Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- MGSM-Rev2 revision: `5f091edeae66ebc605ac4f98dca07a6c345f0060`.
- Dataset license: CC BY-SA 4.0; model license: Apache 2.0.
- All 250 rows have aligned numeric gold answers across the three languages.
- Split with Python `random.Random(20261106)` by **underlying problem**, keeping
  every language version together: 32 pilot, 64 practice, 40 selection, 114
  external evaluation groups. No English translation of an external item may
  enter editor practice.
- `.wsplab/natural-mgsm/plan.json` stores IDs and SHA-256 digests. Each execution
  records its settings and script digest. Final evaluation remains unrun.

## Screening before editor spending

The first eight pilot groups are a CPU feasibility check, not a performance
claim: full-precision unmodified weights, non-thinking mode, temperature 0.7,
top-p 0.8, top-k 20, 2,048 output tokens, a fixed seed per problem, and the
model-card math instruction. CPU use is limited to two threads, at lower process
priority, with CUDA disabled; the Astra run retains the GPU.

Record raw responses, token counts, EOS versus truncation, strict boxed-answer
accuracy and lenient extraction diagnostics. Normalize Unicode decimal digits,
including Bengali digits. Never credit a number from unfinished thinking as a
final answer. Manual review must distinguish extraction errors, mistranslations,
arithmetic errors, and repetitive or unfinished responses. A shared English
formatting instruction is itself a possible language-comprehension confound;
check an instruction translated into the question language before attributing
failures to reasoning.

Also retain a simple instruction to reason in English while receiving the
original Bengali/Swahili question. If that is sufficient, a weight edit must
beat or add value beyond that cheaper control. An English advantage can reflect
unequal training exposure or memorization as well as a removable behavioral
mechanism; the public benchmark cannot distinguish these by itself.

Before a paid attempt, run all 32 pilot groups with adequate output length and
both documented thinking modes. Use the recommended 32,768-token allowance
subject to the model's total context limit; do not create a failure by a short
token cutoff or greedy thinking-mode decoding. If repetition is substantial,
also check the model-card presence-penalty remedy. These inference controls must
be retained as baselines, not ignored to make weight editing look better.

Predeclared qualification rule on the full pilot: English accuracy at least
50%, mean Bengali/Swahili accuracy at most 40%, a gap of at least 20 percentage
points, and at most 10% truncation. If the gap disappears under extraction,
language-prompt or documented decoding controls, reject this rationale for
editor spending. Report pilot uncertainty; 32 groups establish only an
exploratory lead.

## Eventual council comparison, conditional on qualification

Include OpenAI, Claude and Gemini in peers and independent search. Retain each
solo result and the independent-search result chosen on selection data. Compare
both to unchanged weights and the ordinary decoding controls. Freeze the task,
budget, output lengths and selection rule before any editor runs. Do not use
external scores to pick editors, languages, prompts, checkpoints or retries.

Separate two claims: peers beating a same-budget portfolio of independent editors,
and peers beating one editor given the council's entire budget. The latter needs
its own full-budget solo arm; the individual members of a split-budget portfolio
are insufficient controls. Choose that solo editor using earlier pilot evidence,
or test all three with equal total allowances and account for all comparisons.

Primary outcome: mean accuracy on Bengali and Swahili, with English regression
reported. Paired uncertainty must cluster by underlying problem: translations
and repeated samples are not independent data points. The 114 external groups
make this a pilot suited to large effects, not a precise test of small gains.
Any promising result needs a separate replication.

The CIFAR protocol failures remain prerequisites to address: reject unauthorized
network simulation before peers receive its output; use the corrected stopping
prompt; and resolve or conservatively account for missing provider usage. No
paid editor process is started by this screening plan. The previous $14 approval
covered the completed CIFAR comparison, not a new budget.
