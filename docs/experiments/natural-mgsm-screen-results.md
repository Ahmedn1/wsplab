# MGSM-Rev2 candidate screen — 2026-11-06

**A genuine hard-task candidate was found, but editability and council advantage
remain untested.** The unmodified Qwen3-0.6B checkpoint performs much worse on
Bengali and Swahili than on the same eight problems in English in this small,
fixed exploratory sample.

| Language | Correct | Accuracy | Hit 2,048-token cap |
|---|---:|---:|---:|
| English | 6/8 | 75.0% | 0/8 |
| Bengali | 1/8 | 12.5% | 3/8 |
| Swahili | 0/8 | 0.0% | 0/8 |

These are **eight underlying problems**, not 24 independent problems. They are
the first eight groups of a seeded 32-group pilot chosen before inference. The
results are not full-benchmark scores, not comparable to differently configured
published scores, and not a council result. Strict boxed-answer and lenient
grading agree on all 24 outputs. Unicode digits were normalized and checked
against Bengali, Arabic-Indic, full-width and ASCII examples.

## What the traces show

- Problem 85 describes 22 football games with eight more wins than losses.
  English correctly returns 15; Bengali returns 30 and Swahili returns 8.
  All three responses finish normally.
- Problem 76 describes four sandcastle levels with successively halved areas.
  English correctly returns 60. Bengali returns 8; Swahili invents a story
  about mammoths and returns 4. All responses finish normally.
- Bengali problems 118, 223 and 208 hit the cap with repeated phrases or
  repeated restatements. For example, problem 118 repeats one short phrase
  204 times. These are observed repetitive outputs, not evidence that every
  unfinished response would stay wrong with a longer allowance.
- There are also ordinary English reasoning mistakes. On problem 144 the
  model identifies the taxable items but computes the tax incorrectly.
  This is not a clean separation between a perfect English solver and a
  broken translation component.

The all-wrong Swahili result cannot be attributed to truncation in this sample:
every Swahili response terminated normally. But misunderstanding the question
may reflect missing language knowledge. Correct English answers do not prove
that an editable obstacle is hiding equivalent Swahili competence. Unequal
training exposure and memorization are also possible explanations.

## Why use this task

[MGSM-Rev2](https://github.com/google-research-datasets/MGSM-Rev2) contains
corrected versions of a public multilingual grade-school math benchmark. No
weights were damaged, inputs corrupted, or examples selected because this model
failed them. The release corrects translation/wording issues and discloses use
of Gemini for revised translations and verification. Its
[accompanying paper](https://arxiv.org/abs/2511.05162) warns that translation
and extraction errors can create misleading language gaps. Using corrected
data reduces that concern; it does not eliminate the need for controls.

The potential research question is whether peers can diagnose and reduce
language-dependent misinterpretation and repetitive output better than a solo
editor, while retaining English performance. It is more specific than asking
editors to improve an arbitrary low-scoring classifier.

## What must happen before a paid comparison

The eight-group screen does **not** qualify the task for spending. In particular,
3/24 outputs were truncated (12.5% overall; 37.5% of Bengali), so a larger token
allowance and ordinary repetition controls matter. The full predeclared
qualification rule applies to all 32 pilot groups with adequate output length.

Check documented thinking/non-thinking decoding, the model-card repetition
remedy, instructions in the question's language, and a request to reason in
English. Inspect whether the model can correctly paraphrase the input. If a
cheap prompting/decoding change removes the gap, retain it as a baseline. If
the model lacks basic comprehension, deprioritize weight surgery rather than
calling the low score an opportunity by itself.

The eventual comparison should include OpenAI, Claude and Gemini, with both
independent search and a solo editor receiving the council's total budget as
controls. Enforce the simulation boundary before shared evidence is consumed.
Use the reserved groups once after freezing candidates; never split translations
of the same problem across practice and evaluation.

## Reproducibility and status

- Model: `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Data: `google-research-datasets/MGSM-Rev2`, revision
  `5f091edeae66ebc605ac4f98dca07a6c345f0060`.
- CPU float32 inference from the original checkpoint; non-thinking mode;
  temperature 0.7, top-p 0.8, top-k 20; 2,048 output tokens; fixed per-problem
  seed. This short allowance is a feasibility constraint, not the model-card
  recommended benchmark allowance.
- Two CPU threads at lower process priority, CUDA disabled. Runtime: about
  27.6 minutes. **No paid API calls.** The process finished normally.
- All frozen model, tokenizer, dataset and pilot hashes still match after
  inference. All 250 problem groups form complete, disjoint partitions;
  external evaluation has not run.
- [Plan](natural-mgsm-screen-plan.md), [machine-readable results](natural-mgsm-screen-results.json),
  and runner: `scripts/live/mgsm_screen.py`.
- Full raw responses: `.wsplab/natural-mgsm/nonthinking-cpu-8x3-2048/responses.jsonl`.

Follow-up: the full 32-group qualification screen and decoding/comprehension
controls are now running. See the [completed control evidence](natural-mgsm-controls-review.md)
and [live progress](natural-mgsm-controls-results.md). No council session has
started on this task.
