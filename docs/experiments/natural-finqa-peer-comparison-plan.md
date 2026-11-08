# Prepared FinQA peer-versus-solo weight-edit comparison

Status: **completed: no fix found**. See the [results and audit summary](natural-finqa-peer-comparison-results.md).
All four editors returned unchanged weights; no weight edit was attempted.
External inference was skipped after verifying identical checkpoints, retaining
the unused external set. The frozen plan below is retained as experiment history.

The user approved the prepared
$24 comparison and cumulative limits of OpenAI $22, Anthropic $47, Gemini $30.
The exact approval and plan hash are saved in the run's `approval.json`;
live progress and spending are in `.wsplab/finqa-peer-comparison/monitor.json`.
The first launch was stopped during its incomplete private baseline, with zero
provider spending and reservations, to prevent cross-peer HOLDOUT leakage.
Its complete records remain in `.wsplab/finqa-peer-comparison-pre-spend-aborted`.
The replacement plan records this implementation correction and links the
original approval; the approved scope, data, providers and spending caps are unchanged.
That attempt subsequently failed: Claude's concurrent tokenizer import raised
an `ImportError` before its first paid call. OpenAI and Gemini continued, spending
$2.32187765, but their work is **not a valid three-provider comparison**. Records
and checkpoints are preserved in `.wsplab/finqa-peer-comparison-failed-tokenizer`.
No solo or external evaluation occurred. The user explicitly requested "fix and rerun".

The rerun initializes all tokenizers sequentially, gates the first paid request
on every participant completing startup, and cancels further work if a participant
fails. Its new ceilings are **$9.67812235 per arm**: peers receive OpenAI
$1.69037105, Claude $5, Gemini $2.98775130; solo Claude receives $9.67812235.
The rerun can spend at most $19.35624470, or **$21.67812235 including the failed
attempt**, within the original $24 authorization and provider cumulative limits.
The original approved ceilings below are retained as history. All other protocol,
data, models and pass limits remain fixed. Failed-attempt editor findings and
patches are not passed to the fresh editors; this is infrastructure recovery,
not a retry selected for favorable performance.
The [local direct-edit pilot](natural-finqa-weight-edit-results.md) finished with
no retained improvement. That does not gate this comparison. Its purpose is to
test the user's actual question on a genuine task: whether a peer council can
produce a better weight edit than a single model under matched budget ceilings.

## Arms and proposed spending

| Arm | Editors | Forward passes | Python runs | Provider spending ceilings |
|---|---|---:|---:|---|
| Peers | GPT-6.1 Terra, Claude Opus 5.5, Gemini 4 Argon | 6 each, 18 total | 20 each, 60 total | OpenAI $2 + Anthropic $5 + Gemini $5 = **$12** |
| Solo | Claude Opus 5.5 | 18 total | 60 total | Anthropic **$12** |

Maximum additional provider spend is **$24**, allocated OpenAI $2, Anthropic $17,
Gemini $5. Proposed cumulative limits increase from $20/$30/$25 to
**OpenAI $22, Anthropic $47, Gemini $30**. The earlier approval explicitly covered
the corrected CIFAR comparison; the user has now separately approved this FinQA
allocation. No paid preflight or test call was needed.

The solo editor is selected before this experiment's outcomes. This measures a
council against that named solo model, not the strongest possible solo model.
Peers maintain independent weights and share DEV findings after steps, without
a lead or a vote on scripts. A peer's first HOLDOUT generation request and every
subsequent outgoing message are withheld from the others. This prevents early
confirmation results influencing peers that are still searching. HOLDOUT may
only be requested by the official generation tool's problem IDs, never custom
prompts or teacher-forced reads; the audit checks this restriction and that no
peer tunes after confirmation. The original shared Board rounds still apply,
so withholding content introduces no additional synchronization barrier.
All three providers must participate for a three-provider claim. The solo arm
receives no peer findings. Both arms receive the same brief prior local-pilot
finding and otherwise start from identical original weights and practice data.

Both arms have the same maximum total passes, Python runs, editor turns (120),
response token ceiling (16,384 per provider call), high reasoning effort, and
spending ceiling. They may use less; report actual consumption. Peers submit
three candidates and solo one, so this is not an ablation isolating communication
from portfolio diversity or selection. No claim of broad council superiority
will follow from one positive trial.

## Frozen model, task and data

Original Qwen3-0.6B; original full-source-context FinQA prompt; thinking enabled;
unchanged numeric display-answer grader. No prompt changes, evidence annotations,
training, gradients, loss fitting, planted faults or simulated forward passes.
The default embedding/output and final normalization tensors remain frozen.

Reuse the pre-existing company/year report partitions. Choose the first
source-order question with a supported numeric display reference in each report,
before any new generation. Practice has 64 questions (first 32 DEV, next 32
internal HOLDOUT), selection has 32, and external evaluation has **95**. Five
practice reports and three external reports use a later source question because
the first is unscorable. One reserved external report, CME/2010, has no supported
numeric reference and is excluded by this eligibility rule; there is no
replacement report. This is reference eligibility, not selection on model failure.
All chosen IDs, exclusions and source hashes are recorded in the frozen manifest.

External question text and answers have not been inspected by the editor or
used for generation. Preparation tokenizes them only for length checks. The
editor sandbox mounts only practice data, its working directory and model
weights; selection and external files stay outside those mounts.

BF16 vLLM, two concurrent sequences, eager execution, 70% GPU memory, 32,768 total
context and full remaining output allowance. Temperature .6, top-p .95, top-k 20,
presence penalty zero; fixed SHA256-derived item seeds. Each pass permits at most
64 generations, or 32 per model for paired requests. Generation and activation
reads from different peers are serialized on the GPU. Only original-model
generations can be cached across arms; edited responses never substitute for
original responses. Raw responses are archived outside editor mounts.

## Selection, audit and evaluation

Editors choose and confirm their own edits using practice only. They must freeze
before one internal HOLDOUT comparison and not tune afterward. Each finished
candidate is then scored privately on selection. Select the best score within
an arm, with the original model also eligible: an edited tie or regression never
displaces it. Ties among improving candidates prefer fewer changed entries,
then deterministic provider order. There is no pooling of peers' weights.

After all four sessions finish, freeze both winners and audit every contributor's
transcript for training, unauthorized model replay, evaluation access and actual
three-provider participation. External evaluation is blocked until an audit tied
to the exact plan and selections says every session is valid. A violation yields
an invalid experiment, not a council win or a silent two-provider fallback.

Then score original, selected peers and selected solo on the same 95 external
questions. Primary comparison is peers minus solo, with paired changes, standard
error and exact two-sided binomial p. Also compare each arm against original;
for an arm with a positive gain, evaluate a randomized same-size weight edit.
The recorded council weight-edit win requires positive gains over **solo,
original and random control**, each with exact paired p < .05. Report the
individual comparisons even when this joint criterion fails. One seed, a modest
test set and imperfect public labels limit generalization; no follow-up tuning
or favorable-seed retries use this external test.

## Pricing and accounting

Rates were checked against official documentation on 2026-11-07. The conservative
reservation rates per million input/output/cached-input tokens are:

- OpenAI: $1.25/$5/$0.05, covering the documented 30-minute cache-write rate as
  well as ordinary $1 input. [GPT-6.1 Terra pricing](https://developers.openai.com/api/docs/models/gpt-6.1-terra)
- Anthropic: $5/$20/$0.20, covering five-minute cache writes as well as ordinary
  $4 input; the adapter uses the default ephemeral cache duration.
  [Claude Opus 5.5 pricing](https://platform.claude.com/docs/en/models/opus-5-5/overview)
- Gemini: $2/$10/$0.10, including billed thinking tokens. Those promotional
  rates end after November 30; the runner rejects this plan from December 1.
  [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)

The existing guarded ledger reserves the bounded call cost before sending,
reduces output allowance near a ceiling, and conservatively retains a reservation
when a call fails. Missing reported output usage is charged at the requested
ceiling. Provider invoices remain authoritative; local reported costs are estimates.

## Ready-to-run artifacts and validation

Runner: `scripts/live/finqa_peer_trial.py`. Frozen manifest and private fixtures:
`.wsplab/finqa-peer-comparison/`. Approval and launch records are now present;
earlier draft manifests and the unapproved template are retained.
All three configured key files exist; secrets were not printed or tested remotely.

Offline prompt preflight leaves at least 30,602 output tokens for every question.
Eighteen focused tests cover grading, fixture isolation, pre-call spending guards,
matching arm ceilings, selection ties/regressions, approval/plan matching,
external-audit gating, paired IDs, GPU serialization and original-only caching.
Lint passes. The shared budget helpers now lazily import their unrelated digit
benchmark, so the language-model environment does not require scikit-learn.

Full candidate checkpoints use `/tmp/wsplab-finqa-peer-comparison`; at least
20 GiB scratch is required before spending. Durable patches, transcripts, raw
responses, manifests and results remain in the workspace. Existing local-pilot
artifacts are preserved. No live provider-backed comparison has been tested yet.

Execution after approval:

```bash
~/.venvs/wsplab-gpu/bin/python scripts/live/finqa_peer_trial.py --run-editors
```

After the transcript audit and frozen-selection record:

```bash
~/.venvs/wsplab-gpu/bin/python scripts/live/finqa_peer_trial.py --evaluate
```
