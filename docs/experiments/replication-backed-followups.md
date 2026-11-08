# Replication-backed council follow-ups

Status: preparation approved by the user on 2026-11-07, including a non-LLM
example. This is an experiment queue, not a report of our own positive results.
No new provider run or budget increase is authorized by this document itself.
Existing cumulative provider ceilings remain binding. FinQA retains its current
GPU allocation and frozen protocol; prepare these follow-ups without contention.

## Decision

Retain Qwen math as a positive control, investigate selective translation on a
smaller model, and add **ResNet-50 on Spurious ImageNet** as the primary non-LLM
hard-case candidate. Add **Whisper-small non-speech hallucination** as a secondary
speech-model positive control. Whisper is an audio-to-text transformer, not a
general-purpose text LLM; ResNet provides the unambiguous non-language example.

Do not select tasks based on a council-versus-solo result. Admit tasks using
baseline failure, reproducibility, protocol compatibility and data availability.
Report every admitted comparison, including null results and regressions.

## Non-LLM hard case: natural spurious cues in ResNet-50

Use the unmodified torchvision ResNet-50 IMAGENET1K_V1 checkpoint. The official
Spurious ImageNet benchmark contains real OpenImages photographs showing a
spurious cue without its associated object; it does not damage the network or
insert an artificial trigger. Use ImageNetV2 for preservation checks.

Sources:

- [Official benchmark and method](https://github.com/YanNeu/spurious_imagenet)
- [Independent weight-edit report](https://github.com/kevinpl07/weight-patching/blob/main/archive/spurious_resnet/reports/RESULTS_spurious.md)
- [Independent protocol and amendments](https://github.com/kevinpl07/weight-patching/blob/main/archive/spurious_resnet/reports/PLAN_2026-11-04_spurious_repair.md)

Three independent Opus sessions reportedly cut spurious predictions from 50.2%
to 15.7–21.8%, but **zero of three passed all original improvement and clean
accuracy requirements**. A later DFR-style training baseline passed. This is
evidence of editable behavior with a difficult preservation trade-off, not a
successful end-to-end repair or evidence for councils. A scripted recipe
reproduced the edits, so it must be a comparator.

Preparation requirements:

1. Pin model/data revisions and hashes. Inspect archive members rather than
   executing upstream download/setup scripts. Retain image identities, source
   classes and duplicate hashes in the split manifest.
2. Reproduce the published baseline and fixed recipe on a clearly labelled
   replication partition. Those outcomes cannot serve as a fresh council test.
3. Freeze independent DEV, internal confirmation and external evaluation
   partitions before editor sessions. Group uncertainty estimates by cue class;
   photos from the same class are not independent task replications.
4. Score both spurious prediction reduction and clean accuracy, including clean
   accuracy on targeted classes. Overall accuracy alone can hide destruction of
   a few classes. Keep unseen cue classes as a separate transfer result.
5. Resolve the metric discrepancy before implementation: the official README
   describes AUC, whereas the independent report calls its score AP. Implement
   explicitly named ROC-AUC and average precision; never compare them as if
   they were the same metric. ImageNetV2 positives also differ from the official
   benchmark's ImageNet validation positives.
6. The external code uses cached final-layer features. Our editors must invoke
   the official forward runner for every candidate prediction, even for an
   fc-only patch. Cached activation statistics are allowed; replaying the layer
   to score candidate weights in the editor kernel is not.

Controls: original weights; fixed published channel-edit recipe; matched random
edits; council; solo editors; a no-communication multi-provider portfolio.
Include DFR-style retraining as a separately labelled conventional-training
baseline, never as a valid strict weight-programming submission. Compare on
equal available data; distinguish any extra-data training arm.

## Speech positive control: Whisper-small

[Released autonomous-session report](https://github.com/kevinpl07/weight-patching/blob/main/archive/reports/RESULTS_2026-11-05_practical_edits.md)
reports hallucination on 300 non-speech clips falling from 84% to 0.3%, 0.3%
and 0.7% across three blind Opus sessions, with 1,537 entries changed and
normalized speech transcripts unchanged. Raw punctuation sometimes changed.
The report includes deviations and additional speech guards; freeze our own
criteria in advance. Its turbo transfer fails a padded-speech guard without VAD.

This is a stronger positive control than a likely council-win task: successful
solo performance is already near the floor. First reproduce on real environmental
audio and instrumental music, with clean, noisy, padded and difficult speech
guards. Include default decoding, DEV-tuned thresholds, VAD and their combination.
Check raw output and normalization separately: garbage that normalizes to empty
must not count as success. Verify checkpoint reload with unchanged inference code.

The generic released gate framework fits probes for some tasks. Audit the actual
Whisper session code separately before claiming our no-fitting rule is met.
Activation differences of means can be descriptive analysis; supervised fitted
gates and custom layer replay cannot be silently admitted.

## Qwen positive control

[Independent experiment](https://github.com/leofmarciano/weightspace/blob/main/README.md)
reports a four-neuron Opus edit of Qwen3.5-0.8B: MATH 8.4% to 40.0%, GSM8K
12.4% to 53.2%, on 250 questions each with thinking enabled. The no-thinking
control is strong and uses a different sample size. Do not frame this as newly
acquired math ability or compare those percentages as a paired test.

Recover and audit the exact published patch or executable recipe; code availability
alone does not prove that the claimed patch is released. Reuse our pinned Qwen
checkpoint only after confirming patch compatibility. Evaluate original, patched
and no-thinking controls on identical item IDs, recording token limits, truncation,
sampling settings and question-level uncertainty. Keep replay of a known patch
separate from autonomous rediscovery. Do not give the patch recipe to editors in
the discovery comparison.

## Smaller-model translation feasibility

[Gemma translation source](https://github.com/ValerioDolci/frontier-weight-edit)
demonstrates selective tense preference changes, but uses a fitted ridge gate.
That implementation is outside our strict protocol. The published 12B setup also
does not directly fit our 16GB GPU setup.

Screen a predeclared, small set of smaller public checkpoints for the actual
contextual preference issue. Require independently reviewed target labels,
completed-event counterexamples, and broad translation preservation examples.
Both tense forms can be valid: describe the requested preference precisely.
If no repeatable failure exists, record that and stop this track. Do not manufacture
incorrect labels or keep changing checkpoints until a council wins.

## Council comparison contract

- Council includes OpenAI, Claude and Gemini. Use the same base checkpoint,
  visible history, DEV examples, edit scope and inference interface for all arms.
- Match total forward examples and dollars per arm, record actual cost, and
  equalize final-candidate selection opportunities. Tokens alone are not an
  equal-cost comparison across providers.
- Compare against each constituent solo editor and a no-communication portfolio
  where feasible. A win over one weak solo does not establish a general council
  advantage; a win over a portfolio is needed to attribute value to collaboration.
- Use repeated independent editor sessions and a fixed analysis plan. Pair
  external examples, cluster where needed, and account for multiple planned
  task/arm comparisons. Report both effect size and uncertainty.
- Require real nonzero patches after storage-dtype rounding, immutable candidate
  identities, DEV-only selection and one frozen confirmation. Audit all tools
  before external evaluation. A budget-limited incomplete run is not no-fix-found.
- Predeclare preservation margins and test noninferiority with uncertainty;
  failure to detect a significant regression is not proof of preservation.

## Readiness and execution order

Research and source-code inspection complete; no follow-up baseline or editor
comparison has run yet. Public artifact downloads, patch provenance, fresh split
manifests, scorer implementation and CPU validation remain before GPU execution.

1. Finish/audit FinQA independently.
2. Prepare Qwen replay and ResNet fixtures; verify baselines and fixed recipes.
3. Admit ResNet to the hard-case comparison only if its baseline and strict
   runner reproduce correctly. Keep Whisper as a second positive-control option.
4. Screen smaller translation checkpoints after these controls are sound.
5. Freeze concrete run configurations and allocate from verified remaining
   provider budgets before launching paid sessions. Do not spend FinQA's reserved
   allowance on these preparations or infer an unlimited budget from approval.
