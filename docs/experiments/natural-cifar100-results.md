# Genuine-task comparison: pretrained ResNet-20 on CIFAR-100

Status: both arms and the external evaluation are complete. **No council win.**
The selected peer candidate regressed numerically and its search was
protocol-invalid. This is a completed diagnostic experiment, not a valid
demonstration of peer advantage or a definitive comparison of model capability.

## The task is naturally difficult

This is the unmodified public `cifar100_resnet20-23dac2f1.pt` checkpoint from
[chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models),
with 278,324 parameters. There are no planted faults or corrupted inputs.
Ordinary top-1 classification across 100 real image categories reproduces the
author's **68.83%** accuracy on the full 10,000-image test set exactly.

The fixed seed-20261106 split uses 1,000 images for editor practice, 1,000 for
candidate selection and 8,000 for external evaluation. Original-model accuracy
on the external split is **68.875%**. No final score is used to choose candidates.
Under the predefined operational rule, an edit that fails to beat the original
on selection data falls back to the original weights. Raw candidate scores are
also retained, separately from this rule.

See the [original plan](natural-cifar100-plan.md),
[approved corrected plan](natural-cifar100-restart-plan.md), and
[baseline and activation verification](natural-cifar100-baseline.json).

## Frozen external results

All values below are percentages. Selection uses 1,000 images; external accuracy
uses the 8,000 reserved images. The baseline selection score is 70.700%.

| Arm / editor | Selection | External | Changed entries | Forward passes |
|---|---:|---:|---:|---:|
| Original pretrained model | 70.700 | 68.8750 | 0 | — |
| Peers / OpenAI Terra | 70.600 | 68.7625 | 1 | 1 |
| Peers / Claude | 70.700 | 68.8750 | 0 | 8 |
| **Peers / Gemini — selected** | **71.100** | **68.5250** | **36,864** | **2** |
| **Independent / OpenAI Terra — selected** | **70.800** | **68.8000** | **1** | **8** |
| Independent / Claude | 70.700 | 68.8750 | 0 | 5 |
| Independent / Gemini | 70.700 | 68.8750 | 0 | 1 |

The selected peer candidate is **0.350 percentage points below the original**
(paired SE 0.148 points; 56 corrections, 84 newly wrong answers). It is
**0.275 points below the selected independent candidate** (paired SE 0.151
points; 62 better, 84 worse). The framework's raw verdicts are `regressed`
for peers and `not_significant` for independent search. These are numerical
verdicts; they do not certify protocol compliance.

Both selected edits exceeded the baseline on selection data, so the predefined
fallback rule retains them and the operational arm scores equal their raw
scores. Choosing Claude's unchanged model after seeing external results would
be hindsight selection. The individual unchanged submissions remain useful
diagnostics: neither selected edit beat simply retaining the original model.

Each arm had the same three editors (`gpt-6.1-terra`, `claude-opus-5-5`,
`gemini-4-argon`), high reasoning effort, eight permitted forward passes per
editor, and the same provider caps. Actual spending and pass use differed.
This is one fixed attempt per arm, with no retry chosen for favorable results.
All six submitted checkpoints were verified bit for bit against the frozen
public baseline plus their recorded patches, including untouched tensors.

## Protocol findings

- Gemini's peer actually executed modified-network simulations in Python.
  Zero-based tool records 20–23 sweep bias shifts, run convolutions with altered
  weights, apply ReLU, and measure the resulting feature projections. These are
  counterfactual forward computations outside the permitted tool. Record 18
  also reconstructs a convolution from recorded upstream activations.
  The printed results confirm execution; this was not merely proposed code.
- The shared transcript exposed those computations to the other peers. Claude
  explicitly rejected some of them, but that does not certify the whole peer
  search as clean. Its outcome cannot establish a compliant council advantage.
- Terra's peer saved a one-weight edit and ended after one forward pass without
  testing it. The final answer explicitly acknowledges the missing validation.
- Gemini's peer stopped at its conservative request-budget guard after two
  forward passes. It was not a completed eight-pass attempt.
- Gemini's independent session also stopped at that guard, after 37 turns,
  29 Python executions and one original-model forward pass. It saved no edit.
  Its trace includes residual-activation reconstruction and first-order margin
  sensitivity sweeps (notably records 15 and 22–29). Those are a method caveat;
  this report does not certify them as a clean optimization procedure. The
  unchanged submission's measured score is nevertheless exactly reproducible.
- Claude's peer used eight passes, tested six candidate mechanisms, and restored
  the original weights. Claude's solo also restored the original after five
  passes. This supports only a failure to find improvement in those attempts,
  not the models' stronger assertion that improvement without training is impossible.
- Some internal practice HOLDOUT scores were exposed during development.
  This weakens their internal confirmation claims. The framework selection and
  external evaluation sets remained separate from the editor kernels.
- Both arms' stock prompt said all eight passes were required, but their driver
  used `require_all_passes=False`. The runtime permits early termination. This
  prompt-policy mismatch must be acknowledged when interpreting the early stops.
  Future built-in prompts have now been corrected to describe optional-pass
  policies accurately. This change was made after all six sessions had started;
  it does not change their archived prompts or repair this comparison retroactively.

The earlier attempt was stopped for a separate harness bug: activation records
shared storage with tensors later overwritten by in-place operations. That
recorder is fixed and verified against an independent reference for all 21
batch-normalization outputs, with zero activation or final-logit discrepancy.
The earlier attempt remains invalidated; its charges are kept separately.

## Accounting and raw evidence

The corrected comparison's guarded estimate is **$9.3450**, below its $14
ceiling: **$5.2463 peers**, **$4.0987 independent**. By provider, totals are
$0.3018 OpenAI, $2.5082 Anthropic and $6.5350 Gemini. The provider ledgers record
no API errors. There are 23 calls with estimated usage (14 peers, 9 independent).
Missing Gemini output usage is charged at the requested maximum output
allowance rather than counted as free. This bound assumes the provider honors
that allowance. All costs are local estimates, not provider invoices.
The prior invalidated attempt recorded $0.8872 in completed calls and has a
conservative total of $1.8537 including interrupted-call reservations.

Raw plans, ledgers, transcripts, states and checkpoints are in
`.wsplab/experiments/natural-cifar100-corrected/`.
`interim-audit.json` preserves findings made before inspecting external scores;
`audit.json` and `checkpoint-verification.json` contain the completed review.
The [machine-readable comparison](natural-cifar100-results.json) retains all
candidate scores, paired differences, stopping reasons, cost estimates and audit
findings. Both paid workers and the scoring watcher have exited.

No additional paid attempt is authorized by this report. Stronger prompt wording
alone did not enforce the forward-pass boundary. Before another comparison,
address simulation rejection and Gemini usage reporting, and use the corrected
stop-policy prompt. Do not rerun merely to obtain a favorable score.
