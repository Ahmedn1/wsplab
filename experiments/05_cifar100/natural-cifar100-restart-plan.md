# Corrected genuine-task comparison: approved

The first attempt is invalidated, with all transcripts and accounting preserved
under `.wsplab/experiments/natural-cifar100-20261106/`. It was stopped because
the generic Torch activation recorder kept numpy views of tensors. Later
in-place ReLU and residual additions overwrote earlier module outputs. This
could suggest nonexistent batch-normalization defects to an editor. The public
weights and baseline inference were never damaged; the bug affected observations.
The three peers and terra solo had started; the other solos had not started.
No peer-versus-solo performance result is claimed from this attempt.

Fix: snapshot module outputs at hook time with an independent copy. Regression
tests cover both in-place ReLU and residual additions. The real public ResNet's
21 batch-normalization outputs are checked against an equivalent reference with
out-of-place operations. Both activations and final logits must match exactly.
The original 8,000-image baseline must remain 68.875%.

The restarted experiment retains the original model, data, split, prompt, three
providers, 8-pass/40-Python/60-turn limits and selection rules from the
[original plan](natural-cifar100-plan.md). All transcripts start fresh because
the old sessions have already seen incorrect observations. The corrected runner
hash is recorded in each arm's plan and checked when comparing results.

Approved maximum new allocation is **$14 total**, $7 per arm:

| Provider | Per arm | Both arms |
|---|---:|---:|
| OpenAI terra | $0.50 | $1.00 |
| Anthropic Claude | $3.00 | $6.00 |
| Gemini Argon | $3.50 | $7.00 |

The invalidated attempt reports $0.8872 in completed calls. Retaining every
in-flight call's entire reservation gives a conservative total of $1.8537.
Actual provider charges for those interrupted requests are unknown. After
including that upper estimate, prior runs, historical missing Gemini usage and
Astra's original $17 OpenAI working allocation, the proposed cumulative upper
estimates are OpenAI $19.53, Anthropic $27.79 and Gemini $20.94. These are spending
estimates, not account balances or invoices; Astra's older ledger checks after
calls and can overshoot its working allocation.

The prior authorized totals were OpenAI $19, Anthropic $25 and Gemini $25.
The user explicitly approved totals of **OpenAI $20 and Anthropic $30**, retaining
Gemini's $25 limit, and instructed the corrected run to proceed. The task,
harness, baseline, scripts and checks were completed before that approval.

After a fresh ledger check, launch each arm in a new directory:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
  ~/.venvs/wsplab/bin/python experiments/code/natural_trial.py \
  --arm peers --fixture .wsplab/natural-cifar100 \
  --out .wsplab/experiments/natural-cifar100-corrected/peers \
  --openai-cap 0.50 --anthropic-cap 3.00 --gemini-cap 3.50
```

Use `--arm independent` and a separate `independent` output directory for the
matching baseline. Preserve failures, stopping conditions, estimates and code
audits; never treat a budget-truncated solo as proof that peers reason better.
