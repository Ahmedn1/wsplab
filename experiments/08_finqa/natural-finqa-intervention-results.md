# FinQA follow-up with required edits: no fix found

Plan: [natural-finqa-intervention-plan.md](natural-finqa-intervention-plan.md).
Model: unmodified Qwen3-0.6B, original FinQA prompt, thinking on.
Council: OpenAI (terra) + Claude + Gemini peers. Solo: Claude.

All four sessions finished normally and tested every required candidate edit
(2 per peer, 6 for solo). **No edit survived its HOLDOUT confirmation**, so every
session restored the original weights. Both arms returned a checkpoint that is
byte-identical to the base (SHA256 of every safetensors file). Following the
plan, the run is closed as `no_fix_found` by weight identity. The 95-question
external set is still unused.

## Candidates (32 DEV questions, one sample each, paired with the original)

| Session | Candidate | Weights changed | DEV gain (W→R / R→W) | HOLDOUT gain |
|---|---|---:|---:|---:|
| Peer OpenAI | attention head ×0.5 | 131,072 | 0.0 (3/3) | — |
| Peer OpenAI | same head ×0 | 131,072 | +3.1 (7/6) | **−9.4** (3/6) |
| Peer Claude | "wait" neurons ×0.5 | 16,384 | −6.3 (2/4) | — |
| Peer Claude | same neurons ×0 | 16,384 | +6.3 (5/3) | **0.0** (2/2) |
| Peer Gemini | 13 late MLP neurons ×0 | 13,312 | 0.0 (2/2) | — |
| Peer Gemini | same ×0.5 | 13,312 | 0.0 (2/2) | not run (no positive gain) |
| Solo Claude | C1–C6 | 16k–131k | −9.4 to +9.4 | C3: **−3.1** (3/4) |

Paired standard errors on DEV were 6–11 points, and every DEV winner failed
HOLDOUT. That is what selection on noise looks like.

## Spend

| Arm | Spent | Cap |
|---|---:|---:|
| Council | $1.56 (OpenAI $0.15, Claude $1.03, Gemini $0.38) | $7.29 |
| Solo | $1.29 | $7.29 |

This run cost $2.85. Including both earlier FinQA attempts, the total is
**$7.99** of the $24 approved for FinQA.

## Audit

`.wsplab/finqa-intervention-comparison/protocol-audit.json` (all sessions valid).
I read every executed Python block and forward request, screened them for
training, model replay, fitting and access to private data, then reviewed each
analysis block by hand. Findings:

- Gemini did not replay the network this time. It weighted recorded MLP
  activations by `down_proj` column norms, which is weight inspection.
- Gemini picked 13 neurons out of about 86k using a t-statistic over 32 items.
  That is a multiple-comparison screen likely to find noise, but it breaks no rule.
- Every forward request was DEV or HOLDOUT generation. There were no
  teacher-forced or synthetic screens.
- HOLDOUT was used only for the one frozen confirmation of each DEV winner.

## What this tells us

1. **This setup cannot detect realistic edit effects.** With 32 items and one
   sample each, an edit has to move accuracy by about 15–20 points before DEV can
   tell it apart from sampling noise. The editors were choosing among
   indistinguishable candidates. Any future FinQA-style test needs far more DEV
   items, or several samples per item, before a council comparison means anything.
2. **The candidates were generic hypotheses**: scale an attention head, damp
   "wait" or "error" neurons. None targeted a specific mechanism, such as
   reading the wrong row or column, that the error review had identified.
3. Both arms returned identical models, so this says nothing either way about
   councils versus a solo editor.
