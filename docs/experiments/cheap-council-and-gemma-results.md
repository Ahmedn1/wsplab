# Cheap-model council on Seaquest, and the Gemma math trial

## 1. Can a council of cheap models match a strong solo editor? (Seaquest)

**Setup.** The same Seaquest task, rules, 8 passes, edit budget and 2,000 evaluation episodes as
[seaquest-results.md](seaquest-results.md). The council is gpt-6-luna, Claude Haiku 5.5 and Gemini 3.8 Flash sharing
every step; the other arm is the same three working alone. Approved at most $4.30; **spent $2.66**.

| Session | Valid | Entries changed | Evaluation return | Gain over original |
|---|---|---:|---:|---:|
| Original | — | — | 52.13 | — |
| Cheap peer: Luna | yes | 0 (never ran a pass) | 52.13 | 0 |
| Cheap peer: Haiku | yes | 49 | 3.76 | −48.4 |
| Cheap peer: Flash | **no** | 7 | 6.81 | −45.3 |
| Cheap solo: Luna (portfolio pick) | yes | 1 | 54.99 | +2.9 ± 1.1 |
| Cheap solo: Haiku | yes | 0 | 52.13 | 0 |
| Cheap solo: Flash | **no** | 16 | 48.01 | −4.1 |
| *Strong solo: Opus (earlier run)* | yes | 32 | 75.74 | +23.6 |
| *Strong council (earlier run)* | yes | 32 | 74.80 | +22.7 |

**Answer: no.** The cheap council's pick, after falling back to the original because every edit was worse, sits
**23.6 points below Opus alone**, so it fails the pre-registered "within 3 points" bar by far. Sharing didn't
help the cheap models either. The council's members broke the policy outright (3.8 and 6.8), whereas one cheap solo
made a small real gain (+2.9). Pooling three weak analyses produced confident, wrong edits. The earlier runs suggest
the council's gains come from a weak member adopting a strong member's finding; with no strong member there is
nothing to adopt.

**Audit.** Gemini 3.8 Flash, in both arms, computed full network outputs (logits and action probabilities) for
candidate weights on hand-built game states to choose an edit strength. That is simulation, so both Flash sessions
are disqualified. Their edits were worse than the original anyway, so no pick changes. Luna stopped in the peer arm
without ever running a pass.

## 2. Gemma 3 270M on math (the paper's small-gain case)

**Setup.**
- **Task and protocol:** the paper's Gemma follow-up, where Astra gained 6.8% → 7.0–7.7% on MATH-500. Same model
  revision, v5 prompt, sampling and lenient grading.
- **Editors:** strong OpenAI, Claude and Gemini, as peers and alone, 8 passes each.
- **Selection:** 600 practice problems held out to choose each arm's candidate.
- **Edit budget:** 500,000 entries.
- **Spend:** $8.76 (peers $4.26, independent $4.50), in 2.5 hours.

| Session | Passes used | Entries changed | Selection score (original 10.1%) |
|---|---:|---:|---:|
| Peer OpenAI | 8 | 163,840 | 8.3% |
| Peer Claude | 1 (spending guard) | 0 | — |
| Peer Gemini | 2 (spending guard) | 0 | — |
| Solo OpenAI | 8 | 640 | 8.8% |
| Solo Claude | 5 (spending guard) | 163,840 | 9.0% |
| Solo Gemini | 1 (spending guard) | 0 | — |

**Answer: no fix found by anyone.** Every edit that was submitted scored below the original on held-out problems, so
both arms keep the original model. As planned for identical arms, the run is closed by weight identity, without
spending GPU time on MATH-500. This matches the paper, where no edit moved Gemma by more than about a point.

**Caveat about the caps.** Gemma's long prompt (the paper's v5 template plus answers) made Claude and Gemini hit
their per-session caps after 1–5 passes. The test was therefore underpowered for those two. It does not show that a
fully funded council couldn't find a gain. But OpenAI used all 8 passes in both arms and still found only regressions.

## Across all the council tests

| Task | Council vs best solo | Pattern |
|---|---|---|
| Routed digits (synthetic) | +5.3 points, one seed | Weak member adopted a strong member's repair |
| Spurious ImageNet | +0.009, inside noise | Weak member lifted; all below a scripted recipe |
| Seaquest (strong models) | −0.9 ± 0.8 | Weak member lifted; peers converged on one edit |
| Seaquest (cheap models) | −23.6 vs strong solo | No strong member to copy; peers broke the policy |
| Gemma math | tie (both no fix) | Nobody found an edit |

The consistent finding is that **sharing raises the weakest member toward the strongest, but it hasn't produced an
edit better than the strongest member's own.** A cheap council can't substitute for a strong editor.
