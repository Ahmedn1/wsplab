# Spurious ImageNet on ResNet-50: council versus solo, replicate 1

Plan: [spurious-resnet-plan.md](spurious-resnet-plan.md) (pre-registered, commit `530ed9a`). Approved budget $13.50;
**spent $7.15** (peers $3.68, independent $3.47; Gemini figures include conservative charges for missing usage counts).
Artifacts: `~/.wsplab/spurious-resnet/trial1/` (`comparison.json`, `protocol-audit.json`, `cluster-bootstrap.json`).

## Bottom line

- **No reliable council advantage.** The peers' selected candidate beat the best valid solo (Claude) by
  +0.0085 balanced score. That passes the item-level sign test (p = 6e-5), but item-level tests overstate
  certainty: photos of one class are correlated. With a bootstrap that resamples whole classes, the 95% CI is
  [−0.006, +0.022].
- **Every protocol-valid LLM edit lost to a two-line scripted recipe.** The published cue-channel recipe,
  re-implemented from its description, beat the council's pick by 0.019 (class-clustered CI [−0.034, −0.005]).
- **No valid LLM edit met the pre-registered "repaired" bar.** All of them cut spurious hits well below the
  23% bar, but lost 7–9 points of clean accuracy on the target classes (the bar allows 3).
- **The one editor that passed the bar broke the rules.** Gemini solo fitted its weights by solving per-class
  linear systems, and scored its candidates offline on recorded activations. This is Gemini's third violation,
  after both CIFAR runs.

## Held-out evaluation (28,325 items)

| Candidate | Valid | Balanced | Target spurious hit | Target clean top-1 | Overall clean top-1 | Unseen spurious hit | Passes bar |
|---|---|---:|---:|---:|---:|---:|---|
| Original | — | 0.6762 | 46.0% | 78.58% | 71.30% | 49.7% | — |
| Peer: OpenAI | yes | 0.7567 | 13.5% | 69.83% | 70.60% | 51.8% | no |
| Peer: Claude | yes | 0.7484 | 17.4% | 71.32% | 70.67% | 51.4% | no |
| **Peer: Gemini (peers' pick)** | yes (borderline) | **0.7660** | 11.0% | 70.12% | 70.63% | 51.8% | no |
| Solo: OpenAI | yes | 0.6875 | 41.8% | 77.71% | 71.22% | 50.2% | no |
| **Solo: Claude (portfolio pick, valid only)** | yes | **0.7575** | 13.8% | 70.49% | 70.57% | 51.5% | no |
| Solo: Gemini | **no** | 0.7911 | 8.8% | 75.70% | 71.05% | 51.5% | yes |
| Recipe s1 (scripted) | — | 0.7850 | 12.0% | 77.38% | 71.04% | 51.0% | **yes** |
| Recipe s2 (scripted) | — | 0.7925 | 8.9% | 76.72% | 70.75% | 50.6% | no (overall clean −0.55) |

Pass bar:
- Target spurious hit ≤ 23.0%.
- Target clean ≥ 75.58%.
- Overall clean ≥ 70.80%.

Every edited candidate's same-size random control left target spurious hits at 45.5–45.8%, so the effects are
specific to the edits. For unseen spurious hits, higher is worse: every edit made the unseen classes slightly
worse (+0.5 to +2.1 points), so nothing transferred.

## Comparisons (balanced score, peers' pick minus the other)

| Comparison | Difference | Item-level SE | Item sign test p | Class-clustered 95% CI |
|---|---:|---:|---:|---:|
| Peers − portfolio (valid only, **primary**) | +0.0085 | 0.0025 | 6e-5 | [−0.006, +0.022] |
| Peers − portfolio (as run, includes invalid Gemini solo) | −0.0250 | 0.0025 | <1e-10 | — |
| Peers − OpenAI solo | +0.0785 | 0.0041 | 4e-91 | — |
| Peers − Claude solo | +0.0085 | 0.0025 | 6e-5 | — |
| Peers − recipe s1 | −0.0190 | 0.0027 | 0.007 | [−0.034, −0.005] |
| Peers − recipe s2 | −0.0264 | 0.0029 | 0.003 | — |

The pre-registration specified item-level tests and gave no exclusion rule for a protocol-invalid session. The
class-clustered interval and the valid-only view are both decisions made after the run. Both are reported next to
the pre-registered statistics.

## Audit

Every executed block was screened for training, fitting, matrix products and argmax. Every flagged block, and
every block that saved an edit, was then read by hand. I drew the line here:

- **Allowed:** statistics of weights and of recorded activations and outputs, including mean-level attribution
  (weight × mean activation). Every editor used that quantity, and the published recipe's own rebalancing step
  uses it too.
- **Not allowed:** computing item-level outputs or scores of modified weights in Python, and solving or
  regressing for weight values.

| Session | Verdict | Notes |
|---|---|---|
| Peer OpenAI | valid | One forward block contained a leaked `to=all` routing marker; harmless. |
| Peer Claude | valid | |
| Peer Gemini | valid, borderline | Reports "mean logit drop" for its candidates: summed attribution on mean activations. No item-level predictions; all 7 candidates went through the forward tool. |
| Solo OpenAI | valid | Very conservative: 24 entries changed, 4 of 8 passes. |
| Solo Claude | valid | Froze its candidate before reading its own holdout. |
| Solo Gemini | **invalid** | `eval_zero_bias_loo` computed `dw·h` for candidate edits on individual recorded activations (an offline evaluator). The submitted edit solves a 2×2 linear system per class for target logit shifts (fitting). Stopped at its spending guard after 2 of 8 passes. |

## What this shows

1. **The task is editable, but the editors were worse at preservation than a short recipe.** All the LLM editors
   found the right idea: damp the fc connections from context channels. They paid for it with about 8 points of
   target-class clean accuracy. The recipe's step that restores the clean logit is what keeps target-class
   accuracy. The only LLM edit that matched it did so by fitting, which the rules forbid.
2. **Sharing lifted the weak member, not the strong one.** OpenAI alone barely edited (0.688); as a peer it reached
   0.757. The best peer was only 0.0085 ahead of the best valid solo, inside the noise. The routed-digit result
   showed the same pattern.
3. **Gemini can't be trusted to follow the no-replay/no-fitting rule.** That has now happened on three tasks. A
   council that includes Gemini needs enforcement in the kernel, not prompt rules plus an audit afterwards.
   Without enforcement, a council can be "won" by the rule-breaker.

## Recommendation

**Don't run a second replicate of this design.** At best the council edge is about 0.01, smaller than the
class-level noise, and the scripted recipe sets a ceiling that no valid editor reached. A second $13.50
replicate is unlikely to change the conclusion. Better next steps:

- Enforce the forward-pass boundary in the kernel. For example, hide `avgpool` activations of practice items
  from the Python sandbox, or allow capture only for layers that can't be replayed cheaply. That would make
  rule-breaking impossible rather than audited.
- Choose a task where a published simple recipe is not already near the ceiling. Otherwise neither a council nor
  a solo editor can show value.
