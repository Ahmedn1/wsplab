# MGSM control results

Automated completion snapshot: **controls_complete_manual_review_pending**.

**Follow-up: manual review is now complete.** See the [final review](natural-mgsm-controls-review.md) and [64 item judgments](natural-mgsm-translation-review-gpu.json). Recommendation: defer a paid council experiment on this candidate.

This is the frozen pilot, not a council experiment or external test. No paid API calls.

Difficulty-rule checks are screening criteria, not proof of editability. Translation/comprehension review remains necessary.

## cpu_feasibility

Process status: `completed`.

| Condition | English | Bengali | Swahili | Truncated / responses | Numeric difficulty rule |
|---|---:|---:|---:|---:|---|
| english_reason | 6/8 | 2/8 | 0/8 | 0/24 | not_assessed |
| translate | pending | 8 translations | 8 translations | 0/16 | not_assessed |

## gpu_full_pilot

Process status: `completed`.

| Condition | English | Bengali | Swahili | Truncated / responses | Numeric difficulty rule |
|---|---:|---:|---:|---:|---|
| nonthinking_pp | 24/32 | 4/32 | 1/32 | 0/96 | passes_numeric_rule |
| english_reason | 26/32 | 3/32 | 1/32 | 0/96 | passes_numeric_rule |
| localized | 23/32 | 4/32 | 1/32 | 1/96 | passes_numeric_rule |
| thinking_pp | 27/32 | 7/32 | 0/32 | 0/96 | passes_numeric_rule |
| nonthinking | 25/32 | 3/32 | 2/32 | 2/96 | passes_numeric_rule |
| thinking | 27/32 | 7/32 | 1/32 | 0/96 | passes_numeric_rule |
| translate | pending | 32 translations | 32 translations | 0/64 | not_assessed |

CPU feasibility uses short outputs and FP32; full GPU controls use BF16 and the longer allowance.
Compare controls within their backend. Translations are not assigned arithmetic scores.

See [the frozen controls plan](natural-mgsm-controls-plan.md) and the adjacent JSON for per-condition diagnostics.
