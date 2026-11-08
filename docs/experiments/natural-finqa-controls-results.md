# FinQA prompt and evidence controls

Status: **completed**. No paid calls or weight edits.

Original-label numeric accuracy is primary. The 24-item audit sensitivity is post hoc, not a new benchmark. Gold evidence is diagnostic only.

| Condition | Completed | All 32 correct | All 32 accuracy | Sensitivity correct | Unparseable | Truncated |
|---|---:|---:|---:|---:|---:|---:|
| original_nonthinking | 32/32 | 4 | 12.5% | 4/24 | 3 | 0 |
| original_thinking | 32/32 | 17 | 53.1% | 17/24 | 1 | 0 |
| clean_nonthinking | 32/32 | 7 | 21.9% | 6/24 | 2 | 0 |
| clean_thinking | 32/32 | 8 | 25.0% | 8/24 | 0 | 0 |
| evidence_thinking | 32/32 | 10 | 31.2% | 9/24 | 0 | 0 |

Partial counts are progress only. See [the plan](natural-finqa-controls-plan.md) and [the audit](natural-finqa-audit.json).
