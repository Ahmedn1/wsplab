# FinQA control review: original prompt remains strongest

Completed all 96 planned local generations on the original Qwen3-0.6B and the
same 32 pilot questions. No paid API calls, weight edits, new sampled questions,
or external evaluation were used. The benchmark process and its engine exited;
the host GPU process query showed no remaining compute process.

The simplified prompt did not improve the strongest baseline. Annotated evidence
improved the simplified thinking condition by only two net answers, with substantial
regressions as well as fixes. This is not evidence of a peer-council advantage.

## Results

Primary scores use the original published display answers and the corrected
custom numeric parser, not official FinQA program/execution accuracy.

| Condition | Correct / 32 | Accuracy | Secondary sensitivity / 24 | Unparseable |
|---|---:|---:|---:|---:|
| Original, nonthinking | 4 | 12.5% | 4 | 3 |
| Simplified, nonthinking | 7 | 21.9% | 6 | 2 |
| Original, thinking | **17** | **53.1%** | **17** | 1 |
| Simplified, thinking | 8 | 25.0% | 8 | 0 |
| Simplified + annotated evidence, thinking | 10 | 31.25% | 9 | 0 |

No response was truncated. The eight reference-sensitive cases stay in every
generation condition and in the primary score. The other 24 form a post hoc
sensitivity analysis, frozen before these new generations. All eight exclusions
were original-thinking failures; 17/24 is not an unbiased replacement benchmark.
The [audit](natural-finqa-audit.md) explains the reference and interpretation issues.

## Paired comparisons

Each row compares outcomes on the same 32 questions. Changes and standard errors
are percentage points. Exact p values use the two-sided binomial test on discordant
pairs (exact McNemar test).

| Change | Wrong to right | Right to wrong | Accuracy change | Paired SE | Exact p |
|---|---:|---:|---:|---:|---:|
| Simplified vs original, nonthinking | 6 | 3 | +9.38 | 9.38 | .508 |
| Simplified vs original, thinking | 0 | 9 | -28.13 | 8.08 | .00391 |
| Evidence vs full context, both simplified thinking | 6 | 4 | +6.25 | 9.98 | .754 |
| Evidence + simplified vs original thinking | 3 | 10 | -21.88 | 10.75 | .0923 |

These are exploratory comparisons at one sampling seed on 32 questions, without
multiple-comparison adjustment. They do not establish effects across seeds or on
untouched examples. The last row changes both instructions and context, so it
cannot isolate the effect of evidence selection on the original prompt.

On the 24-item sensitivity set, evidence versus simplified full context has five
fixes and four regressions (+4.17 points; exact p = 1). This does not strengthen
the evidence-selection hypothesis. Full per-item outcomes and both sets of paired
statistics are in [the analysis](natural-finqa-controls-analysis.json).

## What changed in the answers

- Two nonthinking improvements merely put already-correct numbers into the
  required answer format (SPGI/2015 and PM/2017). The six scored fixes cannot all
  be described as reasoning improvements.
- Simplified thinking loses nine answers without gaining any. These include
  wrong columns/denominators, arithmetic/sign errors, and coarser rounding. Its
  lower score is not caused by output truncation or more parsing failures.
- Evidence recovers the AAPL/2008 calculation `(19 - 3 - 1) / 3 = 5` and the
  UNP/2013 revenue calculation `2100 / 21963 * 100 = 9.5615...`. However, AAPL's
  trace still incorrectly describes the dollar-valued result as percentage units.
- The third scored gain over original thinking, ADI/2011, follows the flawed
  annotation: it substitutes money-market funds for the requested mutual funds.
  Matching the reference here is not better reasoning. The all-item score retains
  this match; the already-fixed sensitivity subset excludes the case.
- AOS/2004 becomes correct under the scalar tolerance despite adding the purchase
  obligations total, 177.3, to a component, 0.7. The resulting 34.14% happens to
  match the whole-percent reference of 34%. A scored answer is not sufficient
  evidence of a correct derivation.
- Evidence also introduces errors despite retaining the needed cells and headers:
  AES/2002 selects 2001's fourth-quarter prices; IPG/2008 mixes a component with
  the total; SPGI/2015 subtracts an already-negative liability and overstates equity.
  IPG/2015's cash-flow sign error persists in all three thinking conditions.

## Decision and remaining uncertainty

Keep original thinking at 17/32 as the strongest observed baseline. Do not present
8/32 to 10/32 as a council win, or weaken the baseline to make a future comparison
look better. No council participated in these controls.

These results do not currently justify a paid FinQA council run. They also do not
prove that weight editing cannot help, or that context selection never helps.
The evidence condition uses dataset annotations, including some bad ones, and is
not a deployable retrieval system.

If continuing this diagnostic, the next missing comparison is **annotated evidence
with the original prompt**, keeping this completed report unchanged. That fourth
thinking-condition cell would separate a prompt/context interaction from the
current evidence result. A promising pilot result would still need repeat-seed and
untouched-question confirmation under a frozen scoring protocol before serving as
the basis for an OpenAI/Claude/Gemini council versus an equal-budget solo test.
Do not keep changing prompts or tasks until a favorable pilot appears.

Subsequent direction from the user superseded that proposed prompt follow-up:
test direct weight edits rather than require a successful prompt diagnostic.
The [direct-edit pilot](natural-finqa-weight-edit-results.md) now reports the
completed bounded intervention, keeping this control experiment unchanged.

## Verification and artifacts

The nine targeted control/scoring tests and lint passed before launch. Final
verification confirmed all five conditions have exactly the same 32 IDs, all
three source snapshots match their execution hashes, and the runner's final
fixture/model verification completed successfully. Raw original responses and
scores were preserved.

- [Frozen control plan](natural-finqa-controls-plan.md)
- [Final generated results](natural-finqa-controls-results.md)
- [Reference audit](natural-finqa-audit.json)
- [Paired and per-item analysis](natural-finqa-controls-analysis.json)
- Local execution artifacts: `.wsplab/natural-finqa/controls/`
