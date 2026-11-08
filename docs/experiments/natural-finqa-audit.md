# FinQA pilot reference and error audit

Scope: the existing 32 pilot questions only; no practice, selection or external
examples were inspected. The assistant reviewed the source tables/supporting
text, published answers/programs and original thinking traces. The review is
not blinded or independently adjudicated. Full item evidence and calculations
are in [audit.json](natural-finqa-audit.json), frozen before the new controls.

## Measurement findings

The original corrected numeric scores remain **4/32 nonthinking** and **17/32
thinking**. Do not relabel these as official FinQA execution scores.

- **Mutual funds reference error:** ADI/2011 asks about mutual funds, whose row is
  9,223 out of 26,410 (34.922%). The reference instead uses the 17,187 money-market
  funds row (65.078%). The model selects the requested mutual-funds row and
  approximates to 35%; that is correct row/formula selection, not exact agreement
  at the reference's one-decimal precision.
- **Rounding reference issue:** AON/2009 gives 22.55%, while 203 / 900 * 100 is
  22.555...%, rounding to the model's 22.56%. The original tolerance rejects it.
- **Bad execution answer but supported display answer:** UNP/2013 gives 9.6%
  in text and .0001 in execution output. Converting 2.1 billion to 2,100 million
  gives 2,100 / 21,963 * 100 = 9.5615%. Keep the displayed-answer target. The model
  still chooses the wrong denominator and gives 10.3%.
- **Scale/interpretation concerns:** MRO's gold is in thousands of dollars without
  labeling that scale; HII omits the backlog year; SPGI/2018 does not specify the
  US or UK pension trust; NWS's reference adds a stated decrease. L/2015's word
  “reduction” permits a positive magnitude, while the label uses signed change.
- **Approximation concern:** PM/2014 asks approximately. The model's 3,100 versus
  the exact sum 3,098 may be acceptable as an approximation, though its reasoning
  incorrectly asserts that the exact sum is 3,100.

These identify eight reference-sensitive cases. None are deleted from generation
or the primary score. A fixed secondary analysis excludes them and retains 24
questions: nonthinking 4/24, thinking 17/24. All eight exclusions were original
thinking failures, so the resulting 70.8% is a **post hoc sensitivity result**, not
an independent benchmark estimate or an improvement. This concentration also
shows why treating the original 53.1% as pure reasoning accuracy is misleading.

The displayed-answer precision rule can accept modest numerical approximations
(e.g. 33.8% against a whole-percent gold of 34%). It is useful for reproducing
rounded human answers, but does not prove exact arithmetic. Original metrics,
raw predictions, tolerances and parser corrections remain available.

## Residual thinking errors on the 24-question sensitivity set

Seven original failures remain. Categories describe observed traces, not proven
causal neural mechanisms.

| Pilot item | Observed issue |
|---|---|
| HWM/2016 | Compares each quarter to the annual high instead of Q2 versus Q1; repeats concerns about the prompt's numeric example. |
| LMT/2016 | Omits the 2014 column and averages only 2015/2016. |
| MS/2014 | Selects tax-position increase 299 instead of accrued interest 258; fails to form the requested ratio. |
| AAPL/2008 | Reads all three annual values as 19 rather than 19, -3, -1. |
| BLK/2017 | Includes 2022 despite “after 2022”, then miscalculates the resulting ratio. |
| UNP/2013 | Uses freight revenue rather than total operating revenue as denominator and calculates imprecisely. |
| IPG/2015 | Treats increased cash outflow as a positive cash-flow effect. |

There is evidence of table/evidence selection difficulty as well as operation and
sign errors. Arithmetic alone does not account for every failure. Only the new
matched-context controls can test whether supplying evidence actually helps.

## Controls

The [fixed three-condition plan](natural-finqa-controls-plan.md) removes only the
numeric illustration, tests both thinking modes, and then gives original annotated
source evidence with thinking. It does not feed answer labels or programs. Some
annotations are themselves wrong; the all-item and sensitivity results must be
reported together. No council or weight-edit claim follows from an oracle gain.
