# Live peer pilot: fixed before calls

Exploratory question: does sharing improve modular weight repair over the same
models working independently? This is a difficulty pilot, not confirmation of a
general council advantage. Preserve negative results.

- Fixture: frozen routed-digits-v1, seed 21, same practice/selection/evaluation.
- Members: gpt-6.1-terra, claude-opus-5-5, gemini-4-argon, all high effort.
- Arms: one shared peer trio; one independent trio (three solo sessions).
- Each session: 40 Python runs, 6 forward passes, 60 turns, 32,768 output tokens
  per response. Same early-finishing policy, no handoff, warning before unsaved
  edits exhaust the Python budget. Full peer text, no hidden truncation.
- Per arm estimated spending caps: OpenAI $0.75, Anthropic $2, Gemini $3.50.
  Combined pilot ceiling: $12.50, with conservative request reservations. API
  errors retain their reservation as an uncertain charge. No automatic retries.
- Candidate selection: highest selection score; evaluation only after freezing.
  Report independent candidates' evaluation as a clearly labelled exploratory
  diagnostic to answer how each solo model performed, without changing selection.
- Primary comparison: peer-selected versus independently selected candidate on
  identical evaluation items; paired difference and standard error. Also report
  best solo diagnostic, per-branch scores, turns, passes, spend and cap/failure stops.
- Audit improvements for training, direct network execution outside official
  passes, held-out leakage and peer evidence transfer. A gate verdict of
  `confirmed` only means improvement over the damaged model, not over solos.
- A promising result is at least +5 percentage points over the independently
  selected candidate without budget/provider failure explaining the gap. That is
  a follow-up trigger, not statistical proof. Confirm on fresh construction seeds
  22 and 23, including best-of-three attempts from the strongest solo model, before
  asserting an advantage. No extra paid batch launches automatically.

Prices checked against [OpenAI](https://developers.openai.com/api/docs/pricing),
[Anthropic](https://platform.claude.com/docs/en/about-claude/pricing), and
[Google](https://ai.google.dev/gemini-api/docs/pricing). Anthropic fresh input is
conservatively charged at the cache-write rate because the current adapter combines
fresh and cache-written input tokens. Gemini thinking is included in usage accounting.
Reported spend remains an API-usage estimate, not an account billing statement.

## Guard correction before interpreting scores

The first peer attempt was invalidated: a byte-per-token estimate of the entire
growing conversation triggered context and reservation limits prematurely. Claude's
solo attempt was also stopped by this guard. These runs and their costs remain in
the record; they are not evidence of a reasoning failure.

The corrected guard uses reported input tokens for an unchanged prefix and a
conservative byte bound only for new text. It lowers the response token limit near
the monetary cap instead of always reserving a full 32,768-token response. It still
does not truncate peer evidence. A corrected peer run was launched only after
checking that earlier charges, the current solo's maximum outstanding allowance,
and the new peer cap together remained below $12.50 (upper estimate: $12.214).
Unaffected solo sessions that finish normally can be retained: this guard never
changes their input messages or responses when it does not bind. A capped solo
must be rerun or clearly reported as incomplete, subject to remaining credit.
