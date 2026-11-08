# MGSM controls: final review

All seven GPU pilot conditions are complete: 576 math responses and 64
translation probes on 32 frozen problem groups. The CPU feasibility checks are
also complete. No paid provider calls, weight edits or council comparison were
performed. The 114-group external partition remains unused.

**Decision: defer a paid council experiment on this task.** This is a genuine
model weakness, but the pilot provides little positive evidence of a removable
obstacle hiding intact language comprehension. This is a resource-allocation
judgment, not proof that weight editing cannot help.

## Complete math controls

Each cell contains correct answers out of 32. All settings use the original
Qwen3-0.6B checkpoint and the same pilot groups. GPU results use BF16 and allow
up to 32,768 output tokens within the context limit.

| Condition | English | Bengali | Swahili | Truncated / 96 |
|---|---:|---:|---:|---:|
| Nonthinking + presence penalty | 24 | 4 | 1 | 0 |
| English reasoning + presence penalty | 26 | 3 | 1 | 0 |
| Localized instruction + presence penalty | 23 | 4 | 1 | 1 |
| Thinking + presence penalty | 27 | 7 | 0 | 0 |
| Nonthinking, no penalty | 25 | 3 | 2 | 2 |
| Thinking, no penalty | 27 | 7 | 1 | 0 |

All six conditions pass the previously specified numeric difficulty rule. The
language gap persists across these particular prompting and decoding controls;
this does not establish that all possible prompts fail. Bengali reaches 21.9%
with thinking, versus 84.4% English. Swahili remains at 0–6.25%. Near-zero
truncation makes output exhaustion an unlikely main explanation for this pilot.
The conditions vary decoding settings as well as thinking behavior, so their
differences should not be attributed to a single mechanism.

The best Bengali count is exploratory, selected among six conditions, and is
not an externally confirmed improvement. Translations of the same problem are
correlated; the automated report clusters paired uncertainty by problem.

## Full GPU translation review

All 64 translation outputs finished without truncation. The assistant manually
compared each output with its published English counterpart, checking essential
quantities, mathematical relationships, units and requested unknown. The general
rubric was saved before the earlier CPU review; these full-GPU item judgments
were authored after viewing the outputs, not independently preregistered.

| Language | Review outcome | Count |
|---|---|---:|
| Bengali | Mathematically usable English rendering | 4/32 |
| Bengali | Ambiguous mathematical equivalence | 5/32 |
| Bengali | Essential meaning changed or missing | 23/32 |
| Swahili | Mostly untranslated source text | 21/32 |
| Swahili | Corrupted mixture of English and source language | 11/32 |

The four usable Bengali outputs preserve the arithmetic but may change names
or object categories. They are not certified faithful translations. Even
counting all five ambiguous outputs as usable would give only 9/32. No Swahili
output supplied a usable English rendering under this rubric.

Examples of substantive errors include changing four sandcastle levels to two,
dropping 50 pounds per bag, changing tax on nonfood items into a discount, and
changing an answer requested in hours into minutes. These errors undermine the
hypothesis that only downstream arithmetic is blocked.

However, copying Swahili instead of translating it may reflect instruction
following rather than absent language knowledge. One translation sample per
item, judged by the assistant without an independent native-speaker review,
cannot distinguish these explanations conclusively. This is a diagnostic
review, not a new benchmark score or proof of missing latent capability.

All item judgments, references, outputs and source hashes are saved in the
[GPU review JSON](natural-mgsm-translation-review-gpu.json). The separate
[earlier CPU review](natural-mgsm-translation-review.json) found 2/8 usable
Bengali renderings and 0/8 Swahili; CPU and GPU samples must not be pooled as
identical stochastic measurements.

## Implication for the council experiment

Do not spend on this candidate solely because its baseline is low. The pilot
establishes difficulty but does not identify a plausible repair mechanism or
show any advantage from collaboration. A better next candidate would combine
low accuracy with evidence that the required capability is already present,
while preserving a strong solo and cheap-remedy comparison.

A future council comparison should still include Gemini alongside OpenAI and
Anthropic, use matched resources and a frozen selection rule, and retain an
untouched external evaluation. No new paid run is launched by this review;
the previous monetary authorization was specifically for CIFAR-100.

The [automated results](natural-mgsm-controls-results.md) preserve the monitor's
completion snapshot, which said manual review was pending. This document and
the GPU review JSON complete that review. No inference job remains from this
pilot.
