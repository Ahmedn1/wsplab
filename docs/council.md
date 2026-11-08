# Councils and search

One editor is the default, and we recommend starting there. A council is optional.
A two-peer search now has one positive comparison against both members working alone
on modular digit repair, but a repeatable or cost-efficient advantage is not established.

## What has been measured

Gemini costs below are too low, and so are the totals that include it: its OpenAI-compatible endpoint leaves thinking out of
`completion_tokens`, and thinking is billed as output. A test call reported 6 output tokens with 1,036 more in `total_tokens`. wsplab now
counts them, but the figures here were measured before that fix.

A digit classifier with two planted faults, 4 forward passes, 3 repeats per arm. Fixing both faults restores the clean score of
95.77%. All arms fixed the first fault every time, so they differ only on the second.

| Arm | Fixed both faults | Mean score | Mean cost |
|---|---|---|---|
| `gpt-6.1-terra` alone | 3 of 3 | 95.77% | $0.09 |
| `gpt-6.1-sol` alone | 2 of 3 | 94.57% | $0.15 |
| `gpt-6-sol` alone | 1 of 3 | 94.43% | $0.67 |
| Council voting on scripts | 2 of 3 | 94.57% | $0.40 |
| Council voting on finished weights | 2 of 3 | 94.57% | $0.99 |

Three repeats on one small task can only show large effects, so none of these differences is established. The clearest reading:
neither council beat its best member, and both cost more. A council's vote did overrule a weaker chair in some runs. Choosing the
editor mattered at least as much as voting, and the most expensive model was not the best one here.

A second comparison added other providers on the same task. Each member of the cross-provider council chaired one run.

| Arm | Reached the clean score | Mean score | Mean cost |
|---|---|---|---|
| `claude-opus-5-5` alone | 3 of 3 | 95.91% | $0.38 |
| `gemini-4-argon` alone | 3 of 3 | 95.77% | About $0.62 |
| `gpt-6.1-terra` alone (from above) | 3 of 3 | 95.77% | $0.09 |
| Claude, Gemini and terra council | 3 of 3 | 95.84% | $0.74 |

The council reached the clean score with every chair, but so did each of its members alone, so this task cannot show whether it adds
anything. Mixing providers did make members disagree more often. Voters also ranked their own proposal first in 8 of 11 cases
despite anonymous ballots, more than chance would give, so anonymity may not fully hide authorship. Both are small samples.

### A harder task

A trained 64-48-48-10 digit classifier with 62 planted faults in its hidden layers, 3 forward passes and 15 Python runs. The damaged
network scores 35.2%, and the clean one 94.6%.

| Arm | Scores | Mean | Mean cost |
|---|---|---|---|
| `gpt-6.1-terra` alone | 47.9, 47.9, 47.9 | 47.9% | $0.09 |
| `claude-opus-5-5` alone | 49.5, 51.1, 46.1 | 48.9% | $0.60 |
| `gemini-4-argon` alone | 49.5, 49.5, 47.7 | 48.9% | $0.69 |
| Cross-provider council | 49.7, **85.7**, 47.9 | 61.1% | $0.96 |
| The same, with self-votes excluded | 47.9, 48.9, 47.9 | 48.2% | $0.64 |

Fourteen of the fifteen runs got no further than the 12 shifted biases, which stand out when the weights are read. One council run,
chaired by Gemini, also repaired all 14 sign flips and 15 of the 16 rewired connections. Gemini's analysis found the flips and Claude's
winning proposal found the wiring. No run repaired the 20 mirrored units. An audit of that run's code found no network simulation and no
fitting by gradient descent: it worked from the activations a forward pass had recorded.

That is the first case of a council doing what no member did alone, but it is one run in six. The same chair with self-votes excluded
did not repeat it, and the mean is carried by that single run.

We then repeated the 85.7% setup five times with new seeds, with the test fixed in advance: 70% counts as a breakthrough, and the claim
needs 2 of 5. **It failed: 0 of 5.** Four runs made no edit, and one reached 47.1%. In all four, the Gemini chair spent all 15 Python runs on
analysis, in at least one case reaching a promising permutation, then had no run left to save anything. The original
85.7% happened to save on its last run. So the 15-run limit, added to control cost, shaped this result, and the test says little about
what a council can do without it. wsplab now warns an editor before its last runs if nothing is saved. There is still no proof that
councils beat a single editor.
Excluding self-votes showed no benefit. When self-votes were allowed, every member with a proposal ranked its own first on this task,
so Borda decided on second-place votes.

### Sharing findings at every step

`Team` is a cooperative council: after each result, teammates share short notes and one lead writes the next step. With Claude leading,
a 40-run Python limit and the same 70% bar, it failed too, with scores of 49.7, 54.5, 47.9, 47.9 and 47.9. That test is incomplete: Gemini's
account stopped responding partway, so the last three runs lost almost all of one teammate's notes, and the two runs that kept all three
members scored best. In a first version teammates simply echoed one another, so they are now asked for what is new or contested.

`search(..., shared=True)` runs peers instead, with no lead: each model works on its own copy and decides for itself, and all of them
see each other's work after every step. The first live peer pilot selected a candidate scoring 47.9% (against the damaged
35.2% baseline); its individual candidates scored 48.9%, 39.6% and 47.9% on evaluation in a subsequent diagnostic audit.
The winner was selected on a separate selection set, so it need not have the highest evaluation score.
The pilot showed sharing but did not reach the 70% breakthrough threshold. It cost about $6.11 according to the corrected
local accounting and was cut short by its spending limit; it is not a completed fair comparison with solo baselines.

That pilot's peer messages were truncated to 2,500 reply characters and 1,500 result characters. New runs preserve the full
reply and returned result text, with code fences escaped as quotations. This can increase token usage; existing spending
limits still apply. The pilot therefore does not measure the revised full-sharing implementation.

### A positive peer comparison on modular repair

The [routed digit repair task](experiments/routed-digits.md) makes independent partial repairs measurable.
On a fresh model-test seed, terra and Claude peers selected a candidate scoring **88.82%**, versus
**83.53%** for Claude alone and **78.82%** for terra alone, on the same 340 external held-out queries.
Both arms selected candidates using separate selection data. All four sessions finished normally
with six forward passes each and identical per-editor limits and provider spending ceilings.

The peer gain over the selected independent candidate was 5.29 percentage points. Peers cost an
estimated $2.20, versus $1.73 for both independent attempts and $1.50 for Claude alone. This is one
seed on a deliberately modular task, not a demonstrated general or efficiency advantage. An earlier
pilot was disqualified after a contributing peer simulated modified-network predictions in Python.
The new run's audit found no such replay or external evaluation access; a peer's internal practice
holdout bookkeeping issue is disclosed in the [full results](experiments/peer-confirmation-results.md).

### Testing a naturally imperfect model

The genuine-task experiment uses the unmodified public CIFAR-100 ResNet-20 checkpoint.
Its ordinary full-test accuracy reproduces the author's 68.83% exactly; no faults
are planted. OpenAI, Claude and Gemini are included in both peer and independent
arms, with 8,000 images reserved for external evaluation. The first attempt was
stopped after finding that the generic activation recorder returned numpy views
that later in-place operations overwrote. The recorder now snapshots outputs;
regression tests and an independent reference verify the fix. That attempt has
no valid performance result. The [corrected plan](experiments/natural-cifar100-restart-plan.md)
and [baseline verification](experiments/natural-cifar100-baseline.json) are retained;
the user approved revised provider limits for the corrected paid comparison.
That comparison is now complete: the selected peer candidate scored 68.525%
against the selected independent candidate's 68.800% and the original's 68.875%.
Gemini was included in both arms. Its peer simulated modified-network tails
outside the forward tool, invalidating the peer arm as a clean comparison;
both Gemini sessions also reached their budget guards with passes unused.
There is no council win. See the [full results and audit](experiments/natural-cifar100-results.md).

## Voting on scripts

```python
from wsplab.council import Council
from wsplab.editors import Spending, get_editor

council = Council([get_editor(e) for e in ("anthropic:claude-opus-5-5", "openai:some-model")],
                  Spending(max_usd=50, prices={...}), method="borda")
wsplab.run(net, editor=council, ...)
```

The first member chairs and does the analysis alone. When its reply would change weights, every member proposes a reply from the
same transcript, and all of them rank the proposals. Only the winner runs.

- **Ballots are anonymous.** Each voter sees the proposals without names, in its own random order.
- **Methods:** `borda` (default), `plurality`, `approval`, with optional `weights` per member.
- **Ties** go to the chair's ranking.
- **An edit cannot skip the vote.** A static check routes edits to a round, and the kernel refuses edits from any block that did
  not win one.
- **A vote does not override measurement.** The winner still goes through the evaluation gate.

`session/council.jsonl` records every proposal, ballot and tally. The patch records members and rounds.

## Voting on finished weights

```python
wsplab.search(net, editors=[...], selection="vote", spending=Spending(max_tokens=5_000_000), build=..., description=...,
              practice=..., evaluate=...)
```

Each editor runs a whole session of its own. Practice data is split: editors work on one part and candidates are compared on the
other. Members then vote on the finished candidates, with ties going to the better measured score. Only the winner meets the
evaluation.

`selection="metric"` skips the vote and takes the best score. Either way, the result reports the success rate across runs, which
says more about a method than one good run does.

## Cost

Both need a `Spending` limit, and neither runs without one. A script round costs about two calls per member. A search runs a whole
session per candidate, and one of the paper's runs took 158 turns and 20 hours. The limit is checked before each call, so leave room
for one call's cost.

Members using the same underlying model share blind spots, so mix providers.
