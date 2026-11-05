# Councils and search

One editor is the default, and we recommend starting there. A council is optional, and so far there is no evidence that it beats a
good single editor.

## What has been measured

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
