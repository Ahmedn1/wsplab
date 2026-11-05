# Councils and search

One editor is the default. A council is optional, and whether voting beats one strong editor is an open question: measure it on
your own task.

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
session per candidate, and one of the paper's runs took 158 turns and 20 hours. Cached input counts at the full price, so the limit
errs on the safe side.

Members using the same underlying model share blind spots, so mix providers.
