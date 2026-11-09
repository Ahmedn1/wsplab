# Councils of peers (optional)

One editor is the default, and the evidence so far favours it. A council is an option for when you want several
models to work on the same network.

## How it works

```python
from wsplab.editors import Spending
from wsplab.editors.llm import ChatEditor

gemini = ChatEditor("gemini-4-argon", api_key=GEMINI_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
found = wsplab.search(
    net, build=build_net, description="a small image classifier", practice=practice, evaluate=heldout,
    editors=["anthropic:claude-opus-5-5", "openai:gpt-6.1-terra", gemini],
    shared=True, spending=Spending(max_usd=10, prices={...}), budget=8,
)
print(found.result.verdict, found.selection, found.candidates)
```

From the command line, repeat `--editor` and set `--max-usd` or `--max-tokens`.

- **Every peer runs a whole session of its own.** Each has its own kernel, its own copy of the weights, its own
  passes and its own budget.
- **Peers move in rounds.** After each step, a peer waits for the others, then sees what each of them did (the full
  reply) and what came back (the full result). It then decides its own next step: it can adopt, test or contradict
  anything it saw. There is no lead and no vote, and sharing costs no extra calls.
- **One candidate wins, on measurement.** The practice data is split. Editors work on one part, and every finished
  candidate is scored on the other part, the selection set. The best selection score wins, and the original is kept
  if no candidate beats it. Only the winner meets the evaluation and the gate.
- **Without `shared=True`**, the same call runs the editors independently and keeps the best (a search). That is the
  fair baseline for a council: the same models, the same budgets, no sharing.
- **A spending limit is required.** A council multiplies calls, and the limit is checked before each one.

## What we measured

Full results and charts are in [FINDINGS.md](../FINDINGS.md). In short:

- **Sharing lifts the weakest member toward the strongest one.** On Seaquest, gpt-6.1-terra went from 70.5 alone to
  74.8 as a peer. On Spurious ImageNet it went from 0.688 to 0.757.
- **No council beat its strongest member on a natural task.** On Seaquest the council tied Claude Opus alone
  (−0.9 ± 0.8). On Spurious ImageNet its lead (+0.009) sat inside class-level noise. On CIFAR-100, FinQA and Gemma
  math there was no win.
- **The one clear win was synthetic.** On the routed-digit task, built so separate repairs add up, the peers' pick
  beat the best solo editor by 5.3 points, on one seed.
- **A council of cheap models can't stand in for a strong one.** Luna, Haiku and Gemini Flash together found nothing,
  23.6 points below Claude Opus alone.
- **Giving members fixed roles made it worse.** A proposer, critic and analyst council lost 6.0 ± 1.0 points against
  Claude alone. The teammates' long reports filled the proposer's context.

Earlier variants (a chair council that voted on edit scripts, a team with a lead, voting on finished candidates, and
fixed roles) did no better and were removed in 0.1.0. They are kept in `experiments/code/legacy_council.py` so those
experiments can be reproduced.

## Practical advice

- **Mix providers.** Members built on one model share its blind spots.
- **Budget for context.** Peers read each other's work, so their prompts grow faster than a solo editor's. Several
  sessions hit their spending caps or the peer-context bound early.
- **Watch for simulation.** Some models compute the network's outputs in their kernel to score candidates, which the
  rules forbid. Read the transcripts (`session/tool_calls.jsonl`) before trusting a winning edit.
