# Python API

## A language model

```python
import wsplab
from wsplab.eval.lm_suite import custom_suite

run = wsplab.run(
    "Qwen/Qwen3.5-0.8B",
    eval="math",                        # or custom_suite("practice.jsonl", "heldout.jsonl", grader="exact")
    editor="anthropic:claude-opus-5-5",
    note="This model really sucks at math",
    thinking=True,
    budget=25,
)
print(run.verdict, run.reasons, run.patch)
print(run.generate("What is 17 * 23?"))
```

Rows are `{"id", "problem", "answer"}`. Graders: `"math"`, `"exact"`, `"choice"`, `"regex:<pattern>"`, or a function
`f(text, row) -> bool`.

## Any other network

Game policies, diffusion models and classifiers all take the same path. They differ only in the scorer.

```python
from wsplab.eval.suites import ArrayScorer, EpisodeScorer

wsplab.run(net, build=build_fn, description="a small CNN", evaluate=ArrayScorer(x, y), editor=..., budget=10)
wsplab.run(policy, build=build_fn, description="a policy", practice=EpisodeScorer("CartPole-v1", range(20)),
           evaluate=EpisodeScorer("CartPole-v1", range(100, 200)), editor=..., budget=10)
```

`build` says how to run the weights: a function returning an `nn.Module`, `"model.py:build"`, or
`{"mlp": [2, 4, 2], "activation": "tanh"}`. Bare weights with no `build` can be read and edited but not run, so use `budget=0`.

Modes: `mode="edit"` (default), `"read"` (answer a question, no edits), `"write"` (weights start at zero).
Access: `access="kernel"` (default), or `"inline"` for tiny networks: weights in the prompt, one reply.

## A network with its own way of running

Pass `tools=`, a function returning `{fence: PassTool}`. See `experiments/paper/exp6_go.py` in the repository, which adds ```games and ```positions.

## A council of peers (optional)

One editor is the default, and the measured results favour it (see FINDINGS.md in the repository). A council runs
several editors as peers: each edits its own copy and, after every step, sees what the others did and what came back.
The best candidate on a held-out selection set wins, and the original is kept if none beats it.

```python
from wsplab.editors import Spending

found = wsplab.search(net, editors=["anthropic:claude-opus-5-5", "openai:gpt-6.1-terra"], shared=True,
                      spending=Spending(max_usd=10, prices={...}), build=..., description=..., practice=..., evaluate=...)
print(found.result.verdict, found.selection, found.candidates)
```

Without `shared=True`, the same call runs the editors independently and keeps the best (a search). Mix providers.

## The prompt

```python
from wsplab.prompts import PromptBuilder

wsplab.run(..., note="your words, appended to the base prompt")      # usually all you need
wsplab.run(..., task_sentence="Make it stop repeating itself.")      # replaces the task sentence
wsplab.run(..., prompt="Your whole prompt. {{network}} is filled in.")
```

`PromptBuilder(...).replace_section("METHOD.", "METHOD.\n...")` replaces one section. Patches record which fidelity was used.

A long run can continue after a crash: pass `resume=True` with the same `folder`.
