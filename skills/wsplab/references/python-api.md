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

Pass `tools=`, a function returning `{fence: PassTool}`. See `examples/paper/exp6_go.py`, which adds ```games and ```positions.

## A council or a search

```python
from wsplab.council import Council
from wsplab.editors import Spending, get_editor

council = Council([get_editor(e) for e in editors], Spending(max_usd=50, prices={...}))
wsplab.run(net, editor=council, ...)                    # votes on each edit script

wsplab.search(net, editors=editors, spending=Spending(max_tokens=5_000_000), selection="vote", ...)  # votes on finished edits
```

Mix providers: members built on one model share its blind spots.

## The prompt

```python
from wsplab.prompts import PromptBuilder

wsplab.run(..., note="your words, appended to the base prompt")      # usually all you need
wsplab.run(..., task_sentence="Make it stop repeating itself.")      # replaces the task sentence
wsplab.run(..., prompt="Your whole prompt. {{network}} is filled in.")
```

`PromptBuilder(...).replace_section("METHOD.", "METHOD.\n...")` replaces one section. Patches record which fidelity was used.

A long run can continue after a crash: pass `resume=True` with the same `folder`.
