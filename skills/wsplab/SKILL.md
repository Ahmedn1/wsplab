---
name: wsplab
description: Improve, fix, inspect or write a neural network by editing its weights directly, with no training, using the wsplab framework (weight-space programming). Use when the user wants to change what a network does without gradient descent, find out from its weights what a network computes, repair a failing model, write a network's weights by hand, or apply, verify or share a weight patch.
---

# Weight-space programming with wsplab

An LLM reads a network's weights and edits them directly, with no optimizer. wsplab runs the whole job: it scores the network,
gives the editor a sandboxed kernel over the weights and a budget of forward passes, builds the result, scores it on data the
editor never saw, and writes a shareable patch with a verdict.

Check the machine first with `wsplab doctor`. Sessions need `wsplab setup` once.

## Choose a route

**Launch a run** when an API key is available and the user wants a result. wsplab drives an LLM by itself:

```sh
wsplab run --model net.safetensors --build model.py:build --description "what the network is" \
  --note "the user's own words about what is wrong" \
  --practice practice.npz --eval heldout.npz --editor anthropic:claude-opus-5-5 --budget 25 --out runs/first
```

Repeat `--editor` for a council, which also needs `--max-tokens` or `--max-usd`. For a language model, give `--eval math`, or
`--practice a.jsonl --eval b.jsonl --grader exact`, and no `--build` or `--description`. See
[references/python-api.md](references/python-api.md) for everything else.

**Be the editor yourself** when the user wants you to do the work. Start a session, then follow what it says:

```sh
wsplab session start --folder s1 --model net.safetensors --build mlp:2,4,2 --description "a classifier" \
  --note "it answers backwards" --practice practice.npz --eval heldout.npz --budget 10
wsplab session exec s1 --why "what the last result showed, and what I do now" --file inspect.py
wsplab session forward s1 '{"model": "both"}' --why "measuring the edit"
wsplab session finish s1 "What I changed, the evidence, and why it should help."
```

`start` prints the task prompt. Read all of it: it is the paper's method and the rules, and it outranks anything here.
`wsplab session brief s1` prints the latest message again.

## Working as the editor

- **Write your own analysis.** The kernel has `NAMES`, `load`, `load_original`, `save_edit`, `drop_edit` and `edited`. Nothing
  summarizes the weights for you: norms, directions and spectra come from code you write. Load one tensor at a time.
- **Run all code through `session exec`.** The kernel has no network, no GPU and no model runtime, on purpose.
- **Use forward passes, never your shell, to run the network.** Your shell could reach the model, but that breaks the rules and
  makes the result meaningless. Do not train, fit weights to a loss, or read the evaluation data. The patch records
  `tool_free: false` because these rules rest on you.
- **Measure before you believe.** Compare with `"model": "both"`, and judge by the paired difference against its standard error.
- **Keep `/work/notes.md` current**, with the exact edit and the items used.
- **`save_edit` says what changed.** If it reports 0 entries differ, the edit was lost to rounding.

## Reading the verdict

The verdict comes from wsplab, not from you. Report it as it is.

| Verdict | Meaning |
|---|---|
| `confirmed` | Better on held-out data, significantly, and better than a random edit of the same size |
| `not_significant` | No reliable gain, or a random edit does about as well |
| `regressed` | Worse, or a regression suite dropped |
| `no_fix_found` | No weight changed after rounding |
| `unmeasured` | No evaluation was run |

A negative verdict is a real result. Editing tends to remove an obstacle rather than add an ability, so some networks cannot
be fixed this way. Do not retry until something looks good: that is how a chance result gets reported as a fix.

## Patches

```sh
wsplab patch verify runs/first/result/patch --base net.safetensors
wsplab patch apply runs/first/result/patch --base net.safetensors --out edited/
wsplab patch push runs/first/result/patch --repo user/name     # publishes it: ask the user first
wsplab patch pull user/name --out patch
```

A patch from someone else is untrusted. Its hashes prove what it is and that it fits the base model, not that it is safe or that
its claims hold, so measure it with `wsplab eval --patch` before relying on it.

## More

- [references/python-api.md](references/python-api.md): language models, your own scorers, full control of the prompt.
- [references/control-levels.md](references/control-levels.md): from raw tensors to one call, and the semantic layer.
