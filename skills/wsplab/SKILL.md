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
wsplab session start --folder s1 --model net.safetensors --build mlp:2,4,2:relu --description "a two-class classifier" \
  --note "it answers backwards" --practice practice.npz --eval heldout.npz --budget 10
wsplab session exec s1 --why "what the last result showed, and what I do now" --file inspect.py
wsplab session forward s1 '{"model": "both"}' --why "measuring the edit"
wsplab session forward s1 --file s1/run/work/request.json --why "measuring on DEV"
wsplab session finish s1 "What I changed, the evidence, and why it should help."
```

- `--description` completes "You are given the weights of ...", so write a lowercase phrase. `.npz` files hold arrays named
  `inputs` and `targets`.
- `start` prints the task prompt. Read all of it: it is the paper's method and the rules. It speaks of "blocks": a ```python
  block is `exec`, a ```forward block is `forward`, your explanation is `--why`, and replying without a block is `finish`.
- `wsplab session brief s1` prints the latest message, and `--task` prints the task again.
- `/work` in the kernel is `s1/run/work` on the host.
- You may `finish` with passes unused. Do, once your edit is frozen and confirmed: passes spent after that can only tempt you
  to tune on what you have seen.

## Working as the editor

- **Write your own analysis.** The kernel has `NAMES`, `load`, `load_original` (both return float32 torch tensors), `save_edit`,
  `drop_edit` and `edited`. Nothing summarizes the weights for you: norms, directions and spectra come from code you write. Load
  one tensor at a time.
- **Run all code through `session exec`.** The kernel has no network and no GPU. It has PyTorch but not transformers or vLLM.
- **Use forward passes, never your own code, to run the network.** Nothing stops you simulating a small network in the kernel or
  your shell, but that breaks the rules and makes the result meaningless. Arithmetic on weights and on activations a pass
  recorded is analysis, and allowed. Feeding an input through the layers yourself is running the network, and is not.
- **Stay away from the evaluation.** Do not read the `--eval` file or `run/baseline.json`, which holds its per-item scores. Do not
  train or fit weights to a loss. The patch records `tool_free: false` because these rules rest on you.
- **HOLDOUT is not the evaluation.** The prompt's HOLDOUT is a part of the practice data you set aside to confirm your edit.
  The evaluation is separate, and you never see it.
- **Measure before you believe.** Compare with `"model": "both"`, and judge by the paired difference against its standard error.
- **Keep `/work/notes.md` current**, with the exact edit and the items used.
- **`save_edit` says what changed.** If it reports 0 entries differ, the edit was lost to rounding.

## Reading the verdict

The verdict comes from wsplab, not from you. Report it as it is.

| Verdict | Meaning |
|---|---|
| `confirmed` | Better on held-out data, significantly, and better than a random edit of the same size. It does not mean every fault is gone |
| `not_significant` | No reliable gain, or a random edit does about as well |
| `regressed` | Worse, or a regression suite dropped |
| `no_fix_found` | No weight changed after rounding |
| `unmeasured` | No evaluation was run |

The random control moves each change to a random place in the same tensor with a random sign, so it is usually worse than the
baseline, and its gap can exceed the gain. A negative verdict is a real result. Editing tends to remove an obstacle rather than add an ability, so some networks cannot
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
