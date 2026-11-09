# Concepts

## Three choices describe any run

| Choice | Values | What it changes |
|---|---|---|
| Kind | `lm`, `generic` | How the network runs. `lm` needs a tokenizer, sampling and generation. `generic` is inputs in, outputs out. |
| Mode | `read`, `edit`, `write` | Whether edits are allowed, and whether weights start at zero. |
| Access | `kernel`, `inline` | Scripts over lazily loaded tensors, or weights in the prompt and one `edits` block for tiny networks. |

There are no task types. A game policy and a diffusion model are both `generic`. What differs is data: a description, your note,
a scorer, and sometimes a custom pass tool.

## One base prompt

Every run starts from the same text: the authors' last prompt revision. Only three things vary.

- **Facts:** the network, the budget and true limits.
- **The tools section:** language models and generic networks are run differently.
- **Your note:** free text appended at the end.

The system message is the paper's single line, "Answer the user's request."

## What happens in a run

1. Load the network and score it. This is the baseline.
2. Editing session: the editor inspects weights in the kernel, spends forward passes and saves edits.
3. Build: cast to the model's dtype, write the patch and the edited model.
4. Score the edited model on the evaluation, with a size-matched random edit as a control.
5. Reach a verdict and write `report.md`.

A forward pass and the final evaluation run the same code on the same built model, so what gets measured is exactly what the
patch contains. Pass `resume=True` with the same folder to continue after a crash. The baseline and conversation are kept, and
the kernel's variables are lost, which the editor is told.

## The editor

An editor is anything that writes replies. The protocol is plain text: one fenced block per reply, or none for a final answer.
No tool calling is needed, so any chat model works.

`anthropic:<model>`, `openai:<model>` (OpenAI's Responses API), `openai-chat:<model>` (any OpenAI-compatible chat
server, such as vLLM, SGLang, llama.cpp or Ollama, at `OPENAI_BASE_URL`), `codex:<model>` (the route the paper used), a
coding agent through `wsplab session`, or your own object with a `reply` method.

## The kernel

Persistent Python inside bubblewrap: no network, no GPU, a read-only model, a clean environment with no transformers or vLLM, and
a memory cap. `load(name)` reads one tensor lazily, so a model larger than RAM can be inspected.

The kernel's rules are real but not a proof. Gradients are off and the model runtime is absent, yet nothing stops an editor from
writing the network's arithmetic in numpy, so read transcripts when the rules matter.

## Control levels

See [`skills/wsplab/references/control-levels.md`](../skills/wsplab/references/control-levels.md).
