# wsplab

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: alpha](https://img.shields.io/badge/status-alpha-orange)](CHANGELOG.md)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![Findings](https://img.shields.io/badge/findings-11%20experiments-8a2be2)](FINDINGS.md)

Weight-space programming, end to end. An LLM reads a neural network's weights and edits them directly: no gradient
descent, no training pipeline. wsplab runs the whole job, from the baseline to a verdict, and packages the edit as a
small patch anyone can apply to the public base model.

```python
import wsplab

run = wsplab.run(
    "Qwen/Qwen3.5-0.8B",
    note="This model really sucks at math",
    eval="math",
    editor="anthropic:claude-opus-5-5",
    thinking=True,
    budget=25,
)
print(run.verdict, run.patch)
print(run.generate("What is 17 * 23?"))
```

Based on [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) by Takehi,
Teoh, Zhang and Khattab. wsplab is independent and is not affiliated with or endorsed by the authors.

## What it does

- **Runs everything.** It scores the network, then gives the editor a sandboxed kernel and a budget of forward
  passes. It builds the edit, scores it again on data the editor never saw, and reaches a verdict.
- **Works on any network.** Language models and any PyTorch network: a game policy, a CNN and a diffusion model take
  the same path and differ only in how they are scored.
- **Uses the paper's method.** The same base prompt, turn protocol and editing surface. The editor writes its own
  scripts to inspect the weights, so nothing has to fit in context and nothing is pre-summarized. Replaying the
  authors' sessions reproduces their prompts byte for byte.
- **Tells you when it didn't work.** A hidden evaluation, paired statistics and a same-size random-edit control
  decide the verdict, not the editor. `no_fix_found` is a normal outcome.
- **Makes edits shareable.** A patch is a few small files that rebuild the edited model bit for bit from the public
  base. Patches can be verified, stacked and shared on the Hugging Face Hub.
- **Lets you go as low as you want.** From one call down to raw tensors, including replacing the whole prompt.
- **One editor by default; a council of peers is optional.** Several LLMs can edit side by side and see each other's
  work. The [findings](FINDINGS.md) say when that helps (rarely).

## Install

```sh
pip install wsplab                        # patch tools only: needs just numpy
pip install 'wsplab[torch,kernel]'        # run any PyTorch network
pip install 'wsplab[lm,vllm,eval,hub]'    # language models
wsplab setup                              # builds the sandbox environment, once
wsplab doctor                             # says what this machine can run
```

- **GPU versions:** `constraints-gpu.txt` lists the versions tested together for language models; add
  `-c constraints-gpu.txt`. No CUDA toolkit is needed.
- **Sandbox:** editing sessions run in [bubblewrap](https://github.com/containers/bubblewrap) on Linux.
- **Python:** 3.12 or later.

## A network of your own

```python
from wsplab.eval.suites import ArrayScorer

run = wsplab.run(net, build=build_net, description="a small image classifier",
                 note="It confuses cats and dogs", evaluate=ArrayScorer(images, labels),
                 editor="anthropic:claude-opus-5-5", budget=10, max_changed=5_000)
print(run.verdict, run.infer(new_images))
```

`max_changed` is an optional edit budget: `save_edit` refuses edits that would change more entries than that. The
command line covers the same ground, plus `wsplab infer` and `wsplab eval`:

```sh
wsplab run --model net.safetensors --build model.py:build --description "a small image classifier" \
  --note "It confuses cats and dogs" --practice practice.npz --eval heldout.npz \
  --editor anthropic:claude-opus-5-5 --budget 10
```

## A council of peers (optional)

```python
from wsplab.editors import Spending

found = wsplab.search(net, build=build_net, description="a small image classifier", practice=practice,
                      evaluate=heldout, editors=["anthropic:claude-opus-5-5", "openai:gpt-6.1-terra"],
                      shared=True, spending=Spending(max_usd=10, prices={...}))
```

Each peer edits its own copy and sees the others' work after every step. The best candidate on a held-out selection
set wins, and the original is kept if none beats it. On the command line, repeat `--editor`. See
[docs/council.md](docs/council.md).

## Patches

```sh
wsplab patch apply patch/ --base Qwen3.5-0.8B/ --out edited/
wsplab patch verify patch/ --base Qwen3.5-0.8B/
wsplab patch pull user/some-patch --out patch/
```

A patch's hashes prove what it is and that it fits the base model. They don't prove it is safe or that its claims
hold.

## With a coding agent

`skills/wsplab` is an agent skill. It teaches an agent to launch runs, or to act as the editor itself through
`wsplab session`. Copy it into your agent's skills folder.

## Findings

We ran 11 experiments with real editors on one laptop. **[Read the findings](FINDINGS.md)**, with charts:

- **wsplab reproduces the paper's measurements:** the released patches rebuild bit for bit and the prompts replay
  byte for byte.
- **Editing works on steerable behaviour:** on the paper's Seaquest policy, strong editors raised the score from 52
  to 76 with 32 weights changed. It doesn't work on missing capability (Gemma math, multilingual math).
- **A two-line scripted recipe** matched or beat LLM editors on Spurious ImageNet.
- **Councils lift their weakest member** but didn't beat the strongest on any natural task. Cheap councils and fixed
  roles did worse.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/seaquest-dark.png">
  <img alt="Seaquest mean return for each editor and council" src="docs/figures/seaquest-light.png">
</picture>

Everything needed to reproduce them, including code, plans, result files and commands, is in
[`experiments/`](experiments/).

## How wsplab compares

Several weight-space programming projects appeared in November 2026. As of 8 November, from their public READMEs:

| | wsplab | [dspy-wsp](https://pypi.org/project/dspy-wsp/) | [weightspace](https://github.com/leofmarciano/weightspace) | [wsp-kit](https://github.com/AuremHQ/weight-space-programming) | [WSP-toolkit](https://github.com/ZeroTang05/WSP-toolkit) |
|---|---|---|---|---|---|
| LLM writes the edits | yes | yes (a DSPy LM) | yes | no (a numerical estimator) | via tool calls |
| Language models | yes (vLLM, HF) | yes (vLLM, HF, MLX) | yes | small LLMs (≤ 3B) | Qwen3, Llama |
| Any PyTorch network | yes | decision models | not stated | no | no |
| Sandboxed kernel | bubblewrap | bubblewrap | Python sandbox | not stated | not stated |
| Budgeted forward-pass tool | yes | yes | not stated | no | no |
| Hidden evaluation + paired stats + random control | yes | holdout + valset, no control | not stated | metric thresholds | separate test set |
| Patches rebuilt bit for bit from the base | yes, plus Hub, stacking | yes | checkpoints | edited folder | replayable patches |
| Paper fidelity checks (byte-for-byte replay) | yes | no | no | no | no |
| Multiple editors | peers (optional) | population search (GEPA) | no | no | no |
| CLI and agent skill | yes, plus agent-as-editor sessions | CLI | no | CLI, skill, MCP | MCP |

dspy-wsp is the closest in scope, and it adds things wsplab lacks: DSPy integration and a population search over
candidate edits. Other related projects are agent skills
([di37](https://github.com/di37/weight-space-programming-skill),
[GriMReapeR-666](https://github.com/GriMReapeR-666/weight-space-programming-skill)), a research harness for coding
agents ([mfaseehuddin/wsp](https://github.com/mfaseehuddin/wsp)) and a browser version
([webWeightSpace](https://github.com/vishalmysore/webWeightSpace)). None of their code is used here.

## Read more

- [Concepts](docs/concepts.md): kinds, modes, access and control levels.
- [Patches](docs/patches.md): the format and how to share one.
- [Councils of peers](docs/council.md).
- [Fidelity to the paper](docs/fidelity.md): what is the authors' work, what is reconstructed, and what differs.
- [Findings](FINDINGS.md) and [experiments](experiments/).
- [Changelog](CHANGELOG.md).

## Citing

See [CITATION.cff](CITATION.cff). Please also cite the paper.

## License

MIT. Prompt text derives from the authors' release under CC BY 4.0: see [NOTICE](NOTICE).
