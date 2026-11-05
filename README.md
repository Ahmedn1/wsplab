# wsplab

Weight-space programming, end to end. An LLM reads a neural network's weights and edits them directly: no gradient descent, no
training data pipeline. wsplab runs the whole job, from baseline to verdict.

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

Based on [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) by Takehi, Teoh, Zhang
and Khattab. wsplab is independent and is not affiliated with or endorsed by the authors.

## What it does

- **Runs everything.** Scores the network, gives the editor a sandboxed kernel and a budget of forward passes, builds the edit,
  scores it again on data the editor never saw, and reaches a verdict.
- **Works on any network.** Language models, plus any PyTorch network. A game policy, a CNN and a diffusion model take the same
  path: they differ only in how they are scored.
- **Uses the paper's method.** Same base prompt, same turn protocol, same editing surface. The editor writes its own scripts to
  inspect the weights, so nothing has to fit in context and nothing is pre-summarized for it.
- **Tells you when it did not work.** A random-edit control, paired statistics and a hidden evaluation decide the verdict, not
  the editor. `no_fix_found` is a normal outcome.
- **Makes edits shareable.** A patch is a few small files that rebuild the edited model bit for bit from the public base.
- **Lets you go as low as you want.** From one call down to raw tensors, including replacing the whole prompt.
- **One editor by default, or a council.** Several LLMs can vote on edit scripts or on finished candidates.

## Install

```sh
pip install wsplab                        # patch tools only: needs just numpy
pip install 'wsplab[torch,kernel]'        # run any PyTorch network
pip install 'wsplab[lm,vllm,eval,hub]'    # language models
wsplab setup                              # builds the sandbox environment, once
wsplab doctor                             # says what this machine can run
```

`constraints-gpu.txt` lists the versions tested together for language models: add `-c constraints-gpu.txt`. No CUDA toolkit is
needed. Editing sessions run in [bubblewrap](https://github.com/containers/bubblewrap) on Linux. Python 3.12 or later.

## A network of your own

```python
from wsplab.eval.suites import ArrayScorer

run = wsplab.run(net, build=build_net, description="a small image classifier",
                 note="It confuses cats and dogs", evaluate=ArrayScorer(images, labels),
                 editor="anthropic:claude-opus-5-5", budget=10)
print(run.verdict, run.infer(new_images))
```

Or from the command line, which also covers `wsplab infer` and `wsplab eval`:

```sh
wsplab run --model net.safetensors --build model.py:build --description "a small image classifier" \
  --note "It confuses cats and dogs" --practice practice.npz --eval heldout.npz \
  --editor anthropic:claude-opus-5-5 --budget 10
```

## Patches

```sh
wsplab patch apply patch/ --base Qwen3.5-0.8B/ --out edited/
wsplab patch verify patch/ --base Qwen3.5-0.8B/
wsplab patch pull user/some-patch --out patch/
```

A patch's hashes prove what it is and that it fits the base model. They do not prove it is safe or that its claims hold.

## With a coding agent

`skills/wsplab` is an agent skill. It teaches an agent to launch runs, or to be the editor itself through `wsplab session`.
Copy it into your agent's skills folder.

## Read more

- [Concepts](docs/concepts.md): kinds, modes, access and control levels.
- [Patches](docs/patches.md): the format and how to share one.
- [Councils and search](docs/council.md).
- [Fidelity to the paper](docs/fidelity.md): what is the authors' work, what is reconstructed, and what differs.
- [`examples/paper`](examples/paper): the paper's eight experiments as configuration.

## License

MIT. Prompt text derives from the authors' release under CC BY 4.0: see [NOTICE](NOTICE).
