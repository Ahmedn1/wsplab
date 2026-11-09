# Control levels

Every level lowers to the one below, so all of them produce the same audit log and the same patch.

| Level | Use | For |
|---|---|---|
| 0 Bytes | `wsplab.io`, `wsplab.patch` | Tensors and patches, numpy only, no LLM |
| 1 Store | `load`, `save_edit`, `drop_edit`, `edited` | The paper's editing surface |
| 2 Semantic | `M.layers[15].mlp.neuron[11].out` | Named parts of a network |
| 3 Components | `Editor`, `SandboxKernel`, `PassTool`, `PromptBuilder`, scorers | Swapping one part |
| 4 Session | `wsplab.session.Session` | Budgets, handoffs, notices |
| 5 Run | `wsplab.run`, `wsplab.run_lm` | Just a note and an evaluation |

## The semantic layer

Pass `semantic=True` (or `--semantic`) to define `M` in the kernel. The prompt says so, and the patch records `paper+semantic`.

```python
M.layers[15].attn.head[7].out.zero()                  # the paper's 25-pass edit, exactly
M.layers[19].mlp.neuron[11].out.project_out(u, 1.0)   # W' = W - s u (u^T W) on what this neuron writes
M.layers[3].mlp.neuron[2].out.scale(1.5)
M.module["conv1"].unit[4].weights.read()              # any network: a filter, a unit, an embedding row
M.tensor("any.tensor.name").add(delta)
```

Verbs: `.read()`, `.write(x)`, `.scale(c)`, `.zero()`, `.add(x)`, `.project_out(direction, strength)`.

`M` is an address book: it computes no statistics. Llama-style models, Qwen3.5's hybrid and Gemma 3 are mapped, as is GPT-2's
transposed layout. In a hybrid, a layer with no softmax attention has `attn` set to `None`. Anything else falls back to
`M.module`, and `M.tensor` always works.
