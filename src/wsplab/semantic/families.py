"""Where the parts of known architectures live. New code. Unknown architectures fall back to the generic walker, and raw
tensor names always work."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from collections.abc import Callable

from .refs import Parts, Ref


def _text_config(config: dict) -> dict:
    return config.get("text_config") or config


@dataclass
class Geometry:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    layer_types: list | None


def geometry(config: dict) -> Geometry:
    c = _text_config(config)
    heads = c.get("num_attention_heads", c.get("n_head", 0))
    hidden = c.get("hidden_size", c.get("n_embd", 0))
    return Geometry(layers=c.get("num_hidden_layers", c.get("n_layer", 0)), hidden=hidden, heads=heads,
                    kv_heads=c.get("num_key_value_heads", heads),
                    head_dim=c.get("head_dim") or (hidden // heads if heads else 0), layer_types=c.get("layer_types"))


class Builder:
    def __init__(self, names: list[str], load: Callable, save: Callable, shape: Callable):
        self.names = set(names)
        self.load = load
        self.save = save
        self.shape = shape

    def ref(self, tensor: str, index: tuple, output_axis: int) -> Ref:
        return Ref(tensor, index, output_axis, self.load, self.save)

    def whole(self, tensor: str, output_axis: int = 0) -> Ref:
        return self.ref(tensor, (slice(None),) * len(self.shape(tensor)), output_axis)


def _rows(b: Builder, tensor: str, start: int, stop: int) -> Ref:
    """Rows of a Linear weight [out, in]: what these units read. Their output is along axis 0."""
    return b.ref(tensor, (slice(start, stop), slice(None)), 0)


def _cols(b: Builder, tensor: str, start: int, stop: int) -> Ref:
    """Columns of a Linear weight [out, in]: what these inputs write. Their output is along axis 0."""
    return b.ref(tensor, (slice(None), slice(start, stop)), 0)


def _mlp(b: Builder, prefix: str):
    down = f"{prefix}.mlp.down_proj.weight"
    gate, up = f"{prefix}.mlp.gate_proj.weight", f"{prefix}.mlp.up_proj.weight"
    if down not in b.names:
        return None
    width = b.shape(down)[1]

    def neuron(i: int):
        return SimpleNamespace(out=_cols(b, down, i, i + 1), gate=_rows(b, gate, i, i + 1) if gate in b.names else None,
                               up=_rows(b, up, i, i + 1) if up in b.names else None)

    return SimpleNamespace(neuron=Parts(width, neuron, "neuron"), down=b.whole(down), width=width)


def _attention(b: Builder, prefix: str, g: Geometry):
    o = f"{prefix}.self_attn.o_proj.weight"
    if o not in b.names:
        return None
    q, k, v = (f"{prefix}.self_attn.{x}_proj.weight" for x in "qkv")
    group = max(1, g.heads // max(1, g.kv_heads))
    q_width = b.shape(q)[0] // g.heads if q in b.names else g.head_dim

    def head(h: int):
        kv = h // group
        return SimpleNamespace(out=_cols(b, o, h * g.head_dim, (h + 1) * g.head_dim),
                               q=_rows(b, q, h * q_width, (h + 1) * q_width) if q in b.names else None,
                               k=_rows(b, k, kv * g.head_dim, (kv + 1) * g.head_dim) if k in b.names else None,
                               v=_rows(b, v, kv * g.head_dim, (kv + 1) * g.head_dim) if v in b.names else None)

    return SimpleNamespace(head=Parts(g.heads, head, "head"), out=b.whole(o))


def llama_style(b: Builder, g: Geometry, prefix: str):
    """Llama, Mistral, Qwen 2 and 3, Gemma 3, and Qwen3.5's hybrid: a layer without softmax attention has attn = None."""

    def layer(i: int):
        p = f"{prefix}.{i}"
        return SimpleNamespace(index=i, kind=g.layer_types[i] if g.layer_types else "attention", mlp=_mlp(b, p),
                               attn=_attention(b, p, g))

    return Parts(g.layers, layer, "layer")


def gpt2_style(b: Builder, g: Geometry, prefix: str):
    """GPT-2 stores Conv1D weights as [in, out], the transpose of a Linear, and fuses q, k and v."""

    def layer(i: int):
        p = f"{prefix}.{i}"
        fc, proj = f"{p}.mlp.c_fc.weight", f"{p}.mlp.c_proj.weight"
        attn_proj, fused = f"{p}.attn.c_proj.weight", f"{p}.attn.c_attn.weight"
        width = b.shape(proj)[0]

        def neuron(j: int):
            return SimpleNamespace(out=b.ref(proj, (slice(j, j + 1), slice(None)), 1),
                                   up=b.ref(fc, (slice(None), slice(j, j + 1)), 1), gate=None)

        def head(h: int):
            d = g.head_dim

            def part(k: int):
                start = k * g.hidden + h * d
                return b.ref(fused, (slice(None), slice(start, start + d)), 1)

            return SimpleNamespace(out=b.ref(attn_proj, (slice(h * d, (h + 1) * d), slice(None)), 1), q=part(0), k=part(1),
                                   v=part(2))

        return SimpleNamespace(index=i, kind="attention",
                               mlp=SimpleNamespace(neuron=Parts(width, neuron, "neuron"), down=b.whole(proj, 1), width=width),
                               attn=SimpleNamespace(head=Parts(g.heads, head, "head"), out=b.whole(attn_proj, 1)))

    return Parts(g.layers, layer, "layer")


def detect(names: list[str]) -> tuple[str, str] | None:
    """(family, prefix of the layers) from the tensor names."""
    for name in names:
        m = re.match(r"(.*?\.?h)\.\d+\.attn\.c_attn\.weight$", name)
        if m:
            return "gpt2", m.group(1)
    for name in names:
        m = re.match(r"(.*layers)\.\d+\.mlp\.down_proj\.weight$", name)
        if m:
            return "llama", m.group(1)
    return None
