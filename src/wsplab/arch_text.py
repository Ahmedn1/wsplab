"""A description of a language model's architecture for the prompt, generated from config.json. It says what the network
is built from, as the paper's prompts do, and nothing about the values of its weights. Pass your own text to override."""

from __future__ import annotations


def _n(x: int) -> str:
    return f"{x:,}"


def _list(items: list) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" and {items[-1]}"


def describe(name: str, config: dict) -> str:
    c = config.get("text_config") or config
    layers, width = c.get("num_hidden_layers"), c.get("hidden_size")
    heads = c.get("num_attention_heads")
    kv = c.get("num_key_value_heads", heads)
    size = c.get("head_dim") or (width // heads if width and heads else None)
    parts = [f"{_n(layers)} layers of width {_n(width)}"]
    types = c.get("layer_types")
    attention = f"{heads} query heads and {kv} key/value heads of size {size}"
    if types and len(set(types)) > 1:
        full = [i for i, t in enumerate(types) if t in ("full_attention", "attention")]
        others = sorted(set(types) - {"full_attention", "attention"})
        parts.append(f"layers {_list(full)} are full-attention layers with {attention}, the other {len(types) - len(full)} "
                     f"are {_list([t.replace('_', ' ') for t in others])} layers")
    else:
        parts.append(f"every layer has attention with {attention}")
    width_mlp = c.get("intermediate_size")
    if width_mlp:
        act = c.get("hidden_act", c.get("hidden_activation", ""))
        kind = {"silu": "SwiGLU", "gelu_pytorch_tanh": "GeGLU"}.get(act, "gated")
        parts.append(f"each layer has a {kind} MLP of width {_n(width_mlp)}")
    tied = c.get("tie_word_embeddings", config.get("tie_word_embeddings"))
    vocab = c.get("vocab_size")
    if vocab:
        parts.append(f"vocabulary {_n(vocab)}")
    parts.append("input embedding and output layer tied" if tied else "separate input embedding and output layer")
    return f"{name} ({'; '.join(parts)})"


def default_frozen(names: list[str]) -> tuple[str, ...]:
    """The paper's scope: the final norm, the embedding and the output layer."""
    frozen = []
    for n in names:
        if n.endswith(("embed_tokens.weight", "lm_head.weight", "wte.weight", "ln_f.weight", "ln_f.bias")):
            frozen.append(n)
        elif n.endswith("norm.weight") and ".layers." not in n and ".h." not in n and "visual" not in n and "mtp" not in n:
            frozen.append(n)
    return tuple(frozen)


def text_tensors(names: list[str]) -> list[str]:
    """The text part of a multimodal checkpoint, which is what the paper's sessions saw."""
    skip = ("visual.", "vision_tower", "mtp.", "multi_modal_projector")
    return [n for n in names if not any(s in n for s in skip)]
