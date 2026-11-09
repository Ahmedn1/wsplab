"""The semantic layer: M.layers[15].mlp.neuron[11].out, M.layers[15].attn.head[7].out, or for any network
M.module["layers.0"].unit[3]. Everything lowers to whole-tensor save_edit. Addresses only, never statistics."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

from .families import Builder, detect, geometry, gpt2_style, llama_style
from .refs import Parts, Ref

__all__ = ["Parts", "Ref", "SemanticMap", "kernel_prelude", "prompt_addendum"]


class Modules:
    """Any network, from tensor names alone: "a.b.weight" is module "a.b". unit[i] is index i along the first axis (a
    Linear's output unit, a Conv's filter, an Embedding's row), column[j] along the second."""

    def __init__(self, b: Builder):
        self.b = b
        self.names = sorted({n.rsplit(".", 1)[0] for n in b.names if n.endswith(".weight")})

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, name: str):
        b, weight, bias = self.b, f"{name}.weight", f"{name}.bias"
        if weight not in b.names:
            raise KeyError(f"no module {name!r}; see list(M.module)")
        shape = b.shape(weight)
        rest = (slice(None),) * (len(shape) - 1)

        def unit(i: int):
            return SimpleNamespace(weights=b.ref(weight, (slice(i, i + 1), *rest), 0),
                                   bias=b.ref(bias, (slice(i, i + 1),), 0) if bias in b.names else None)

        def column(j: int):
            tail = (slice(None),) * (len(shape) - 2)
            return b.ref(weight, (slice(None), slice(j, j + 1), *tail), 0)

        return SimpleNamespace(weight=b.whole(weight), bias=b.whole(bias) if bias in b.names else None,
                               unit=Parts(shape[0], unit, "unit"),
                               column=Parts(shape[1] if len(shape) > 1 else 0, column, "column"))


class SemanticMap:
    def __init__(self, names: list[str], config: dict, *, load: Callable, save: Callable, shape: Callable):
        self.builder = Builder(names, load, save, shape)
        self.module = Modules(self.builder)
        found = detect(names)
        self.family = found[0] if found else "generic"
        self.layers = None
        if found:
            make = gpt2_style if found[0] == "gpt2" else llama_style
            self.layers = make(self.builder, geometry(config), found[1])

    @classmethod
    def from_store(cls, store, load: Callable | None = None) -> SemanticMap:
        return cls(store.names, store.checkpoint.config, load=load or store.load, save=store.save_edit,
                   shape=store.checkpoint.shape)

    def tensor(self, name: str, output_axis: int = 0) -> Ref:
        """Any tensor by name: the way down to raw access."""
        return self.builder.whole(name, output_axis)


def kernel_prelude() -> str:
    """Defines M inside the kernel, on the same load and save_edit the editor uses."""
    return "from wsplab.semantic import SemanticMap\nM = SemanticMap.from_store(_store, load)\n"


def prompt_addendum(family: str) -> str:
    """Added to a prompt only when the semantic layer is switched on, which is recorded as a change from the paper's."""
    common = (" Also defined: M, named addresses that lower to save_edit, with .read(), .write(x), .scale(c), .zero(), "
              ".add(x) and .project_out(direction, strength). Any tensor: M.tensor(name). Any module: "
              'M.module["name"].unit[i], .column[j], .weight, .bias; list(M.module) gives the names.')
    if family == "generic":
        return common
    return common + (" Layers: M.layers[l].mlp.neuron[i].out (what the neuron writes), .gate, .up; M.layers[l].attn.head[h].out, "
                     ".q, .k, .v; a layer without softmax attention has attn None. M addresses things and computes no "
                     "statistics.")
