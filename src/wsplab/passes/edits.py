"""The `edits` language of the paper's one-reply experiments, as specified in the prompts of experiments 4 and 5:
W1[12, 0] = 2.5, W1[0:6, 3] = [1, 0, -1], W3[:, 2] *= 2. New code: the authors' reader was not released. Values are read
with ast.literal_eval, so nothing an editor writes is executed."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

import numpy as np

_BLOCK = re.compile(r"^```edits[ \t]*\n(.*?)^```[ \t]*$", re.M | re.S)
_STATEMENT = re.compile(r"^(?P<name>[\w./-]+)\s*\[(?P<index>[^\]]*)\]\s*(?P<op>=|\*=|\+=|-=)\s*(?P<value>.+)$", re.S)
MAX_VALUE = 1e6


@dataclass
class Applied:
    statements: int = 0
    errors: list[str] = field(default_factory=list)
    tensors: dict[str, np.ndarray] = field(default_factory=dict)

    def report(self) -> dict:
        return {"statements_read": self.statements, "errors": self.errors, "tensors": sorted(self.tensors)}


def last_block(reply: str) -> str | None:
    blocks = _BLOCK.findall(reply)
    return blocks[-1] if blocks else None


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def statements(block: str) -> list[str]:
    """Join continuation lines: a statement ends when its brackets balance."""
    out, current, depth = [], "", 0
    for raw in block.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        current = f"{current} {line.strip()}".strip()
        depth = current.count("[") - current.count("]")
        if depth <= 0:
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


def _index(text: str) -> tuple:
    parts = []
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            bits = [int(x) if x.strip() else None for x in part.split(":")]
            if len(bits) > 3:
                raise ValueError(f"bad range {part!r}")
            parts.append(slice(*bits))
        else:
            parts.append(int(part))
    return tuple(parts)


def _value(text: str):
    value = np.asarray(ast.literal_eval(text.strip()), dtype=np.float64)
    if not np.isfinite(value).all() or np.abs(value).max(initial=0) > MAX_VALUE:
        raise ValueError("values must be finite, with |v| <= 1e6")
    return value


def apply_edits(block: str, load) -> Applied:
    """Apply statements in order. `load(name)` gives a tensor as a float array. A statement that cannot be read is
    skipped, and the others still apply, as the prompts promise."""
    result = Applied()
    for text in statements(block):
        try:
            m = _STATEMENT.match(text)
            if m is None:
                raise ValueError("not a statement")
            name, op = m["name"], m["op"]
            if name not in result.tensors:
                result.tensors[name] = np.array(load(name), dtype=np.float64)
            tensor = result.tensors[name]
            index = _index(m["index"])
            value = _value(m["value"])
            if op == "=":
                tensor[index] = value
            elif op == "*=":
                tensor[index] *= value
            elif op == "+=":
                tensor[index] += value
            else:
                tensor[index] -= value
            result.statements += 1
        except (ValueError, IndexError, KeyError, SyntaxError, TypeError) as e:
            result.errors.append(f"{text[:80]}: {e}")
    return result


def render_weights(checkpoint, names: list[str] | None = None, decimals: int = 3) -> str:
    """Every weight as text, rounded as in the prompts of experiments 1 and 2: one matrix row per line."""
    parts = []
    for name in names or checkpoint.names():
        array = checkpoint.float(name)
        rows = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(1, -1)
        body = "\n".join(" ".join(f"{x:.{decimals}f}" for x in row) for row in rows)
        parts.append(f"{name}, shape {' x '.join(map(str, array.shape))}:\n{body}")
    return "\n\n".join(parts)
