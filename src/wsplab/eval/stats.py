"""Paired statistics and score merging. merge_scores follows exp8_math/code/merge.py (MIT)."""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import asdict, dataclass


@dataclass
class Paired:
    """edited - original, over items. Each item's score is the mean over its samples."""

    items: int
    original: float
    edited: float
    difference: float
    standard_error: float
    wrong_to_right: int
    right_to_wrong: int

    def as_dict(self) -> dict:
        return asdict(self)

    def z(self) -> float:
        if self.standard_error == 0:
            return math.inf if self.difference > 0 else -math.inf if self.difference < 0 else 0.0
        return self.difference / self.standard_error


def paired(original: list[list[float]], edited: list[list[float]]) -> Paired:
    """Compare two models item by item. `original[i]` and `edited[i]` hold item i's sample scores."""
    if len(original) != len(edited) or not original:
        raise ValueError("paired comparison needs the same items for both models")
    a = [sum(x) / len(x) for x in original]
    b = [sum(x) / len(x) for x in edited]
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1) / n) if n > 1 else 0.0
    up = down = 0
    for xs, ys in zip(original, edited):
        for x, y in zip(xs, ys):
            up += x < y
            down += x > y
    return Paired(n, sum(a) / n, sum(b) / n, mean, se, up, down)


def merge_scores(folder: str | os.PathLike, sizes: dict[str, int] | None = None) -> dict:
    """Combine <folder>/scores.json and <folder>/parts/*/scores.json into acc / cut_share / mean_tokens per set."""
    sizes = sizes or {}
    acc: dict[str, dict] = {}
    full = os.path.join(folder, "scores.json")

    def is_set(key: str, value) -> bool:
        return not key.startswith("_") and isinstance(value, dict)

    if os.path.exists(full):
        for k, v in json.load(open(full)).items():
            if is_set(k, v) and not v.get("shard"):
                acc[k] = {"ok": v.get("ok", v["acc"] * v["n_problems"] * v["samples"]),
                          "total": v.get("total", v["n_problems"] * v["samples"]),
                          "cut": v.get("cut", v["cut_share"] * v["n_problems"] * v["samples"]),
                          "tokens_sum": v.get("tokens_sum", v["mean_tokens"] * v["n_problems"] * v["samples"]),
                          "problems": v["n_problems"], "from": "full"}
    for f in glob.glob(os.path.join(folder, "parts", "*", "scores.json")):
        for k, v in json.load(open(f)).items():
            if not is_set(k, v) or (k in acc and acc[k]["from"] == "full"):
                continue
            a = acc.setdefault(k, {"ok": 0, "total": 0, "cut": 0, "tokens_sum": 0, "problems": 0, "from": "parts",
                                   "shards": []})
            if v.get("shard") in a["shards"]:
                continue
            a["shards"].append(v.get("shard"))
            for x in ("ok", "total", "cut", "tokens_sum"):
                a[x] += v[x]
            if "ok_lenient" in v and a.get("ok_lenient", 0) is not None:
                a["ok_lenient"] = a.get("ok_lenient", 0) + v["ok_lenient"]
            else:
                a["ok_lenient"] = None
            a["problems"] += v["n_problems"]
    return {k: {"acc": a["ok"] / a["total"], "cut_share": a["cut"] / a["total"],
                "mean_tokens": a["tokens_sum"] / a["total"], "problems": a["problems"],
                "complete": a["problems"] == sizes.get(k, a["problems"]),
                "acc_lenient": a["ok_lenient"] / a["total"] if a.get("ok_lenient") is not None else None}
            for k, a in acc.items()}
