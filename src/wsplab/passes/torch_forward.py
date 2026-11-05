"""The forward-pass tool for any network: run it on practice items and report scores. Same contract as the language
model tool: "original", "edited" or "both", paired statistics, and failed requests cost nothing."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..eval.stats import paired
from .base import PassFailed, PassInvalid, PassResult


class TorchForward:
    fence = "forward"

    def __init__(self, runner, scorer, *, work: str | Path, base: Path, edited: Callable[[], Path],
                 edit_digest: Callable[[], str | None], max_items: int = 100_000):
        self.runner = runner
        self.scorer = scorer
        self.work = Path(work)
        self.base = Path(base)
        self.edited = edited
        self.edit_digest = edit_digest
        self.max_items = max_items
        self._seen: set[str] = set()

    def _request(self, body: str) -> dict:
        try:
            request = json.loads(body)
        except json.JSONDecodeError as e:
            raise PassInvalid(f"JSONDecodeError: {e}") from e
        if not isinstance(request, dict) or request.get("model") not in ("original", "edited", "both"):
            raise PassInvalid('"model" must be "original", "edited" or "both"')
        return request

    def _ids(self, request: dict) -> list[str]:
        known = self.scorer.ids()
        ids = request.get("items") or known
        missing = sorted(set(ids) - set(known))
        if missing:
            raise PassFailed(f"ValueError: unknown item ids: {missing[:10]}")
        if len(ids) > self.max_items:
            raise PassInvalid(f"{len(ids)} items is more than the limit of {self.max_items}")
        return ids

    def _check_repeat(self, request: dict) -> None:
        key = hashlib.sha1(f"{json.dumps(request, sort_keys=True)}|{self.edit_digest()}".encode()).hexdigest()
        if key in self._seen:
            raise PassInvalid("this request is identical to an earlier one with unchanged edits")
        self._seen.add(key)

    def _weights(self, model: str) -> Path:
        return self.base if model == "original" else self.edited()

    def _run(self, model: str, ids: list[str], request: dict, folder: Path):
        weights = self._weights(model)
        folder.mkdir(parents=True, exist_ok=True)
        try:
            call = self.runner.call(weights)
            scores = self.scorer.score(call, ids)
            if hasattr(self.scorer, "outputs"):
                np.save(folder / "outputs.npy", self.scorer.outputs(call, ids))
            names = request.get("activations") or []
            if names:
                if not hasattr(self.scorer, "inputs"):
                    raise ValueError("this scorer has no fixed inputs to capture activations on")
                batch = self.scorer.inputs[self.scorer._index(ids)]
                for name, array in self.runner.capture(weights, batch, names).items():
                    np.save(folder / f"act_{name}.npy", array)
        except Exception as e:  # noqa: BLE001 - a failed run must not cost the editor a pass
            raise PassFailed(f"{type(e).__name__}: {e}") from e
        (folder / "scores.json").write_text(json.dumps(scores.per_item))
        return scores

    def run(self, body: str, request_number: int) -> PassResult:
        request = self._request(body)
        ids = self._ids(request)
        self._check_repeat(request)
        started = time.time()
        k = request_number
        if request["model"] != "both":
            folder = self.work / f"fwd_{k}"
            scores = self._run(request["model"], ids, request, folder)
            summary = {"ok": True, "model": request["model"], "items": len(ids), "score": round(scores.mean, 6),
                       **scores.extra, "dir": f"/work/{folder.name}/"}
            where = f"/work/{folder.name}/"
        else:
            a = self._run("original", ids, request, self.work / f"fwd_{k}_original")
            b = self._run("edited", ids, request, self.work / f"fwd_{k}_edited")
            p = paired([a.per_item[i] for i in ids], [b.per_item[i] for i in ids])
            summary = {"ok": True, "model": "both", "items": len(ids), "original": round(a.mean, 6),
                       "edited": round(b.mean, 6), **a.extra,
                       "paired": {"items": p.items, "score_difference": round(p.difference, 6),
                                  "standard_error": round(p.standard_error, 6),
                                  "samples_worse_to_better": p.wrong_to_right, "samples_better_to_worse": p.right_to_wrong}}
            where = f"/work/fwd_{k}_original/ and /work/fwd_{k}_edited/"
        summary["s"] = round(time.time() - started, 1)
        return PassResult("run", where, summary)
