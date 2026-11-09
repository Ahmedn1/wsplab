"""The forward-pass tool for language models: the ```forward block of the paper's prompts.

The server was not released. This follows the JSON spec in the prompts and the summaries in requests/fwd_*.done.json.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..eval.graders import Grader
from ..eval.stats import paired
from ..runners.base import Generation, LMRunner, Prompt
from .base import PassFailed, PassInvalid, PassResult


@dataclass
class ForwardLimits:
    n_layers: int
    full_attention: tuple[int, ...]
    gen_max: int = 512
    max_tokens: int = 32768
    tf_max_texts: int = 1024
    tf_max_tokens: int = 32768
    tf_total_tokens: int = 524288
    act_max_tokens: int = 8192
    topk_max: int = 50
    attn_max_texts: int = 2
    attn_max_tokens: int = 2048


class GenerationCache:
    """Answers already generated in this run, one entry per item.

    Seeds are fixed by the item, so asking the same weights the same question repeats the same answer, and most of a paired
    pass used to go on regenerating the original model's answers. An entry is reused only when everything that determines
    the answer matches: see `key`. It lives outside /work, so an editor cannot change the answers it is compared against, and
    on disk, so a resumed run keeps it. The final evaluation never reads it.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, list[Generation]] = {}
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:  # the last line of a run that was killed mid-write
                        continue
                    self.entries[row["key"]] = [Generation(g["text"], g["finish"], g["ids"]) for g in row["samples"]]

    @staticmethod
    def key(weights: str, prompt: Prompt, n: int, sampling: dict, generator: str) -> str:
        fields = {"weights": weights, "text": prompt.text, "seed": prompt.seed, "n": n, "sampling": sampling, "generator": generator}
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> list[Generation] | None:
        return self.entries.get(key)

    def put(self, key: str, samples: list[Generation]) -> None:
        self.entries[key] = samples
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"key": key, "samples": [{"text": g.text, "finish": g.finish_reason, "ids": g.token_ids}
                                                        for g in samples]}) + "\n")


def item_seed(item_id: str) -> int:
    """The sampling seed fixed by the item. Inferred: the prompts promise this, without giving the formula."""
    return int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:4], "big")


def summarize(records: list[dict]) -> dict:
    """Accuracy, length and failure counts of graded answers, as in the logged summaries."""
    n = len(records)
    good = sum(r["correct"] for r in records)
    by_source: dict[str, list[int]] = {}
    for r in records:
        by_source.setdefault(r.get("source", "other"), []).append(r["correct"])
    cut = sum(r["finish"] == "length" for r in records)
    return {"generations": n, "graded": n, "accuracy": good / n if n else 0.0,
            "accuracy_by_source": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_source.items())},
            "mean_tokens": round(sum(r["tokens"] for r in records) / n) if n else 0,
            "cut_share": round(cut / n, 4) if n else 0.0,
            "failures": {"wrong_answer": sum(not r["correct"] and r["finish"] != "length" and r["pred"] is not None
                                             for r in records),
                         "no_answer_finished": sum(not r["correct"] and r["finish"] != "length" and r["pred"] is None
                                                   for r in records),
                         "cut_at_max_tokens": sum(not r["correct"] and r["finish"] == "length" for r in records)}}


class LMForward:
    fence = "forward"

    def __init__(self, runner: LMRunner, *, practice: dict[str, dict], grader: Grader, limits: ForwardLimits,
                 sampling: dict, work: str | Path, base: Path, edited: Callable[[], Path],
                 edit_digest: Callable[[], str | None], user_message: Callable[[dict], str],
                 cache: GenerationCache | None = None):
        self.runner = runner
        self.practice = practice
        self.grader = grader
        self.limits = limits
        self.sampling = dict(sampling)
        self.work = Path(work)
        self.base = Path(base)
        self.edited = edited
        self.edit_digest = edit_digest
        self.user_message = user_message
        self.cache = cache
        self._seen: set[str] = set()

    # Checking a request ----------------------------------------------------------------------------------------------
    def parse(self, body: str) -> dict:
        try:
            request = json.loads(body)
        except json.JSONDecodeError as e:
            raise PassInvalid(f"JSONDecodeError: {e}") from e
        if not isinstance(request, dict) or request.get("type") not in ("generate", "teacher_forced"):
            raise PassInvalid('"type" must be "generate" or "teacher_forced"')
        return request

    def _items(self, request: dict) -> list[tuple[str, dict]]:
        ids = request.get("problems") or []
        missing = [i for i in ids if i not in self.practice]
        if missing:
            raise PassFailed(f"ValueError: unknown practice ids: {missing[:10]}")
        items = [(i, self.practice[i]) for i in ids]
        items += [(f"prompt_{k}", {"id": f"prompt_{k}", "own_prompt": p}) for k, p in enumerate(request.get("prompts") or [])]
        if not items:
            raise PassInvalid('give "problems" or "prompts"')
        return items

    def _check_generate(self, request: dict) -> None:
        lim = self.limits
        model = request.get("model")
        if model not in ("original", "edited", "both"):
            raise PassInvalid('"model" must be "original", "edited" or "both"')
        n = request.get("n", 1)
        if not isinstance(n, int) or n < 1:
            raise PassInvalid('"n" must be a positive integer')
        total = n * (len(request.get("problems") or []) + len(request.get("prompts") or []))
        limit = lim.gen_max // 2 if model == "both" else lim.gen_max
        if total > limit:
            raise PassInvalid(f"{total} generations is more than the limit of {limit}")
        if request.get("max_tokens", 1) > lim.max_tokens:
            raise PassInvalid(f'"max_tokens" is at most {lim.max_tokens}')
        acts = request.get("activations")
        if acts:
            self._check_layers(acts.get("layers", []), lim.n_layers)

    def _check_layers(self, layers: list[int], top: int) -> None:
        bad = [x for x in layers if not isinstance(x, int) or not 0 <= x <= top]
        if bad:
            raise PassInvalid(f"layers out of range 0..{top}: {bad}")

    def _check_teacher_forced(self, request: dict) -> None:
        lim = self.limits
        if request.get("model") not in ("original", "edited"):
            raise PassInvalid('"model" must be "original" or "edited"')
        texts = request.get("texts", request.get("chats"))
        if not texts:
            raise PassInvalid('give "texts" or "chats"')
        if len(texts) > lim.tf_max_texts:
            raise PassInvalid(f"{len(texts)} texts is more than the limit of {lim.tf_max_texts}")
        ret = request.get("return") or {}
        if ret.get("topk", 0) > lim.topk_max:
            raise PassInvalid(f'"topk" is at most {lim.topk_max}')
        self._check_layers(ret.get("hidden_states", []), lim.n_layers)
        self._check_layers(ret.get("mlp_act", []), lim.n_layers - 1)
        attn = ret.get("attn_layers") or []
        wrong = [x for x in attn if x not in lim.full_attention]
        if wrong:
            raise PassInvalid(f"attn_layers must be full-attention layers {list(lim.full_attention)}, not {wrong}")
        if attn and len(texts) > lim.attn_max_texts:
            raise PassInvalid(f"attn_layers works on at most {lim.attn_max_texts} texts")

    def _check_repeat(self, body: str) -> None:
        """An identical request with unchanged edits is refused without cost."""
        canonical = json.dumps(json.loads(body), sort_keys=True)
        key = hashlib.sha1(f"{canonical}|{self.edit_digest()}".encode()).hexdigest()
        if key in self._seen:
            raise PassInvalid("this request is identical to an earlier one with unchanged edits")
        self._seen.add(key)

    # Running ---------------------------------------------------------------------------------------------------------
    def run(self, body: str, request_number: int) -> PassResult:
        request = self.parse(body)
        started = time.time()
        if request["type"] == "generate":
            self._check_generate(request)
            items = self._items(request)
            self._check_repeat(body)
            result = self._generate(request, items, request_number)
        else:
            self._check_teacher_forced(request)
            self._check_repeat(body)
            result = self._teacher_forced(request, request_number)
        if isinstance(result.summary, dict):
            result.summary["s"] = round(time.time() - started, 1)
        return result

    def _weights(self, model: str) -> Path:
        return self.base if model == "original" else self.edited()

    def _prompts(self, items: list[tuple[str, dict]]) -> list[Prompt]:
        out = []
        for item_id, row in items:
            message = row["own_prompt"] if "own_prompt" in row else self.user_message(row)
            out.append(Prompt(item_id, self.runner.render(message), item_seed(item_id)))
        return out

    def _fingerprint(self, model: str) -> str:
        """What identifies these weights: the base model's files, or the hash of the current edit."""
        if model == "edited" and self.edit_digest():
            return f"edit:{self.edit_digest()}"
        files = sorted(self.base.glob("*.safetensors")) if self.base.is_dir() else [self.base]
        return "base:" + ",".join(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files if f.exists())

    def _generate_with_cache(self, model: str, prompts: list[Prompt], n: int, sampling: dict) -> tuple[list, int]:
        """All answers, and how many items came from the cache. Weights are only built if something has to be generated."""
        if self.cache is None:
            return self.runner.generate(self._weights(model), prompts, n, sampling), 0
        fingerprint = self._fingerprint(model)
        generator = str(getattr(self.runner, "generator", ""))
        keys = [GenerationCache.key(fingerprint, p, n, sampling, generator) for p in prompts]
        found = [self.cache.get(k) for k in keys]
        missing = [i for i, samples in enumerate(found) if samples is None]
        if missing:
            fresh = self.runner.generate(self._weights(model), [prompts[i] for i in missing], n, sampling)
            for i, samples in zip(missing, fresh):
                found[i] = samples
                self.cache.put(keys[i], samples)
        return found, len(prompts) - len(missing)

    def _run_model(self, model: str, prompts, items, request, name: str) -> tuple[dict, list[list[dict]], list]:
        sampling = {**self.sampling, "max_tokens": request.get("max_tokens", self.sampling.get("max_tokens"))}
        try:
            outputs, reused = self._generate_with_cache(model, prompts, request.get("n", 1), sampling)
        except Exception as e:  # noqa: BLE001 - any failure of the backend must not cost the editor a pass
            raise PassFailed(f"{type(e).__name__}: {e}") from e
        per_item: list[list[dict]] = []
        for (item_id, row), samples in zip(items, outputs):
            rows = []
            for k, g in enumerate(samples):
                grade = self.grader.grade(g.text, row) if "answer" in row else None
                rows.append({"id": item_id, "source": row.get("source", "own"), "sample": k, "text": g.text,
                             "pred": grade.pred if grade else None, "correct": bool(grade and grade.correct),
                             "finish": g.finish_reason, "tokens": g.tokens})
            per_item.append(rows)
        with open(self.work / name, "w") as f:
            for rows in per_item:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        flat = [r for rows in per_item for r in rows]
        return {**summarize(flat), "reused_items": reused}, per_item, outputs

    def _save_activations(self, model, prompts, outputs, acts, folder: Path) -> dict:
        folder.mkdir(parents=True, exist_ok=True)
        repeated = [p for p, samples in zip(prompts, outputs) for _ in samples]
        answers: list[Generation] = [g for samples in outputs for g in samples]
        arrays = self.runner.activations(self._weights(model), repeated, answers, acts.get("layers", []),
                                         bool(acts.get("mlp")), self.limits.act_max_tokens)
        for name, array in arrays.items():
            np.save(folder / f"{name}.npy", array)
        (folder / "index.json").write_text(json.dumps({"samples": [{"id": p.id} for p in repeated],
                                                       "layers": acts.get("layers", [])}))
        return {"dir": f"/work/{folder.name}/", "files": {f"{k}.npy": list(v.shape) for k, v in arrays.items()}}

    def _generate(self, request: dict, items, k: int) -> PassResult:
        prompts = self._prompts(items)
        model = request["model"]
        acts = request.get("activations")
        if model != "both":
            name = f"gen_{k}.jsonl"
            summary, _, outputs = self._run_model(model, prompts, items, request, name)
            info = self._save_activations(model, prompts, outputs, acts, self.work / f"gen_{k}_acts") if acts else None
            head = {"activations": info} if acts else {}
            return PassResult("generate", f"/work/{name}", {**head, "ok": True, "type": "generate", "model": model,
                                                            "file": f"/work/{name}", **summary})
        names = {m: f"gen_{k}_{m}.jsonl" for m in ("original", "edited")}
        a, per_a, out_a = self._run_model("original", prompts, items, request, names["original"])
        b, per_b, out_b = self._run_model("edited", prompts, items, request, names["edited"])
        info = None
        if acts:
            info = {m: self._save_activations(m, prompts, o, acts, self.work / f"gen_{k}_{m}_acts")
                    for m, o in (("original", out_a), ("edited", out_b))}
        graded = [(x, y) for x, y in zip(per_a, per_b) if "answer" in dict(items)[x[0]["id"]]]
        result = paired([[float(r["correct"]) for r in x] for x, _ in graded],
                        [[float(r["correct"]) for r in y] for _, y in graded]) if graded else None
        pair = None if result is None else {
            "items": result.items, "accuracy_difference": round(result.difference, 4),
            "standard_error": round(result.standard_error, 4), "samples_wrong_to_right": result.wrong_to_right,
            "samples_right_to_wrong": result.right_to_wrong}
        files = {m: f"/work/{n}" for m, n in names.items()}
        summary = {"ok": True, "type": "generate", "model": "both", "files": files, "activations": info, "original": a,
                   "edited": b, "paired": pair}
        return PassResult("generate", f"{files['original']} and {files['edited']}", summary)

    def _teacher_forced(self, request: dict, k: int) -> PassResult:
        texts = request.get("texts") or [self._chat(c, request) for c in request["chats"]]
        ret = request.get("return") or {}
        try:
            arrays = self.runner.read(self._weights(request["model"]), texts, ret)
        except Exception as e:  # noqa: BLE001
            raise PassFailed(f"{type(e).__name__}: {e}") from e
        folder = self.work / f"fwd_{k}"
        folder.mkdir(parents=True, exist_ok=True)
        lengths = arrays.pop("lengths")
        for name, array in arrays.items():
            np.save(folder / f"{name}.npy", array)
        (folder / "meta.json").write_text(json.dumps({"lengths": [int(x) for x in lengths]}))
        logprobs = arrays.get("token_logprobs")
        summary = {"ok": True, "lengths": [int(x) for x in lengths],
                   "files": {name: list(a.shape) for name, a in arrays.items()},
                   "edited_tensors": 0 if request["model"] == "original" else len(self._edit_names())}
        if logprobs is not None:
            summary["mean_token_logprob"] = [float(logprobs[i, :int(n) - 1].mean()) for i, n in enumerate(lengths)]
        summary["type"] = "teacher_forced"
        return PassResult("teacher_forced", f"/work/fwd_{k}/", summary)

    def _chat(self, chat: list[dict], request: dict) -> str:
        text = chat[-1]["content"] if chat else ""
        return self.runner.render(text)

    def _edit_names(self) -> list[str]:
        folder = self.work / "edited"
        return [f.name for f in folder.glob("*.npy")] if folder.is_dir() else []
