"""Run a language model: vLLM (or plain transformers) generates, transformers reads activations.

New code, following the behaviour the prompts describe. Hidden state l is the residual stream after layer l, with 0 the
embeddings, taken by hooks so it is the same at every layer: the final entry of transformers' own hidden_states has
already passed the final norm. MLP activations are the inputs of down_proj. Attention is captured from each attention
module, so hybrid models, where only some layers have it, index correctly.
"""

from __future__ import annotations

import functools
import gc
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .. import log
from ..upstream import home
from .base import Generation, Prompt

MAX_RESULT_BYTES = 6 << 30
TOKENS_PER_BATCH = 8192
LOGIT_CHUNK = 1024
STARTUP_RETRIES = 3
POLL_SECONDS = 2.0


def _frees_gpu(method):
    """Release GPU memory once the method has returned and its model is really gone. Doing this inside the method frees
    nothing, because its own variables still hold the model, and vLLM then starts short of memory."""

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        finally:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return wrapper


def _run_worker(command: list[str], env: dict, progress_file: str, total: int, log_file: str) -> tuple[int, str]:
    """Run a generation worker and report its progress while it works. Its output goes to a file, not a pipe, so a
    chatty engine cannot fill the pipe and stall. Returns the exit code and the output."""
    progress = log.Progress(log.current_activity(), total)
    with open(log_file, "w") as out:
        worker = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env, text=True)
        try:
            while worker.poll() is None:
                time.sleep(POLL_SECONDS)
                try:
                    with open(progress_file) as f:
                        row = json.load(f)
                except (OSError, ValueError):
                    continue  # not written yet, or caught mid-write
                progress.update(row["done"], row.get("out_tokens_per_s"))
        except BaseException:
            worker.kill()
            worker.wait()
            raise
    if worker.returncode == 0:
        progress.finish()
    with open(log_file) as f:
        return worker.returncode, f.read()


def _root_cause(stderr: str) -> str:
    """The error lines of a failed worker. vLLM's last line only says to look further up, so the tail alone explains nothing."""
    lines = [x.strip() for x in stderr.splitlines() if "Error" in x and "File " not in x and "raise " not in x]
    kept = list(dict.fromkeys(lines))[-5:]
    return "\n".join(kept) or stderr[-2000:]


def _model_dir(weights) -> Path:
    path = Path(weights)
    return path.parent if path.is_file() else path


class LM:
    """`generator` is "vllm" or "hf". `tokenizer` is the folder with the tokenizer and chat template, by default the
    weights' own."""

    def __init__(self, tokenizer: str | os.PathLike, *, thinking: bool = False, generator: str = "vllm", device: str = "cuda",
                 gpu_memory: float = 0.85, max_model_len: int | None = None, engine: dict | None = None,
                 enforce_eager: bool = False, batch_size: int = 4):
        if generator not in ("vllm", "hf"):
            raise ValueError(f"unknown generator {generator!r}")
        self.tokenizer_path = str(tokenizer)
        self.thinking = thinking
        self.generator = generator
        self.device = device
        self.gpu_memory = gpu_memory
        self.max_model_len = max_model_len
        self.engine = engine or {}
        self.enforce_eager = enforce_eager
        self.batch_size = batch_size
        self._tok = None

    @property
    def tok(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return self._tok

    def render(self, user_message: str) -> str:
        return self.tok.apply_chat_template([{"role": "user", "content": user_message}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=self.thinking)

    # Generation ------------------------------------------------------------------------------------------------------
    def generate(self, weights, prompts: list[Prompt], n: int, sampling: dict) -> list[list[Generation]]:
        return (self._vllm if self.generator == "vllm" else self._hf_generate)(weights, prompts, n, sampling)

    def _engine(self, weights) -> dict:
        """Sessions are text only, so a multimodal checkpoint's vision side is switched off."""
        config = _model_dir(weights) / "config.json"
        multimodal = config.exists() and "vision_config" in json.loads(config.read_text())
        return {**({"limit_mm_per_prompt": {"image": 0, "video": 0}} if multimodal else {}), **self.engine}

    def _vllm(self, weights, prompts, n, sampling) -> list[list[Generation]]:
        with tempfile.TemporaryDirectory() as tmp:
            request = {"weights": str(_model_dir(weights)), "tokenizer": self.tokenizer_path, "prompts": [p.text for p in prompts],
                       "seeds": [p.seed for p in prompts], "n": n, "sampling": {k: v for k, v in sampling.items() if k != "seed"},
                       "gpu_memory": self.gpu_memory, "max_model_len": self.max_model_len, "engine": self._engine(weights),
                       "enforce_eager": self.enforce_eager, "out": os.path.join(tmp, "out.jsonl")}
            request["progress"] = os.path.join(tmp, "progress.json")
            path = os.path.join(tmp, "request.json")
            with open(path, "w") as f:
                json.dump(request, f)
            error = ""
            # FlashInfer's sampler compiles a kernel, which needs nvcc; vLLM's own gives the same distribution without it.
            env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
            command = [sys.executable, "-m", "wsplab.runners.vllm_worker", path]
            for attempt in range(STARTUP_RETRIES):
                code, output = _run_worker(command, env, request["progress"], len(prompts) * n, os.path.join(tmp, "worker.log"))
                if code == 0:
                    break
                error = _root_cause(output)
                log.warning(f"the generation worker failed (attempt {attempt + 1} of {STARTUP_RETRIES}): {error.splitlines()[-1]}")
                log.debug(f"generation worker output:\n{output[-4000:]}")
            else:
                raise RuntimeError(f"generation worker failed: {error}")
            rows = [json.loads(line) for line in open(request["out"])]
        return [[Generation(x["text"], x["finish"], x["ids"]) for x in row] for row in rows]

    def _load(self, weights, attention: bool = False):
        import torch
        from transformers import AutoModelForCausalLM

        kw = {"attn_implementation": "eager"} if attention else {}
        model = AutoModelForCausalLM.from_pretrained(str(_model_dir(weights)), dtype=torch.bfloat16, **kw).to(self.device).eval()
        return model

    @_frees_gpu
    def _hf_generate(self, weights, prompts, n, sampling) -> list[list[Generation]]:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        class PresencePenalty(LogitsProcessor):
            """Subtracts a constant from every token already generated, not from the prompt, as vLLM does."""

            def __init__(self, penalty: float, start: int):
                self.penalty = penalty
                self.start = start

            def __call__(self, input_ids, scores):
                seen = torch.zeros_like(scores, dtype=torch.bool)
                seen.scatter_(1, input_ids[:, self.start:], True)
                return scores - self.penalty * seen

        model = self._load(weights)
        out: list[list[Generation]] = []
        progress = log.Progress(log.current_activity(), len(prompts) * n)
        with torch.no_grad():
            for k, p in enumerate(prompts):
                progress.update(k * n)
                torch.manual_seed(p.seed)
                ids = self.tok(p.text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                processors = LogitsProcessorList()
                if sampling.get("presence_penalty"):
                    processors.append(PresencePenalty(sampling["presence_penalty"], ids.shape[1]))
                temperature = sampling.get("temperature", 1.0)
                result = model.generate(ids, do_sample=temperature > 0, temperature=temperature or None,
                                        top_p=sampling.get("top_p", 1.0), top_k=sampling.get("top_k", 0),
                                        repetition_penalty=sampling.get("repetition_penalty", 1.0),
                                        max_new_tokens=sampling["max_tokens"], num_return_sequences=n,
                                        logits_processor=processors, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
                rows = []
                for row in result:
                    new = row[ids.shape[1]:].tolist()
                    eos = model.generation_config.eos_token_id
                    eos = set(eos if isinstance(eos, list) else [eos])
                    stop = next((i for i, t in enumerate(new) if t in eos), None)
                    kept = new if stop is None else new[:stop]
                    rows.append(Generation(self.tok.decode(kept, skip_special_tokens=True),
                                           "length" if stop is None else "stop", kept))
                out.append(rows)
        progress.finish()
        return out

    # Reading activations ---------------------------------------------------------------------------------------------
    @staticmethod
    def _parts(model):
        """The decoder layers and the embedding, found by shape rather than by architecture name."""
        from torch import nn

        wanted = (model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config).num_hidden_layers
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and name.endswith("layers") and len(module) == wanted:
                return module, model.get_input_embeddings()
        raise ValueError("could not find the decoder layers of this model")

    def _forward_capture(self, model, ids, mask, hidden: list[int], mlp: list[int], attn: list[int],
                         need_logits: bool = False) -> tuple:
        import torch

        layers, embed = self._parts(model)
        got: dict[str, object] = {}
        handles = []

        def keep(key):
            def hook(_m, _i, out):
                got[key] = out[0] if isinstance(out, tuple) else out
            return hook

        if 0 in hidden:
            handles.append(embed.register_forward_hook(keep("hidden_0")))
        for layer in hidden:
            if layer > 0:
                handles.append(layers[layer - 1].register_forward_hook(keep(f"hidden_{layer}")))
        for layer in mlp:
            down = layers[layer].mlp.down_proj
            handles.append(down.register_forward_pre_hook(lambda _m, a, k=f"mlp_act_{layer}": got.__setitem__(k, a[0])))
        for layer in attn:
            handles.append(layers[layer].self_attn.register_forward_hook(
                lambda _m, _i, o, k=f"attn_{layer}": got.__setitem__(k, o[1])))
        norm_out: dict = {}
        try:
            with torch.no_grad():
                # Asking for one position's logits keeps the output layer from running on every token at once: at a
                # 248k vocabulary and 8k tokens that alone is more than a 16 GB card holds.
                out = model(input_ids=ids, attention_mask=mask, output_attentions=bool(attn), logits_to_keep=1,
                            output_hidden_states=need_logits)
        finally:
            for h in handles:
                h.remove()
        if need_logits:
            norm_out["x"] = out.hidden_states[-1]  # after the final norm, unlike the hooks' residual stream
        return got, norm_out

    def _full_logits(self, model, norm_out, ids, rows, width_b, n, longest, topk, request, out) -> None:
        """Log-probabilities through the output layer a slice of positions at a time."""
        import torch

        head = model.get_output_embeddings()
        cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
        x = norm_out["x"]
        for r in range(ids.shape[0]):
            row = rows.start + r
            for a in range(0, width_b, LOGIT_CHUNK):
                b = min(a + LOGIT_CHUNK, width_b)
                with torch.no_grad():
                    logits = head(x[r, a:b]).float()
                    if cap:
                        logits = torch.tanh(logits / cap) * cap
                    logprobs = torch.log_softmax(logits, -1)
                if request.get("token_logprobs"):
                    stop = min(b, width_b - 1)
                    if stop > a:
                        token = logprobs[:stop - a].gather(-1, ids[r, a + 1:stop + 1, None])[:, 0]
                        out.setdefault("token_logprobs", np.zeros((n, longest - 1), np.float32))[row, a:stop] = token.cpu().numpy()
                if topk:
                    top = logprobs.topk(topk, -1)
                    out.setdefault("topk_logprobs", np.zeros((n, longest, topk), np.float32))[row, a:b] = top.values.cpu().numpy()
                    out.setdefault("topk_ids", np.zeros((n, longest, topk), np.int64))[row, a:b] = top.indices.cpu().numpy()

    def _in_worker(self, method: str, *args) -> dict[str, np.ndarray]:
        """Run a read in a process of its own, as vLLM does, so the parent never holds a CUDA context: vLLM needs nearly all of
        the card next."""
        folder = home() / "tmp"
        folder.mkdir(parents=True, exist_ok=True)
        options = {"tokenizer": self.tokenizer_path, "thinking": self.thinking, "device": self.device,
                   "batch_size": self.batch_size}
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            path = os.path.join(tmp, "request.pkl")
            with open(path, "wb") as f:
                pickle.dump({"options": options, "method": method, "args": args, "out": os.path.join(tmp, "out.npz")}, f)
            done = subprocess.run([sys.executable, "-m", "wsplab.runners.hf_worker", path], capture_output=True, text=True)
            if done.returncode:
                raise RuntimeError(f"read worker failed: {_root_cause(done.stderr)}")
            with np.load(os.path.join(tmp, "out.npz"), allow_pickle=False) as z:
                return {k: z[k] for k in z.files}

    def read(self, weights, texts: list[str], request: dict) -> dict[str, np.ndarray]:
        return self._in_worker("read_here", str(weights), texts, request)

    def activations(self, weights, prompts: list[Prompt], answers: list[Generation], layers: list[int], mlp: bool,
                    max_tokens: int) -> dict[str, np.ndarray]:
        return self._in_worker("activations_here", str(weights), prompts, answers, layers, mlp, max_tokens)

    @_frees_gpu
    def read_here(self, weights, texts: list[str], request: dict) -> dict[str, np.ndarray]:
        import torch

        hidden, mlp = list(request.get("hidden_states") or []), list(request.get("mlp_act") or [])
        attn = list(request.get("attn_layers") or [])
        topk = int(request.get("topk") or 0)
        encoded = [self.tok(t, add_special_tokens=False).input_ids for t in texts]
        lengths = np.array([len(e) for e in encoded])
        n, longest = len(encoded), int(lengths.max())
        model = self._load(weights, attention=bool(attn))
        width = model.config.get_text_config().hidden_size
        estimate = n * longest * 4 * (width * len(hidden) + 2 * topk + 1) + n * longest * longest * 4 * len(attn)
        if estimate > MAX_RESULT_BYTES:
            raise ValueError(f"the result would take {estimate / 2**30:.1f} GB, more than the 6 GB limit of a pass")
        out: dict[str, np.ndarray] = {"lengths": lengths, "input_ids": np.zeros((n, longest), np.int64)}
        i = 0
        while i < n:
            size = self.batch_size
            while size > 1 and size * max(len(e) for e in encoded[i:i + size]) > TOKENS_PER_BATCH:
                size -= 1
            batch = encoded[i:i + size]
            start, i = i, i + len(batch)
            width_b = max(len(e) for e in batch)
            ids = torch.zeros(len(batch), width_b, dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for r, e in enumerate(batch):
                ids[r, :len(e)] = torch.tensor(e, device=self.device)
                mask[r, :len(e)] = 1
            want_logits = bool(request.get("token_logprobs") or topk)
            got, norm_out = self._forward_capture(model, ids, mask, hidden, mlp, attn, want_logits)
            rows = slice(start, start + len(batch))
            out["input_ids"][rows, :width_b] = ids.cpu().numpy()
            if want_logits:
                self._full_logits(model, norm_out, ids, rows, width_b, n, longest, topk, request, out)
            for key, value in got.items():
                array = value.float().cpu().numpy()
                if key.startswith("attn_"):
                    target = out.setdefault(key, np.zeros((n, array.shape[1], longest, longest), np.float32))
                    target[rows, :, :width_b, :width_b] = array
                else:
                    target = out.setdefault(key, np.zeros((n, longest, array.shape[-1]), np.float32))
                    target[rows, :width_b] = array
        return out

    @_frees_gpu
    def activations_here(self, weights, prompts: list[Prompt], answers: list[Generation], layers: list[int], mlp: bool,
                         max_tokens: int) -> dict[str, np.ndarray]:
        """Works on the token ids the model generated, never on retokenized text, which can differ."""
        import torch

        model = self._load(weights)
        mlp_layers = [x for x in layers if x < self._parts(model)[0].__len__()] if mlp else []
        sums: dict[str, list[np.ndarray]] = {"hidden_last_prompt": [], "hidden_mean_answer": []}
        if mlp:
            sums.update(mlp_last_prompt=[], mlp_mean_answer=[])
        for prompt, answer in zip(prompts, answers):
            head = self.tok(prompt.text, add_special_tokens=False).input_ids
            tail = answer.token_ids[:max_tokens]
            ids = torch.tensor([head + tail], device=self.device)
            got, _ = self._forward_capture(model, ids, torch.ones_like(ids), layers, mlp_layers, [])
            last = len(head) - 1

            def gather(names):
                at_prompt = torch.stack([got[k][0, last].float() for k in names]).cpu().numpy()
                means = []
                for k in names:
                    answer_part = got[k][0, last + 1:].float()
                    means.append(answer_part.mean(0) if answer_part.shape[0] else torch.zeros_like(got[k][0, last].float()))
                return at_prompt, torch.stack(means).cpu().numpy()

            a, b = gather([f"hidden_{x}" for x in layers])
            sums["hidden_last_prompt"].append(a)
            sums["hidden_mean_answer"].append(b)
            if mlp:
                a, b = gather([f"mlp_act_{x}" for x in mlp_layers])
                sums["mlp_last_prompt"].append(a)
                sums["mlp_mean_answer"].append(b)
        return {k: np.stack(v) for k, v in sums.items()}
