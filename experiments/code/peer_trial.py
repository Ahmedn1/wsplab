"""Budgeted live peer/solo comparison. Explicit CLI launch makes paid API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from wsplab.editors import BudgetExceeded, Spending
from wsplab.editors.claude import AnthropicEditor
from wsplab.editors.llm import ChatEditor, OpenAIEditor

MODELS = {"openai": "gpt-6.1-terra", "anthropic": "claude-opus-5-5", "gemini": "gemini-4-argon"}
# USD/MTok. Anthropic fresh input is conservatively charged at its cache-WRITE
# rate, since Usage combines fresh input and cache creation. This is an upper estimate.
PRICES = {"openai:gpt-6.1-terra": (1., 5., .05),
          "anthropic:claude-opus-5-5": (5., 20., .2), "gemini:gemini-4-argon": (2., 10., .1)}
CAPS = {"openai": .75, "anthropic": 2., "gemini": 3.5}
MAX_OUTPUT = 32768


class Ledger:
    def __init__(self, path, caps):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.caps = caps
        self.spent = {provider: 0. for provider in caps}
        self.reserved = {provider: 0. for provider in caps}
        self.calls = []

    def save(self):
        data = {"caps": self.caps, "spent": self.spent, "reserved": self.reserved, "calls": self.calls}
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n")
        temp.replace(self.path)

    def reserve(self, provider, amount):
        with self.lock:
            if self.spent[provider] + self.reserved[provider] + amount > self.caps[provider]:
                raise BudgetExceeded(f"{provider} request would exceed its reserved ${self.caps[provider]:.2f} allowance")
            self.reserved[provider] += amount
            self.save()

    def available(self, provider):
        with self.lock:
            return self.caps[provider] - self.spent[provider] - self.reserved[provider]

    def finish(self, provider, reserved, usage, elapsed, error=None, reported_usage=None):
        with self.lock:
            price_in, price_out, cached = PRICES[f"{provider}:{MODELS[provider]}"]
            # An API error may still have been billed. Keep the full reservation
            # as an estimated charge instead of silently freeing it for retries.
            cost = reserved if usage is None else (
                (usage.input_tokens - usage.cached_tokens) * price_in + usage.cached_tokens * cached
                + usage.output_tokens * price_out) / 1e6
            self.reserved[provider] -= reserved
            self.spent[provider] += cost
            self.calls.append({"provider": provider, "usd_estimate": cost,
                               "usage": asdict(usage) if usage else None, "seconds": elapsed,
                               "reported_usage": asdict(reported_usage) if reported_usage else None,
                               "usage_was_estimated": reported_usage is not None,
                               "error": error, "at": time.time()})
            self.save()


class Guarded:
    def __init__(self, editor, provider, ledger, max_output=MAX_OUTPUT):
        self.editor, self.provider, self.ledger = editor, provider, ledger
        self.max_output = max_output
        self.name = editor.name
        self.last_system = None
        self.last_messages = ()
        self.last_input_tokens = None

    def input_bound(self, system, messages):
        current = tuple((m["role"], m["content"]) for m in messages)
        if (self.last_input_tokens is not None and system == self.last_system
                and current[:len(self.last_messages)] == self.last_messages):
            added = current[len(self.last_messages):]
            return self.last_input_tokens + sum(len(text.encode()) + 100 for _, text in added) + 1024
        return len(system.encode()) + sum(len(text.encode()) + 100 for _, text in current) + 4096

    def reply(self, system, messages):
        # A deliberately conservative input-token bound, with protocol overhead.
        size = self.input_bound(system, messages)
        # Stay below long-context pricing tiers and explicitly report cap truncation.
        if size > 180_000:
            raise BudgetExceeded("conservative context bound exceeded; full peer text was not silently truncated")
        rates = PRICES[self.name]
        output = min(self.max_output, int((self.ledger.available(self.provider) * 1e6 - size * rates[0]) / rates[1]))
        if output < 1024:
            raise BudgetExceeded(f"{self.provider} has insufficient budget for another bounded response")
        if self.provider == "anthropic":
            self.editor.max_tokens = output
        else:
            self.editor.params["max_output_tokens" if self.provider == "openai" else "max_tokens"] = output
        reserve = (size * rates[0] + output * rates[1]) / 1e6
        self.ledger.reserve(self.provider, reserve)
        start = time.monotonic()
        try:
            reply = self.editor.reply(system, messages)
        except Exception as exc:
            self.ledger.finish(self.provider, reserve, None, time.monotonic() - start, type(exc).__name__)
            raise
        reported = None
        if reply.text.strip() and reply.usage.output_tokens == 0:
            # A non-empty long Gemini reply has been observed with zero reported
            # output. Do not silently treat missing usage as free generation.
            reported = reply.usage
            reply.usage = replace(reply.usage, output_tokens=output)
        self.ledger.finish(self.provider, reserve, reply.usage, time.monotonic() - start, reported_usage=reported)
        self.last_system = system
        self.last_messages = tuple((m["role"], m["content"]) for m in messages)
        self.last_input_tokens = reply.usage.input_tokens or None
        return reply


def editor(provider):
    key = Path(f"/tmp/{provider}_key.txt").read_text().strip()
    if provider == "openai":
        return OpenAIEditor(MODELS[provider], api_key=key, effort="high", max_output_tokens=MAX_OUTPUT,
                            max_retries=0, timeout=900)
    if provider == "anthropic":
        return AnthropicEditor(MODELS[provider], api_key=key, effort="high", max_tokens=MAX_OUTPUT, max_retries=0)
    result = ChatEditor(MODELS[provider], api_key=key,
                        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                        reasoning_effort="high", max_tokens=MAX_OUTPUT, max_retries=0, timeout=900)
    result.name = f"gemini:{MODELS[provider]}"
    return result


def main():
    if __package__:
        from .modular_job import run_benchmark
    else:
        from modular_job import run_benchmark

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["peers", "independent"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--provider", choices=list(MODELS), help="One independent solo session only")
    parser.add_argument("--providers", choices=list(MODELS), nargs="+")
    parser.add_argument("--anthropic-cap", type=float, default=CAPS["anthropic"])
    parser.add_argument("--strict-counterfactuals", action="store_true")
    args = parser.parse_args()
    if args.provider and args.arm != "independent":
        parser.error("--provider is only for an independent solo session")
    if args.provider and args.providers:
        parser.error("choose --provider or --providers")
    if args.anthropic_cap <= 0:
        parser.error("caps must be positive")
    providers = [args.provider] if args.provider else (args.providers or list(MODELS))
    if len(set(providers)) != len(providers):
        parser.error("duplicate providers")
    caps = {provider: CAPS[provider] for provider in providers}
    if "anthropic" in caps:
        caps["anthropic"] = args.anthropic_cap
    extra_note = ("\nClarification of the existing no-simulation rule: do NOT compute predictions, logits, losses, "
                  "accuracy or AUC for modified weights inside Python, even by replaying only the last layer(s) "
                  "from saved activations. That is a counterfactual forward pass and must use the official forward tool. "
                  "Inspecting weights, correlations and already-recorded outputs is allowed. Save a plausible candidate "
                  "early; analysis that never reaches save_edit cannot repair the model.") if args.strict_counterfactuals else ""
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {"arm": args.arm, "seed": args.seed, "models": MODELS, "providers": providers, "caps": caps, "prices": PRICES,
            "guard": "reported-prefix-tokens-v2",
            "output_budget_policy": "at most 32768; reduced near the monetary cap, minimum 1024",
            "extra_note": extra_note,
            "effort": "high", "max_output_tokens": MAX_OUTPUT, "max_runs": 40, "passes": 6,
            "created_at": time.time(), "task_sha256": hashlib.sha256(
                Path(__file__).with_name("modular_net.py").read_bytes()).hexdigest()}
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    ledger = Ledger(args.out / "ledger.json", caps)
    editors = [Guarded(editor(provider), provider, ledger) for provider in providers]
    spending = Spending(max_usd=sum(caps.values()), prices=PRICES)
    start = time.monotonic()
    try:
        result = run_benchmark(editors=editors, spending=spending, folder=args.out / "run",
                               shared=args.arm == "peers", seed=args.seed, extra_note=extra_note)
        summary = {"arm": args.arm, "status": "completed", "verdict": result.result.verdict,
                   "baseline": result.result.baseline, "final": result.result.final,
                   "candidates": result.candidates, "selection": result.selection,
                   "seconds": time.monotonic() - start, "spend": ledger.spent}
    except Exception as exc:
        # Avoid serializing exception messages which may include credential-bearing URLs.
        summary = {"arm": args.arm, "status": "failed", "error_type": type(exc).__name__,
                   "seconds": time.monotonic() - start, "spend": ledger.spent}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
