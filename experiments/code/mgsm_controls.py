"""Frozen, resumable local MGSM controls. No editors, training or provider calls."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

if __package__:
    from .mgsm_screen import INSTRUCTION, SEED, digest, normalize_digits, prepare
    from .monitor_run import process_identity
else:
    from mgsm_screen import INSTRUCTION, SEED, digest, normalize_digits, prepare
    from monitor_run import process_identity

CONDITIONS = {
    "nonthinking": {"thinking": False, "presence_penalty": 0., "instruction": "standard"},
    "thinking": {"thinking": True, "presence_penalty": 0., "instruction": "standard"},
    "nonthinking_pp": {"thinking": False, "presence_penalty": 1.5, "instruction": "standard"},
    "thinking_pp": {"thinking": True, "presence_penalty": 1.5, "instruction": "standard"},
    "english_reason": {"thinking": False, "presence_penalty": 1.5, "instruction": "english_reason"},
    "localized": {"thinking": False, "presence_penalty": 1.5, "instruction": "localized"},
    "translate": {"thinking": False, "presence_penalty": 1.5, "instruction": "translate"},
}
LOCALIZED = {
    "en": INSTRUCTION,
    "bn": "ধাপে ধাপে যুক্তি দিয়ে সমাধান করুন এবং চূড়ান্ত উত্তরটি \\boxed{}-এর মধ্যে লিখুন।",
    "sw": "Tafadhali tatua hatua kwa hatua, na uweke jibu lako la mwisho ndani ya \\boxed{}.",
}
FULL_ORDER = ["nonthinking_pp", "english_reason", "localized", "thinking_pp",
              "nonthinking", "thinking", "translate"]
CONTEXT = 32768


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def instruction(row, condition):
    style = condition["instruction"]
    if style == "translate":
        return ("Translate the following math word problem into English. Preserve all quantities "
                "and relationships. Do not solve it or give its answer.\n\n" + row["problem"])
    ending = LOCALIZED[row["language"]] if style == "localized" else INSTRUCTION
    if style == "english_reason":
        ending = "Read the question carefully, then reason and answer in English. " + ending
    return row["problem"] + "\n" + ending


def grade_response(text, condition, row):
    from wsplab.eval import MathGrader

    if condition["instruction"] == "translate":
        return {"strict_correct": None, "lenient_correct": None,
                "thinking_unfinished": False, "review": "manual_translation_review_required"}
    unfinished = condition["thinking"] and "</think>" not in text
    final = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    final = normalize_digits(final)
    return {"strict_correct": bool(not unfinished and MathGrader().grade(final, row).correct),
            "lenient_correct": bool(not unfinished and MathGrader(lenient=True).grade(final, row).correct),
            "thinking_unfinished": unfinished}


def presence_processor(penalty, prompt_length):
    import torch
    from transformers import LogitsProcessor

    class Presence(LogitsProcessor):
        def __call__(self, input_ids, scores):
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, input_ids[:, prompt_length:], True)
            return scores - penalty * seen
    return Presence()


def load_records(path, expected_ids):
    if not path.exists():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)) or not set(ids) <= set(expected_ids):
        raise ValueError("Duplicate or unexpected response IDs; preserve and inspect the run")
    return records


def summarize(records, condition, expected):
    languages = {}
    for language in ("en", "bn", "sw"):
        subset = [r for r in records if r["language"] == language]
        if not subset:
            continue
        languages[language] = {"n": len(subset), "truncated": sum(r["truncated"] for r in subset),
                               "mean_tokens": sum(r["tokens"] for r in subset) / len(subset)}
        if condition["instruction"] != "translate":
            languages[language].update(
                correct=sum(r["strict_correct"] for r in subset),
                accuracy=sum(r["strict_correct"] for r in subset) / len(subset),
                extraction_disagreements=sum(r["strict_correct"] != r["lenient_correct"] for r in subset))
    return {"completed": len(records) == expected, "responses": len(records), "expected": expected,
            "languages": languages, "condition": condition}


def wait_gpu(out, protected_pid, hours):
    identity = process_identity(protected_pid) if protected_pid else None
    deadline, idle = time.monotonic() + hours * 3600, 0
    while time.monotonic() < deadline:
        protected_alive = bool(identity and process_identity(protected_pid) == identity)
        result = subprocess.run(["nvidia-smi", "-i", "0",
                                 "--query-gpu=memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20, check=True)
        free, utilization = [int(x.strip()) for x in result.stdout.strip().split(",")]
        idle = idle + 1 if not protected_alive and free >= 14000 and utilization < 10 else 0
        write_json(out / "status.json", {"status": "waiting_for_gpu", "protected_pid": protected_pid,
                   "protected_process_alive": protected_alive, "free_mib": free,
                   "gpu_utilization": utilization, "idle_observations": idle, "at": time.time()})
        if idle >= 2:
            return
        time.sleep(30)
    raise TimeoutError("GPU never became available; no inference started")


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root, out = args.fixture.resolve(), args.out.resolve()
    prepare(root)  # Verify the original frozen files; never rebuild or repartition.
    if not 1 <= args.groups <= 32 or not 1 <= args.max_tokens <= CONTEXT:
        raise ValueError("Only the frozen 32-group pilot and supported context may be used")
    names = args.conditions.split(",")
    if len(names) != len(set(names)) or not set(names) <= set(CONDITIONS):
        raise ValueError("Unknown or duplicate conditions")
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    screen_source = source.with_name("mgsm_screen.py")
    monitor_source = source.with_name("monitor_run.py")
    metadata = {"fixture_plan_sha256": digest(root / "plan.json"), "fixture": str(root),
                "script_sha256": digest(source), "screen_script_sha256": digest(screen_source),
                "monitor_script_sha256": digest(monitor_source),
                "groups": args.groups, "max_output_tokens": args.max_tokens,
                "context_limit": CONTEXT, "backend": args.backend, "threads": args.threads,
                "active_hour_limit": args.run_hours,
                "conditions": {n: CONDITIONS[n] for n in names}, "order": names,
                "localized_instructions": LOCALIZED, "localized_instruction_author": "assistant; not native-speaker verified",
                "seed": SEED, "paid_api_calls": 0,
                "versions": {p: version(p) for p in ["torch", "transformers"]},
                "gpu_engine": {"memory_fraction": .70, "max_num_seqs": 2,
                               "enforce_eager": True, "max_num_batched_tokens": 2048},
                "qualification_scope": "Only full 32-group, adequate-length GPU results; CPU is feasibility only"}
    if args.backend == "vllm":
        metadata["versions"]["vllm"] = version("vllm")
    meta_path = out / "execution.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Run settings or code changed; preserve the previous run")
    write_json(meta_path, metadata)
    for p in [source, screen_source, monitor_source]:
        (out / p.name).write_bytes(p.read_bytes())
    write_json(out / "process.json", {"pid": os.getpid(), "at": time.time(), "paid_api_calls": 0})
    if args.backend == "vllm":
        wait_gpu(out, args.wait_pid, args.wait_hours)
        prepare(root)  # Recheck after a potentially long wait.
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        model = LLM(model=str(root / "model"), tokenizer=str(root / "model"), dtype="bfloat16",
                    gpu_memory_utilization=.70, max_model_len=CONTEXT, enforce_eager=True,
                    seed=0, max_num_seqs=2, max_num_batched_tokens=2048)
    else:
        from transformers import LogitsProcessorList
        model = AutoModelForCausalLM.from_pretrained(root / "model", local_files_only=True,
                    trust_remote_code=False, dtype=torch.float32).to("cpu").eval()
    pilot = [json.loads(s) for s in (root / "pilot.jsonl").read_text().splitlines()][:args.groups * 3]
    all_summaries = {}
    started = time.monotonic()
    for name in names:
        condition = CONDITIONS[name]
        rows = [r for r in pilot if name != "translate" or r["language"] != "en"]
        path = out / f"{name}.jsonl"
        records = load_records(path, [r["id"] for r in rows])
        completed = {r["id"] for r in records}
        todo = [r for r in rows if r["id"] not in completed]
        batch_size = 2 if args.backend == "vllm" else 1
        for start in range(0, len(todo), batch_size):
            if time.monotonic() - started >= args.run_hours * 3600:
                all_summaries[name] = summarize(records, condition, len(rows))
                write_json(out / "summary.json", {"status": "stopped_time_limit",
                           "conditions": all_summaries, "paid_api_calls": 0,
                           "qualification": "incomplete_do_not_claim_qualified"})
                write_json(out / "status.json", {"status": "stopped_time_limit", "at": time.time()})
                return
            batch = todo[start:start + batch_size]
            prompts = [tokenizer.apply_chat_template([{"role": "user", "content": instruction(r, condition)}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=condition["thinking"]) for r in batch]
            lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
            limits = [min(args.max_tokens, CONTEXT - size, 2048 if name == "translate" else CONTEXT)
                      for size in lengths]
            if any(limit <= 0 for limit in limits):
                raise ValueError("Prompt exceeds context")
            before = time.monotonic()
            write_json(out / "status.json", {"status": "running", "condition": name,
                       "completed_in_condition": len(records), "expected_in_condition": len(rows),
                       "batch_ids": [r["id"] for r in batch], "at": time.time(), "paid_api_calls": 0})
            if args.backend == "vllm":
                params = [SamplingParams(temperature=.6 if condition["thinking"] else .7,
                          top_p=.95 if condition["thinking"] else .8, top_k=20,
                          presence_penalty=condition["presence_penalty"], max_tokens=limit,
                          seed=SEED + row["group"]) for row, limit in zip(batch, limits)]
                outputs = model.generate(prompts, params, use_tqdm=False)
                generations = [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason)
                               for o in outputs]
            else:
                inputs = tokenizer(prompts[0], return_tensors="pt", add_special_tokens=False)
                torch.manual_seed(SEED + batch[0]["group"])
                processors = LogitsProcessorList([presence_processor(condition["presence_penalty"], lengths[0])])
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=limits[0], do_sample=True,
                             temperature=.6 if condition["thinking"] else .7,
                             top_p=.95 if condition["thinking"] else .8, top_k=20,
                             logits_processor=processors, pad_token_id=tokenizer.eos_token_id)
                ids = output[0, lengths[0]:].tolist()
                eos = model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos)
                generations = [(tokenizer.decode(ids, skip_special_tokens=True), len(ids),
                                "stop" if ids and ids[-1] in eos else "length")]
            for row, limit, prompt_length, (text, tokens, finish) in zip(batch, limits, lengths, generations):
                record = {**row, "condition": name, "text": text, "tokens": tokens, "finish": finish,
                          "truncated": finish == "length", "max_output_tokens": limit,
                          "prompt_tokens": prompt_length, "batch_seconds": time.monotonic() - before,
                          **grade_response(text, condition, row)}
                with path.open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                print(json.dumps({k: record[k] for k in ["id", "condition", "tokens", "truncated", "strict_correct"]}), flush=True)
        all_summaries[name] = summarize(records, condition, len(rows))
        write_json(out / f"{name}-summary.json", all_summaries[name])
    prepare(root)
    write_json(out / "summary.json", {"status": "completed", "conditions": all_summaries,
               "seconds_this_invocation": time.monotonic() - started, "paid_api_calls": 0,
               "qualification": "manual_review_required", "fixture_hashes_verified_after_inference": True})
    write_json(out / "status.json", {"status": "completed", "at": time.time(), "paid_api_calls": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--conditions", default=",".join(FULL_ORDER))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--wait-hours", type=float, default=24)
    parser.add_argument("--run-hours", type=float, default=12)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        if args.out.exists():
            write_json(args.out / "status.json", {"status": "failed", "type": type(exc).__name__,
                       "error": str(exc), "at": time.time()})
        raise
