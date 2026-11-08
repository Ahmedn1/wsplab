"""Unpaid local screening of a frozen multilingual-math candidate; no weight edits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import unicodedata
from pathlib import Path

MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_REVISION = "5f091edeae66ebc605ac4f98dca07a6c345f0060"
LANGUAGES = ("en", "bn", "sw")
SEED = 20261106
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize_digits(text):
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c
                   for c in unicodedata.normalize("NFKC", text))


def prepare(root):
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        for filename, expected in plan["files"].items():
            if digest(root / filename) != expected:
                raise ValueError(f"Frozen file changed: {filename}")
        return plan
    data = {}
    for language in LANGUAGES:
        with (root / "data" / f"mgsm_{language}.tsv").open() as stream:
            data[language] = list(csv.reader(stream, delimiter="\t"))
        assert len(data[language]) == 250
        assert all(len(row) == 2 for row in data[language])
    for i in range(250):
        assert len({normalize_digits(data[l][i][1]).replace(",", "").strip()
                    for l in LANGUAGES}) == 1, f"Unaligned gold answers at {i + 1}"
    indices = list(range(250))
    random.Random(SEED).shuffle(indices)
    splits = {"pilot": indices[:32], "practice": indices[32:96],
              "selection": indices[96:136], "evaluation": indices[136:]}
    pilot = [{"id": f"mgsm-rev2-{i + 1}-{language}", "group": i + 1,
              "language": language, "problem": data[language][i][0],
              "answer": normalize_digits(data[language][i][1]).replace(",", "").strip()}
             for i in splits["pilot"] for language in LANGUAGES]
    (root / "pilot.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pilot))
    files = [root / "pilot.jsonl", *sorted((root / "data").glob("*.tsv")),
             *sorted((root / "model").glob("*.json")), root / "model/model.safetensors"]
    plan = {"status": "candidate_not_yet_qualified", "model": "Qwen/Qwen3-0.6B",
            "model_revision": MODEL_REVISION, "dataset": "google-research-datasets/MGSM-Rev2",
            "data_revision": DATA_REVISION, "seed": SEED, "languages": LANGUAGES,
            "splits_zero_based": splits, "files": {str(p.relative_to(root)): digest(p) for p in files},
            "instruction": INSTRUCTION, "paid_api_calls": 0,
            "screening_note": "CPU short-output non-thinking smoke is feasibility only. "
            "Do not infer weak capability from truncated outputs. Confirm with recommended-length "
            "thinking and non-thinking controls before a paid comparison.",
            "qualification_rule": "On all 32 pilot groups with adequate output length: English >=50%; "
            "mean Bengali/Swahili <=40%; gap >=20 percentage points; truncation <=10%. "
            "A pilot gap is task-selection evidence, not evidence that editing can repair it.",
            "created_at": time.time()}
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def run(root, groups, max_tokens, device, thinking, threads):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wsplab.eval import MathGrader

    plan = prepare(root)
    torch.set_num_threads(threads)
    if not 1 <= groups <= 32:
        raise ValueError("Only the frozen pilot may be screened")
    tag = f"{'thinking' if thinking else 'nonthinking'}-{device}-{groups}x3-{max_tokens}"
    out = root / tag
    out.mkdir(exist_ok=True)
    metadata = {"script_sha256": digest(__file__), "plan_sha256": digest(root / "plan.json"),
                "groups": groups, "max_tokens": max_tokens, "device": device,
                "thinking": thinking, "threads": threads, "torch": torch.__version__}
    metadata_path = out / "execution.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("Execution settings changed; preserve the previous run")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    rows = [json.loads(line) for line in (root / "pilot.jsonl").read_text().splitlines()][:groups * 3]
    records = out / "responses.jsonl"
    previous = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    completed = {r["id"] for r in previous}
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        root / "model", local_files_only=True, trust_remote_code=False,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16).to(device).eval()
    grader = MathGrader(lenient=False)
    lenient = MathGrader(lenient=True)
    started = time.monotonic()
    for row in rows:
        if row["id"] in completed:
            continue
        torch.manual_seed(SEED + row["group"])
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"] + "\n" + INSTRUCTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        before = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True,
                                    temperature=.6 if thinking else .7,
                                    top_p=.95 if thinking else .8, top_k=20,
                                    pad_token_id=tokenizer.eos_token_id)
        ids = output[0, inputs.input_ids.shape[1]:].tolist()
        response = tokenizer.decode(ids, skip_special_tokens=True)
        final = response.rsplit("</think>", 1)[-1] if "</think>" in response else response
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        truncated = not ids or ids[-1] not in eos
        no_final = thinking and "</think>" not in response
        normalized = normalize_digits(final)
        strict = grader.grade(normalized, row)
        loose = lenient.grade(normalized, row)
        record = {**row, "text": response, "tokens": len(ids), "truncated": truncated,
                  "thinking_unfinished": no_final, "strict_correct": bool(strict.correct and not no_final),
                  "lenient_correct": bool(loose.correct and not no_final),
                  "seconds": time.monotonic() - before}
        previous.append(record)
        with records.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({k: record[k] for k in ["id", "tokens", "truncated", "strict_correct", "seconds"]}), flush=True)
    summary = {"status": "completed_feasibility_screen", "qualification": "not_assessed",
               "model": plan["model"], "groups": groups, "max_tokens": max_tokens,
               "thinking": thinking, "device": device, "seconds_this_invocation": time.monotonic() - started,
               "paid_api_calls": 0, "note": plan["screening_note"], "languages": {}}
    for language in LANGUAGES:
        subset = [r for r in previous if r["language"] == language]
        summary["languages"][language] = {key: sum(r[key] for r in subset) / len(subset)
                                          for key in ["strict_correct", "lenient_correct", "truncated", "tokens"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.prepare_only:
        print(json.dumps(prepare(args.folder), indent=2))
    else:
        run(args.folder, args.groups, args.max_tokens, args.device, args.thinking, args.threads)
