"""Every logged forward request gets the verdict the authors' server gave it, and the published scores recompute."""

import concurrent.futures as cf
import json
import subprocess

import pytest

from wsplab import upstream
from wsplab.eval import merge_scores, paired
from wsplab.eval.graders import MathGrader
from wsplab.passes import ForwardLimits, LMForward, PassFailed, PassInvalid
from wsplab.runners.fake import FakeLM

QWEN = ForwardLimits(n_layers=24, full_attention=(3, 7, 11, 15, 19, 23), gen_max=512, max_tokens=32768,
                     tf_max_tokens=32768)
GEMMA = ForwardLimits(n_layers=18, full_attention=(5, 11, 17), gen_max=2048, max_tokens=8192, tf_max_tokens=8192)
RUNS = {"exp8_math/runs/p50c": QWEN, "exp8_math/gemma_followup/runs/g50b": GEMMA}


def tree():
    out = subprocess.run(["gh", "api", f"repos/{upstream.REPO}/git/trees/{upstream.COMMIT}?recursive=1", "--jq",
                          '.tree[]|select(.type=="blob")|.path'], capture_output=True, text=True)
    if out.returncode:
        pytest.skip("gh is not available")
    return out.stdout.split()


class AnyId(dict):
    """Practice ids that follow the release's pattern exist; others do not."""

    def __contains__(self, key):
        import re
        return bool(re.fullmatch(r"(gsm8k_train_\d+|math_train_\w+_\d+|aime_[\w-]+)", key))

    def __getitem__(self, key):
        return {"id": key, "source": key.split("_")[0], "problem": key, "answer": "1"}


@pytest.mark.network
@pytest.mark.parametrize("run", RUNS)
def test_logged_requests_get_the_same_verdict(run, tmp_path):
    paths = [p for p in tree() if p.startswith(f"{run}/requests/")]
    with cf.ThreadPoolExecutor(16) as pool:
        list(pool.map(upstream.fetch, paths))
    base = upstream.cache_dir() / run / "requests"
    tool = LMForward(FakeLM(), practice=AnyId(), grader=MathGrader(), limits=RUNS[run],
                     sampling={"temperature": 0.6, "max_tokens": 8}, work=tmp_path, base=tmp_path / "b",
                     edited=lambda: tmp_path / "e", edit_digest=lambda: None, user_message=lambda r: r["problem"])
    accepted = rejected = 0
    for done in sorted(base.glob("fwd_*.done.json")):
        k = int(done.name.split("_")[1].split(".")[0])
        body = (base / f"fwd_{k}.json.claimed").read_text()
        logged = json.loads(done.read_text())
        tool._seen.clear()
        error = str(logged.get("error", ""))
        if logged.get("ok") or error.startswith(("RuntimeError", "IndexError")):
            # RuntimeError: their vLLM crashing at start-up. IndexError: a bug in their attention read, fixed later.
            tool.run(body, k)
            accepted += 1
        else:
            with pytest.raises((PassFailed, PassInvalid), match="unknown practice ids"):
                tool.run(body, k)
            rejected += 1
    assert accepted >= 50


def fetch_eval(name):
    paths = [p for p in tree() if p.startswith(f"exp8_math/eval/{name}/")]
    with cf.ThreadPoolExecutor(16) as pool:
        list(pool.map(upstream.fetch, [p for p in paths if p.endswith((".json",))]))
    return upstream.cache_dir() / "exp8_math" / "eval" / name


@pytest.mark.network
@pytest.mark.parametrize("name", ["unedited", "p10c", "p25c", "p50c"])
def test_published_scores_recompute_from_their_parts(name):
    folder = fetch_eval(name)
    want = json.loads((folder / "merged.json").read_text())
    got = merge_scores(folder)
    assert set(got) == set(want)
    for benchmark, row in want.items():
        for key in ("acc", "cut_share", "mean_tokens", "problems"):
            assert got[benchmark][key] == pytest.approx(row[key], rel=1e-12), (benchmark, key)


@pytest.mark.network
def test_paired_math500_gain_of_the_published_edit():
    def per_problem(name):
        rows = {}
        for f in sorted(fetch_eval(name).glob("parts/math500*/math500_per_problem.json")):
            rows.update({r["id"]: r["correct"] for r in json.loads(f.read_text())})
        return rows

    base, edit = per_problem("unedited"), per_problem("p50c")
    ids = sorted(base)
    assert len(ids) == 500 and set(ids) == set(edit)
    p = paired([[float(x) for x in base[i]] for i in ids], [[float(x) for x in edit[i]] for i in ids])
    assert p.original == pytest.approx(0.079, abs=5e-4) and p.edited == pytest.approx(0.481, abs=5e-4)
    assert p.difference / p.standard_error > 10
