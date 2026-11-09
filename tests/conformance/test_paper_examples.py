"""The paper's experiments as configuration: released answers score as logged, and released replies rebuild the released
networks through wsplab's own edit reader."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from wsplab.passes.edits import apply_edits, last_block
from wsplab.upstream import fetch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "paper"))

pytestmark = pytest.mark.network


def test_core_has_no_experiment_specific_code():
    """Game policies, diffusion models and Go networks are configuration, not types."""
    banned = re.compile(r"\b(minatar|kinetix|breakout|seaquest|asterix|pacman|freeway|diffusion|sycophan|exp[1-7]_)", re.I)
    for path in (ROOT / "src" / "wsplab").rglob("*"):
        if path.suffix in (".py", ".txt"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if banned.search(line) and "exp7_sycophancy" not in line:
                    pytest.fail(f"{path}:{number}: {line.strip()[:100]}")


@pytest.mark.parametrize("game", ["breakout", "pong", "pacman", "freeway"])
def test_exp1_released_answers_score_as_logged(game):
    import exp1_game

    answer = fetch(f"exp1_which_game/runs/{game}/answer.md").read_text()
    assert exp1_game.score(answer) == game


@pytest.mark.parametrize("shape", ["circle", "square", "triangle"])
def test_exp2_released_answers_score_as_logged(shape):
    import exp2_diffusion

    assert exp2_diffusion.score(fetch(f"exp2_diffusion/runs/{shape}/answer.md").read_text()) == shape


@pytest.mark.parametrize("language", ["python", "romaji", "english"])
def test_exp3_released_answers_score_as_logged(language):
    import exp3_language

    rows = {r["id"]: r for r in json.loads(fetch("exp3_language/eval/scores.json").read_text())}
    row = next(r for k, r in rows.items() if k.startswith(language) and k.endswith("_r0"))
    answer = fetch(f"exp3_language/runs/{language}_r0/final_answer.md").read_text()
    assert exp3_language.score(answer) == row["corpus_guess"]


def test_exp4_released_reply_rebuilds_the_released_network():
    import exp4_car

    prompt = fetch(exp4_car.PROMPT).read_text()
    shown = exp4_car.shown_network(prompt)
    released = np.load(fetch("exp4_fix_car/networks/edited_actor.npz"))
    assert set(shown) == set(released.files)
    reply = fetch("exp4_fix_car/runs/oneshot_repair/reply/answer.md").read_text()
    applied = apply_edits(last_block(reply), shown.__getitem__)
    assert applied.errors == [] and applied.statements == 51
    for name in released.files:
        ours = applied.tensors.get(name, shown[name])
        assert np.allclose(ours, released[name], atol=1e-5), name


@pytest.mark.parametrize("game", ["breakout", "seaquest", "asterix"])
def test_exp5_released_reply_rebuilds_the_released_network(game):
    import exp5_minatar

    reply = fetch(f"exp5_write_from_scratch/runs/{game}/reply/answer.md").read_text()
    applied = apply_edits(last_block(reply), lambda n: np.zeros(exp5_minatar.SHAPES[n]))
    logged = json.loads(fetch(f"exp5_write_from_scratch/runs/{game}/apply.json").read_text())["edit_report"]
    assert applied.errors == [] and applied.statements == logged["statements"]
    released = np.load(fetch(f"exp5_write_from_scratch/networks/{game}/hand.npz"))
    for short, name in exp5_minatar.NAMES.items():
        ours = applied.tensors.get(short, np.zeros(exp5_minatar.SHAPES[short]))
        assert np.allclose(ours, released[name], atol=1e-6), short


def test_examples_use_the_authors_exact_prompts():
    import exp1_game
    import exp3_language

    assert exp1_game.config("pong")["prompt"].fidelity == "exact"
    text = exp3_language.config("python")["prompt"].build()
    assert text == fetch("exp3_language/runs/python_r0/message.txt").read_text()


def sandbox_or_skip():
    from wsplab.kernel.client import KernelError, sandbox_python

    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment")


def test_exp7_attention_tool_layout(tmp_path):
    import exp7_sycophancy

    from wsplab.passes import PassInvalid
    from wsplab.runners.fake import FakeLM

    tool = exp7_sycophancy.AttentionTool(FakeLM(), tmp_path, {"A": tmp_path / "a", "B": tmp_path / "b"}, layers=2)
    result = tool.run(json.dumps(["one two", "three"]), 1)
    assert result.where == "/work/attn/r1/"
    a = np.load(tmp_path / "attn" / "r1" / "t0_A.npy")
    assert a.dtype == np.float16 and a.shape[0] == 2 and a.ndim == 4
    assert (tmp_path / "attn" / "r1" / "t1_B.npy").exists()
    assert len(json.loads((tmp_path / "attn" / "r1" / "tokens.json").read_text())) == 2
    with pytest.raises(PassInvalid):
        tool.run(json.dumps(["x"] * 17), 2)


def test_exp7_prelude_runs_in_the_kernel_on_two_mounted_checkpoints(tmp_path):
    sandbox_or_skip()
    import exp7_sycophancy

    import wsplab
    from wsplab.editors import ScriptedEditor
    from wsplab.io import bf16_bits, save
    from wsplab.kernel import Limits
    from wsplab.runners.fake import FakeLM

    rng = np.random.default_rng(0)
    for side in "AB":
        folder = tmp_path / side
        folder.mkdir()
        save(folder / "model.safetensors", {"w": ("BF16", bf16_bits(rng.standard_normal((2, 2)).astype(np.float32)))})
        (folder / "config.json").write_text("{}")
    cfg = exp7_sycophancy.config(str(tmp_path / "A"), str(tmp_path / "B"), runner=FakeLM(), layers=2)
    code = "a, b = Checkpoint('A'), Checkpoint('B')\nprint(a.keys(), float(abs(a.get('w') - b.get('w')).sum()) > 0)"
    replies = [f"Comparing them.\n```python\n{code}\n```", "Now the tool.\n```attention\n[\"hello\"]\n```",
               *[f"Filling the budget {i}.\n```attention\n[\"t{i}\"]\n```" for i in range(19)], "A is the sycophant."]
    r = wsplab.run(tmp_path / "A", editor=ScriptedEditor(replies), folder=tmp_path / "run", limits=Limits(seconds=60, memory_gb=4), **cfg)
    assert r.answer == "A is the sycophant."
    assert "['w'] True" in (r.folder / "session" / "tool_calls.jsonl").read_text()
    assert (r.folder / "work" / "attn" / "r1" / "t0_B.npy").exists()


def test_exp6_network_matches_the_prompt():
    pytest.importorskip("torch")
    import exp6_go

    net = exp6_go.GoNet()
    names = set(net.state_dict())
    assert {"L1.W", "L6.b", "out.W", "out.b"} <= names
    assert sum(p.numel() for p in net.parameters()) == 1_480_274


def test_exp6_plays_real_go_through_the_kernel_and_its_own_tools(tmp_path):
    pytest.importorskip("pgx")
    sandbox_or_skip()
    import exp6_go

    import wsplab
    from wsplab.editors import ScriptedEditor
    from wsplab.kernel import Limits

    cfg = exp6_go.config(10)
    design = "b = np.zeros(82, np.float32); b[mv(4, 4)] = 3.0; print(save('out.b', b), SHAPES['L1.W'], inp(0, 0, 1), PASS)"
    games = '{"games": {"random": 2}}'
    replies = [f"Prefer the centre.\n```python\n{design}\n```", f"Trying it.\n```games\n{games}\n```",
               "Now a position.\n```positions\n{\"positions\": [{\"moves\": [40, \"pass\"]}]}\n```"]
    replies += [f"Again.\n```games\n{{\"games\": {{\"random\": {2 * (i + 1)}}}}}\n```" for i in range(1)]
    replies += [f"Again {i}.\n```positions\n{{\"positions\": [{{\"moves\": [{i}]}}]}}\n```" for i in range(7)]
    replies += ["The network plays the centre first."]
    r = wsplab.run(exp6_go.GoNet(), editor=ScriptedEditor(replies), folder=tmp_path / "run",
                   limits=Limits(seconds=120, memory_gb=8), **cfg)
    assert r.state.passes == 10 and r.answer.startswith("The network")
    assert (r.folder / "work" / "pass_1" / "summary.json").exists()
    assert (r.folder / "work" / "pass_2" / "logits.npy").exists()
    assert "(243, 512) 1 81" in (r.folder / "session" / "tool_calls.jsonl").read_text()
    from wsplab import patch as P

    assert [t["name"] for t in P.Patch.load(r.patch).tensors] == ["out.b"]


def test_exp6_says_plainly_that_the_authors_opponents_are_missing(tmp_path):
    pytest.importorskip("pgx")
    import exp6_go

    from wsplab.passes import PassFailed

    tool = exp6_go.GamesTool(None, tmp_path, lambda: None)
    with pytest.raises(PassFailed, match="were not released"):
        tool.run('{"games": {"strong": 2}}', 1)
