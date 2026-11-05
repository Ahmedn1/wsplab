"""The kernel inside a real bubblewrap sandbox."""

import shutil

import numpy as np
import pytest

from wsplab.io import bf16_bits, save
from wsplab.kernel import KernelConfig, KernelError, Limits, SandboxKernel, clean_output
from wsplab.kernel.client import sandbox_python

needs_sandbox = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is not installed")


@pytest.fixture
def model(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    rng = np.random.default_rng(2)
    save(folder / "model.safetensors", {"w": ("BF16", bf16_bits(rng.standard_normal((8, 4)).astype(np.float32))),
                                        "norm.weight": ("BF16", bf16_bits(np.ones(4, np.float32)))})
    (folder / "config.json").write_text("{}")
    (folder / "secret.bin").write_text("not a model file")
    (folder / "heldout.json").write_text("evaluation data left beside the model")
    return folder


def kernel(model, tmp_path, **kw):
    limits = kw.pop("limits", Limits())
    try:
        sandbox_python()
    except KernelError:
        pytest.skip("no sandbox environment: run `wsplab setup`")
    return SandboxKernel(KernelConfig(model=model, work=tmp_path / "work", limits=limits, **kw))


@pytest.fixture
def k(model, tmp_path):
    with kernel(model, tmp_path, frozen=("norm.weight",)) as kern:
        yield kern


def test_clean_output():
    assert clean_output("", 10) == "(no output)"
    assert clean_output("caf\u00e9", 10) == "caf?"
    assert clean_output("x" * 12, 10) == "x" * 10 + "\n[output cut at 10 characters]"


@needs_sandbox
def test_variables_persist_and_last_expression_is_shown(k):
    assert k.run("x = 20") == "(no output)"
    assert k.run("print('hi')\nx + 22") == "hi\n42\n"


@needs_sandbox
def test_paper_surface_is_defined(k):
    out = k.run("sorted(NAMES), type(load('w')).__name__, load('w').dtype, torch.is_grad_enabled()")
    assert "['norm.weight', 'w']" in out and "Tensor" in out and "torch.float32" in out and "False" in out


@needs_sandbox
def test_save_edit_round_trip_and_host_sees_it(k):
    out = k.run("w = load('w'); w[:, 0] = 0; save_edit('w', w)")
    assert "8 of 32 entries differ" in out
    assert k.edited() == ["w"] and k.edits_used()[1] is not None
    assert "0." in k.run("float(load('w')[0, 0].abs())")
    assert k.run("drop_edit('w')") == "(no output)"
    assert k.edited() == []


@needs_sandbox
def test_refusals_reach_the_editor(k):
    assert "out of scope" in k.run("save_edit('norm.weight', load('norm.weight') + 1)")
    assert "constant" in k.run("save_edit('w', load('w') * 2)")


@needs_sandbox
def test_traceback_shows_only_the_editors_code(k):
    out = k.run("x = 1\nundefined_name")
    assert 'File "<run>", line 2' in out and "NameError" in out and "server.py" not in out


@needs_sandbox
def test_syntax_errors_are_reported(k):
    assert "SyntaxError" in k.run("def (")


@needs_sandbox
def test_subprocess_output_is_captured(k):
    assert "from child" in k.run("import subprocess; subprocess.run(['/usr/bin/echo', 'from child'])")


@needs_sandbox
def test_output_is_cut_and_ascii(k):
    out = k.run("print('\\u00e9' * 20000)")
    assert out.startswith("?") and out.endswith("[output cut at 10,000 characters]")


@needs_sandbox
def test_no_network(k):
    out = k.run("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\n    print('OPEN')\n"
                "except OSError as e:\n    print('blocked')")
    assert out.strip() == "blocked"


@needs_sandbox
def test_no_gpu_and_no_model_runtime(k):
    assert k.run("torch.cuda.is_available()").strip() == "False"
    assert "ModuleNotFoundError" in k.run("import transformers")
    assert "ModuleNotFoundError" in k.run("import vllm")


@needs_sandbox
def test_host_files_and_other_model_files_are_invisible(k, tmp_path):
    out = k.run(f"import os\nprint(os.path.exists({str(tmp_path)!r}), os.path.exists('/home'), "
                "os.path.exists('/model/secret.bin'), os.path.exists('/model/model.safetensors'))")
    assert out.strip() == "False False False True"


@needs_sandbox
def test_only_model_files_are_mounted(k):
    assert k.run("import os\nsorted(os.listdir('/model'))").strip() == "['config.json', 'model.safetensors']"


def test_model_folder_is_read_only(k):
    assert "Read-only" in k.run("open('/model/x', 'w')")
    assert "Read-only" in k.run("open('/model/model.safetensors', 'r+b')")


@needs_sandbox
def test_work_survives_a_timeout_but_variables_do_not(model, tmp_path):
    with kernel(model, tmp_path, limits=Limits(seconds=2)) as k:
        k.run("open('/work/notes.md', 'w').write('kept')\nw = load('w'); w[:, 0] = 0; save_edit('w', w)\nx = 1")
        out = k.run("import time; time.sleep(30)")
        assert out.startswith("[timed out after 2 s") and "restarted" in out
        assert k.restarts == 1
        assert "NameError" in k.run("x")
        assert k.notes() == "kept" and k.edited() == ["w"]
        assert "kept" in k.run("open('/work/notes.md').read()")


@needs_sandbox
def test_timeout_kills_child_processes_too(model, tmp_path):
    with kernel(model, tmp_path, limits=Limits(seconds=2)) as k:
        k.run("import subprocess\nsubprocess.Popen(['/usr/bin/sleep', '987654'])\nimport time; time.sleep(30)")
        import subprocess
        assert subprocess.run(["pgrep", "-f", "sleep 987654"], capture_output=True).stdout == b""


@needs_sandbox
def test_memory_limit_restarts_the_kernel(model, tmp_path):
    with kernel(model, tmp_path, limits=Limits(memory_gb=1)) as k:
        out = k.run("import numpy as np\na = np.ones(1 << 31, dtype=np.uint8)\nprint('survived', a.sum())")
        assert "survived" not in out and "restarted" in out and k.restarts == 1
        assert k.run("1 + 1").strip() == "2"


@needs_sandbox
def test_memory_mapped_model_does_not_count_against_the_limit(model, tmp_path):
    with kernel(model, tmp_path, limits=Limits(memory_gb=1)) as k:
        assert "torch.Size([8, 4])" in k.run("load('w').shape")


@needs_sandbox
def test_read_mode_has_no_save_edit(model, tmp_path):
    with kernel(model, tmp_path, mode="read") as k:
        assert "NameError" in k.run("save_edit('w', load('w'))")


@needs_sandbox
def test_locked_run_cannot_edit(k):
    assert "not approved" in k.run("save_edit('w', load('w') + 1)", allow_edits=False)
    assert k.edited() == []
    assert "entries differ" in k.run("w = load('w'); w[:, 0] = 0; save_edit('w', w)", allow_edits=True)


@needs_sandbox
def test_practice_is_mounted_read_only(model, tmp_path):
    practice = tmp_path / "practice"
    practice.mkdir()
    (practice / "items.jsonl").write_text('{"id": 1}\n')
    with kernel(model, tmp_path, practice=practice) as k:
        assert "1" in k.run("open('/practice/items.jsonl').read()")
        assert "Read-only" in k.run("open('/practice/x', 'w')")


def test_unsafe_mode_runs_without_a_sandbox(model, tmp_path):
    with SandboxKernel(KernelConfig(model=model, work=tmp_path / "w", sandbox="none")) as k:
        assert k.run("2 + 2").strip() == "4"
