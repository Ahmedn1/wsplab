"""Set up the sandbox environment and check what this machine can run."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .upstream import home

SANDBOX_PACKAGES = ["numpy", "scipy", "scikit-learn", "pandas", "sympy", "networkx", "safetensors", "tokenizers"]
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def sandbox_dir() -> Path:
    return home() / "sandbox"


def setup_commands(target: Path, python: str = sys.executable) -> list[list[str]]:
    """The dedicated environment the kernel runs in. It has no transformers or vLLM and CPU-only PyTorch, so the editor
    cannot simply run the model there. It uses uv when it can, else venv and pip."""
    bin_python = str(target / "bin" / "python")
    if shutil.which("uv"):
        install = ["uv", "pip", "install", "--python", bin_python]
        return [["uv", "venv", "--python", python, str(target)], [*install, "torch", "--index-url", TORCH_CPU_INDEX],
                [*install, *SANDBOX_PACKAGES]]
    install = [bin_python, "-m", "pip", "install"]
    return [[python, "-m", "venv", str(target)], [*install, "torch", "--index-url", TORCH_CPU_INDEX],
            [*install, *SANDBOX_PACKAGES]]


def setup(force: bool = False) -> Path:
    target = sandbox_dir()
    if force and target.exists():
        shutil.rmtree(target)
    if (target / "bin" / "python").exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    for command in setup_commands(target):
        subprocess.run(command, check=True)
    return target


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    needed_for: str


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _gpu() -> tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return False, "no NVIDIA driver found"
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                         text=True)
    return out.returncode == 0, out.stdout.strip() or out.stderr.strip()


def _bwrap_works() -> tuple[bool, str]:
    if shutil.which("bwrap") is None:
        return False, "install bubblewrap (apt install bubblewrap)"
    probe = ["bwrap", "--unshare-all", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib64", "/lib64",
             "--symlink", "usr/lib", "/lib", "--proc", "/proc", "--dev", "/dev", "/usr/bin/true"]
    done = subprocess.run(probe, capture_output=True, text=True)
    return done.returncode == 0, "working" if done.returncode == 0 else done.stderr.strip()[:200]


def checks() -> list[Check]:
    bwrap_ok, bwrap_detail = _bwrap_works()
    gpu_ok, gpu_detail = _gpu()
    python = sandbox_dir() / "bin" / "python"
    return [Check("python", sys.version_info >= (3, 12), sys.version.split()[0], "everything"),
            Check("bubblewrap", bwrap_ok, bwrap_detail, "editing sessions"),
            Check("sandbox environment", python.exists(), str(python) if python.exists() else "run `wsplab setup`",
                  "editing sessions"),
            Check("torch", _has("torch"), "installed" if _has("torch") else "pip install 'wsplab[torch]'",
                  "running networks"),
            Check("transformers", _has("transformers"), "installed" if _has("transformers") else "pip install 'wsplab[lm]'",
                  "language models"),
            Check("vllm", _has("vllm"), "installed" if _has("vllm") else "pip install 'wsplab[vllm]'",
                  "fast language model generation"),
            Check("math-verify", _has("math_verify"), "installed" if _has("math_verify") else "pip install 'wsplab[eval]'",
                  "math grading"),
            Check("GPU", gpu_ok, gpu_detail, "language models at useful speed")]
