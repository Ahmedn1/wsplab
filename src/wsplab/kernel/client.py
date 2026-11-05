"""Start and talk to the kernel. Sandboxing follows harness/sandbox.py (MIT): bubblewrap, no network, no GPU, a clean
environment and /work kept between runs. New here: the kernel is persistent, timeouts kill the whole PID namespace, and
memory is capped with a cgroup, because RLIMIT_AS would also count the memory-mapped model.
"""

from __future__ import annotations

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..io.safetensors import Checkpoint, is_model_file
from ..store import WeightStore
from ..upstream import home

PACKAGE = Path(__file__).resolve().parent.parent
RESTARTED = "the kernel was restarted, so its variables are gone"


@dataclass
class Limits:
    seconds: int = 900
    memory_gb: float | None = None
    output_chars: int = 10_000
    threads: int = 8


@dataclass
class KernelConfig:
    model: Path
    work: Path
    practice: Path | None = None
    mode: str = "edit"
    frozen: tuple = ()
    refuse_rescale: bool = True
    names: list[str] | None = None
    prelude: str = ""
    limits: Limits = field(default_factory=Limits)
    sandbox: str = "bwrap"
    python: str | None = None
    binds: dict = field(default_factory=dict)


class KernelError(RuntimeError):
    pass


def sandbox_python() -> str:
    python = home() / "sandbox" / "bin" / "python"
    if not python.exists():
        raise KernelError(f"no sandbox environment at {python.parent.parent}: run `wsplab setup`")
    return str(python)


def clean_output(text: str, limit: int) -> str:
    """As the prompts promise: ASCII only, cut at the limit, never empty."""
    text = text.encode("ascii", "replace").decode()
    if len(text) > limit:
        text = text[:limit] + f"\n[output cut at {limit:,} characters]"
    return text or "(no output)"


def _memory_cgroup_works() -> bool:
    if shutil.which("systemd-run") is None:
        return False
    probe = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=1G", "true"]
    return subprocess.run(probe, capture_output=True).returncode == 0


def _model_binds(model: Path) -> list[str]:
    """Bind each model file on its own: a Hugging Face snapshot is made of symlinks that would break inside."""
    if model.is_file():
        return ["--ro-bind", str(model.resolve()), f"/model/{model.name}"]
    binds = []
    for f in sorted(model.iterdir()):
        if f.is_file() and is_model_file(f.name):
            binds += ["--ro-bind", str(f.resolve()), f"/model/{f.name}"]
    return binds


class SandboxKernel:
    """Persistent Python over a model's weights. `sandbox="none"` runs it as a plain process: no isolation at all."""

    def __init__(self, config: KernelConfig):
        self.config = config
        self.config.work = Path(config.work)
        self.config.work.mkdir(parents=True, exist_ok=True)
        self.store = WeightStore(Checkpoint(config.model), self.config.work / "edited", mode=config.mode,
                                 frozen=config.frozen, refuse_rescale=config.refuse_rescale, names=config.names)
        self.process: subprocess.Popen | None = None
        self.restarts = 0
        self._cgroup = None

    # Lifecycle -------------------------------------------------------------------------------------------------------
    def _server_config(self) -> dict:
        c = self.config
        isolated = c.sandbox == "bwrap"
        return {"model": "/model" if isolated else str(c.model), "work": "/work" if isolated else str(c.work),
                "scratch": "/tmp", "mode": c.mode, "frozen": list(c.frozen), "refuse_rescale": c.refuse_rescale,
                "names": c.names, "prelude": c.prelude}

    def _command(self) -> list[str]:
        c = self.config
        python = c.python or (sandbox_python() if c.sandbox == "bwrap" else sys.executable)
        code = ("import sys, json; sys.path.insert(0, %r); from wsplab.kernel.server import main; "
                "main(json.loads(sys.argv[1]))")
        config = json.dumps(self._server_config())
        if c.sandbox == "none":
            return [python, "-I", "-c", code % str(PACKAGE.parent), config]
        if c.sandbox != "bwrap":
            raise KernelError(f"unknown sandbox {c.sandbox!r}")
        if shutil.which("bwrap") is None:
            raise KernelError("bubblewrap is not installed: install `bubblewrap`, or pass sandbox='none' to run without "
                              "isolation")
        venv = Path(python).parent.parent
        real = Path(os.path.realpath(python)).parent.parent
        cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
               "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--ro-bind", str(venv), "/opt/venv", "--ro-bind", str(PACKAGE), "/opt/wsplab/wsplab",
               "--tmpfs", "/model", *_model_binds(Path(c.model)), "--remount-ro", "/model",
               "--bind", str(c.work), "/work", "--chdir", "/work",
               "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin",
               "--setenv", "OMP_NUM_THREADS", str(c.limits.threads), "--setenv", "CUDA_VISIBLE_DEVICES", ""]
        if not str(real).startswith("/usr"):
            cmd += ["--ro-bind", str(real), str(real)]
        if c.practice is not None:
            cmd += ["--ro-bind", str(Path(c.practice).resolve()), "/practice"]
        for host, inside in c.binds.items():
            cmd += ["--ro-bind", str(Path(host).resolve()), str(inside)]
        cmd += ["/opt/venv/bin/python", "-I", "-c", code % "/opt/wsplab", config]
        if c.limits.memory_gb and _memory_cgroup_works():
            cap = int(c.limits.memory_gb * 2**30)
            cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0",
                   *cmd]
        return cmd

    def start(self) -> None:
        self.process = subprocess.Popen(self._command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, start_new_session=True, bufsize=1)
        line = self._read_line(120)
        if line is None or not json.loads(line).get("ready"):
            err = self._kill()
            raise KernelError(f"the kernel did not start: {err[-2000:]}")

    def _read_line(self, seconds: float) -> str | None:
        """One line from the kernel, or None if it died or the time ran out."""
        p = self.process
        with selectors.DefaultSelector() as sel:
            sel.register(p.stdout, selectors.EVENT_READ)
            if not sel.select(timeout=seconds):
                return None
        return p.stdout.readline() or None

    def _kill(self) -> str:
        """Kill the kernel and everything it started, and return what it wrote to stderr."""
        p = self.process
        if p is None:
            return ""
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            err = p.stderr.read() or ""
        except ValueError:
            err = ""
        p.wait()
        self.process = None
        return err

    def restart(self) -> None:
        self._kill()
        self.restarts += 1
        self.start()

    def close(self) -> None:
        self._kill()

    def __enter__(self) -> SandboxKernel:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # Running code ----------------------------------------------------------------------------------------------------
    def run(self, code: str, *, allow_edits: bool = True) -> str:
        if self.process is None:
            self.start()
        p, limits = self.process, self.config.limits
        started = time.time()
        p.stdin.write(json.dumps({"code": code, "allow_edits": allow_edits}) + "\n")
        p.stdin.flush()
        line = self._read_line(limits.seconds)
        if line is None:
            timed_out = time.time() - started >= limits.seconds
            self._kill()
            self.restarts += 1
            self.start()
            reason = f"timed out after {limits.seconds} s" if timed_out else "the run was killed, most likely out of memory"
            return f"[{reason}: {RESTARTED}]"
        return clean_output(json.loads(line)["output"], limits.output_chars)

    # What the session needs to know ------------------------------------------------------------------------------------
    def edited(self) -> list[str]:
        return self.store.edited()

    def files(self) -> list[str]:
        return sorted(f.name for f in self.config.work.iterdir() if not f.name.startswith("."))

    def notes(self) -> str:
        p = self.config.work / "notes.md"
        return p.read_text(errors="replace") if p.exists() else ""

    def edits_used(self) -> tuple[list[str], str | None]:
        return self.store.edited(), self.store.digest()
