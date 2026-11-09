"""Progress messages for long runs: which step a run is on, how far generation has got, what the editor just did.

Messages go to the `wsplab` logger. While a run is going they are also written to `run.log` in its folder, and printed
unless `quiet=True` or you have set up logging yourself, in which case they reach your handlers instead. A run sets the
`wsplab` logger to INFO unless you have given it a level: `logging.getLogger("wsplab").setLevel(logging.WARNING)` keeps
progress out of your handlers (and out of run.log).

Nothing from the evaluation data goes into a message: only counts, scores and timings. Practice results and the editor's
own words may appear, since the editor sees those anyway.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import time
from pathlib import Path

logger = logging.getLogger("wsplab")

_label = contextvars.ContextVar("wsplab_label", default="")  # which session a message is about, in a council
_activity = contextvars.ContextVar("wsplab_activity", default="")  # what a generation is for: "baseline evaluation", ...
_folders = contextvars.ContextVar("wsplab_folders", default=())  # the run folders whose run.log takes this message

CONSOLE_FORMAT = "%(asctime)s  %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


class _Console(logging.StreamHandler):
    """wsplab's own console handler, so it is added once and removed by whoever added it."""


class _RunFile(logging.FileHandler):
    """A run's run.log."""


def _say(level: int, message: str) -> None:
    label = _label.get()
    logger.log(level, f"[{label}] {message}" if label else message)


def info(message: str) -> None:
    _say(logging.INFO, message)


def debug(message: str) -> None:
    _say(logging.DEBUG, message)


def warning(message: str) -> None:
    _say(logging.WARNING, message)


def error(message: str) -> None:
    _say(logging.ERROR, message)


@contextlib.contextmanager
def labelled(label: str):
    """Messages inside are prefixed with `label`: which council member they are about."""
    token = _label.set(label)
    try:
        yield
    finally:
        _label.reset(token)


@contextlib.contextmanager
def activity(name: str):
    """Generation progress inside is reported under `name`."""
    token = _activity.set(name)
    try:
        yield
    finally:
        _activity.reset(token)


def in_member() -> bool:
    """Whether messages are about one member of a search or council, whose own verdict matters less than the winner's."""
    return bool(_label.get())


def current_activity(default: str = "generating") -> str:
    return _activity.get() or default


def duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def plural(n: int, word: str, many: str | None = None) -> str:
    """plural(1, "tensor") -> "1 tensor", plural(3, "pass", "passes") -> "3 passes"."""
    return f"{n:,} {word if n == 1 else many or word + 's'}"


def tokens(n: float) -> str:
    """12,345 -> "12.3k", 65,536,000 -> "65.5M"."""
    for size, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return f"{n:.0f}"


def _configured_elsewhere() -> bool:
    """Whether someone set up logging: handlers of theirs on the wsplab logger or the root logger."""
    theirs = [h for h in logger.handlers if not isinstance(h, (_Console, _RunFile))]
    return bool(theirs) or bool(logging.getLogger().handlers)


def configure(level: int = logging.INFO) -> None:
    """Print wsplab's messages at `level` and above, for the command line or a notebook. Replaces an earlier call's."""
    for h in [h for h in logger.handlers if isinstance(h, _Console)]:
        logger.removeHandler(h)
    handler = _Console()
    handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, "%H:%M:%S"))
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(min(level, logging.INFO))  # run.log keeps progress messages even when the console shows fewer


@contextlib.contextmanager
def run_log(folder: Path, quiet: bool = False):
    """Log a run to folder/run.log, and to the console unless `quiet` or logging is set up already. A run nested in
    another (a council member's session) writes its own run.log, and its messages reach the outer run.log too."""
    folder = Path(folder)
    active = _folders.get()
    if folder in active:  # run() handing over to run_lm() for the same folder
        yield
        return
    configured = _configured_elsewhere()  # before adding run.log's handler, which is ours
    token = _folders.set((*active, folder))
    added: list[logging.Handler] = []
    old_level = logger.level
    try:
        file = _RunFile(folder / "run.log", encoding="utf-8")
        file.setFormatter(logging.Formatter(FILE_FORMAT, "%Y-%m-%d %H:%M:%S"))
        file.setLevel(logging.INFO)
        file.addFilter(lambda record, f=folder: f in _folders.get())
        logger.addHandler(file)
        added.append(file)
        if logger.level == logging.NOTSET:  # otherwise the root logger's WARNING would keep progress out of run.log
            logger.setLevel(logging.INFO)
        if not configured:
            if not quiet and not any(isinstance(h, _Console) for h in logger.handlers):
                console = _Console()
                console.setFormatter(logging.Formatter(CONSOLE_FORMAT, "%H:%M:%S"))
                console.setLevel(logging.INFO)
                logger.addHandler(console)
                added.append(console)
        yield
    finally:
        for h in added:
            logger.removeHandler(h)
            h.close()
        logger.setLevel(old_level)
        _folders.reset(token)


def run_context() -> contextvars.Context:
    """The current context, to hand to worker threads so their messages reach the same run.log files."""
    return contextvars.copy_context()


class Progress:
    """Reports a long generation now and then: done so far, speed, and a rough time left. Quiet for short ones, such as
    most forward passes, which only get a line at DEBUG."""

    def __init__(self, what: str, total: int, unit: str = "answers", every: float = 0.05, max_gap: float = 600,
                 min_gap: float = 60, clock=time.monotonic):
        self.what, self.total, self.unit = what, total, unit
        self.every, self.max_gap, self.min_gap = every, max_gap, min_gap
        self.clock = clock
        self.start = clock()
        self.last_time = self.start
        self.last_share = 0.0
        self.done = 0
        debug(f"{what}: {total:,} {unit} to generate")

    def update(self, done: int, out_tokens_per_s: float | None = None) -> None:
        now = self.clock()
        self.done = done
        if done >= self.total:  # finish() says so, with the time it took
            return
        share = done / self.total if self.total else 1.0
        since = now - self.last_time
        if since < self.min_gap or (share - self.last_share < self.every and since < self.max_gap):
            return
        self.last_time, self.last_share = now, share
        elapsed = now - self.start
        parts = [f"{self.what}: {done:,}/{self.total:,} {self.unit} ({share:.0%})"]
        if out_tokens_per_s:
            parts.append(f"{out_tokens_per_s:,.0f} tokens/s")
        if done:
            parts.append(f"about {duration(elapsed * (self.total - done) / done)} left")
        else:
            parts.append(f"none finished yet after {duration(elapsed)}")
        info(", ".join(parts))

    def finish(self) -> None:
        elapsed = self.clock() - self.start
        message = f"{self.what}: {self.total:,} {self.unit} in {duration(elapsed)}"
        info(message) if elapsed >= self.min_gap else debug(message)
