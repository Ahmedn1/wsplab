"""Live-test helpers: one ledger across every run. Needs OPENAI_API_KEY and LIVE_DIR, the folder for the ledger.

TOTAL stops new calls; one call can still pass it by what it costs."""
import fcntl, json, os, threading
from pathlib import Path
from wsplab.editors import Metered, Spending, Usage
from wsplab.editors.llm import OpenAIEditor

TOTAL = 24.0     # stop a dollar short of the user's $25, to absorb the one call that can overshoot
LEDGER = Path(os.environ["LIVE_DIR"]) / "ledger.json"
PRICES = {"gpt-6-astra": (10, 50, 1), "gpt-6-sol": (2, 10, .2), "gpt-6-luna": (.1, .5, .01), "gpt-6.1-sol": (2, 10, .1),
          "gpt-6.1-terra": (1, 5, .05)}
_lock = threading.Lock()


def spent() -> float:
    return json.loads(LEDGER.read_text())["usd"] if LEDGER.exists() else 0.0


class Ledgered(Spending):
    """A run's cap, also bounded by what is left of the whole budget, written to disk after every call."""
    def add(self, editor, usage):
        before = self.usd
        super().add(editor, usage)
        with _lock, open(str(LEDGER) + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"usd": 0.0, "calls": 0, "by_editor": {}}
            data["usd"] += self.usd - before
            data["calls"] += 1
            data["by_editor"][editor] = data["by_editor"].get(editor, 0.0) + self.usd - before
            LEDGER.write_text(json.dumps(data, indent=1))
        if spent() >= TOTAL:
            self.max_usd = 0.0   # the whole budget is gone: every run stops at its next call


def spending(cap: float) -> Ledgered:
    left = TOTAL - spent()
    if left <= 0.2:
        raise SystemExit(f"budget exhausted: ${spent():.2f} spent")
    return Ledgered(max_usd=min(cap, left), prices={f"openai:{k}": v for k, v in PRICES.items()})


def editor(model: str, effort: str | None = "high"):
    return OpenAIEditor(model, api_key=os.environ["OPENAI_API_KEY"], effort=effort)
