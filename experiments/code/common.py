"""Live-test helpers: a ledger per provider, written after every call, with a working limit below each real budget."""
import fcntl, json, os, threading
from pathlib import Path
from wsplab.editors import Spending
from wsplab.editors.claude import AnthropicEditor
from wsplab.editors.llm import ChatEditor, OpenAIEditor

LIVE = Path(os.environ["LIVE_DIR"])
LIMITS = {"openai": 17.0, "anthropic": 8.0, "gemini": 9.0}   # budgets: $19, $25, $25
PRICES = {"openai:gpt-6-astra": (10, 50, 1), "openai:gpt-6-sol": (2, 10, .2), "openai:gpt-6-luna": (.1, .5, .01),
          "openai:gpt-6.1-sol": (2, 10, .1), "openai:gpt-6.1-terra": (1, 5, .05), "anthropic:claude-opus-5-5": (4, 20, .4),
          "gemini:gemini-4-argon": (2, 10, .1)}
_lock = threading.Lock()


def _key(provider: str) -> str:
    return os.environ[f"{provider.upper()}_API_KEY"]


def ledger_path(provider: str) -> Path:
    return LIVE / f"ledger2_{provider}.json"


def spent(provider: str) -> float:
    p = ledger_path(provider)
    return json.loads(p.read_text())["usd"] if p.exists() else 0.0


class Ledgered(Spending):
    def add(self, editor, usage):
        before = self.usd
        super().add(editor, usage)
        provider = editor.split(":")[0].split("#")[0].replace("openai-chat", "gemini")
        p = ledger_path(provider)
        with _lock, open(str(p) + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(p.read_text()) if p.exists() else {"usd": 0.0, "calls": 0, "by_editor": {}}
            data["usd"] += self.usd - before
            data["calls"] += 1
            data["by_editor"][editor] = data["by_editor"].get(editor, 0.0) + self.usd - before
            p.write_text(json.dumps(data, indent=1))
        if data["usd"] >= LIMITS[provider]:
            self.max_usd = 0.0


def spending(cap: float) -> Ledgered:
    return Ledgered(max_usd=cap, prices=PRICES)


def editor(spec: str, effort: str = "high"):
    provider, _, model = spec.partition(":")
    if provider == "openai":
        return OpenAIEditor(model, api_key=_key("openai"), effort=effort)
    if provider == "anthropic":
        return AnthropicEditor(model, api_key=_key("anthropic"), effort=effort)
    if provider == "gemini":
        e = ChatEditor(model, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=_key("gemini"),
                       reasoning_effort=effort, max_tokens=32768)
        e.name = f"gemini:{model}"
        return e
    raise ValueError(spec)
