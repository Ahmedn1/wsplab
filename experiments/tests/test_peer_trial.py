import pytest
from peer_trial import Guarded, Ledger

from wsplab.editors import BudgetExceeded, Reply, Usage


def test_inflight_calls_reserve_budget_and_usage_releases_the_difference(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", {"openai": 1.})
    ledger.reserve("openai", .7)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("openai", .4)
    ledger.finish("openai", .7, Usage(100_000, 20_000), 1.)
    assert ledger.spent["openai"] == pytest.approx(.2)
    assert ledger.reserved["openai"] == pytest.approx(0.)
    ledger.reserve("openai", .7)


def test_unknown_api_charge_is_retained_after_failure(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", {"gemini": 1.})
    ledger.reserve("gemini", .6)
    ledger.finish("gemini", .6, None, 1., "TimeoutError")
    assert ledger.spent["gemini"] == pytest.approx(.6)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("gemini", .5)


def test_guard_reuses_measured_prefix_tokens_but_recounts_changed_history(tmp_path):
    class Editor:
        name = "openai:gpt-6.1-terra"

        def __init__(self):
            self.params = {}

        def reply(self, system, messages):
            return Reply("next", Usage(2000, 20))

    guard = Guarded(Editor(), "openai", Ledger(tmp_path / "ledger.json", {"openai": 2.}))
    messages = [{"role": "user", "content": "x" * 10000}]
    guard.reply("system", messages)
    longer = messages + [{"role": "assistant", "content": "next"}, {"role": "user", "content": "result"}]
    assert guard.input_bound("system", longer) == 2000 + 4 + 100 + 6 + 100 + 1024
    changed = [{"role": "user", "content": "y" * 10000}]
    assert guard.input_bound("system", changed) > 10000


def test_nonempty_reply_with_zero_output_usage_reserves_its_output_ceiling(tmp_path):
    class Editor:
        name = "gemini:gemini-4-argon"

        def __init__(self):
            self.params = {}

        def reply(self, system, messages):
            return Reply("A nonempty answer", Usage(100, 0))

    ledger = Ledger(tmp_path / "ledger.json", {"gemini": 1.})
    guard = Guarded(Editor(), "gemini", ledger)
    reply = guard.reply("system", [{"role": "user", "content": "question"}])
    assert reply.usage.output_tokens == 32768
    assert ledger.calls[0]["usage_was_estimated"]
    assert ledger.calls[0]["reported_usage"]["output_tokens"] == 0
    assert ledger.spent["gemini"] > .32


def test_custom_output_ceiling_is_used_for_request_and_missing_usage(tmp_path):
    class Editor:
        name = "gemini:gemini-4-argon"

        def __init__(self):
            self.params = {}

        def reply(self, system, messages):
            assert self.params["max_tokens"] == 16384
            return Reply("An answer", Usage(100, 0))

    ledger = Ledger(tmp_path / "ledger.json", {"gemini": 1.})
    reply = Guarded(Editor(), "gemini", ledger, max_output=16384).reply("system", [])
    assert reply.usage.output_tokens == 16384
    assert ledger.spent["gemini"] == pytest.approx(.16404)
