"""Progress messages: formatting, when they are printed, where they are written, and what never goes into them."""

import json
import logging
import sys
import threading

import pytest

from wsplab import log
from wsplab.runners import lm
from wsplab.runners.vllm_worker import progress_bar


@pytest.fixture
def clean_logging(monkeypatch):
    """No handlers of anyone else's, as in a plain script, and the wsplab logger as a fresh process has it. Call it in
    the test itself: pytest adds its own handler to the root logger after fixtures have run."""
    monkeypatch.setattr(log.logger, "handlers", [])
    level = log.logger.level
    log.logger.setLevel(logging.NOTSET)
    yield lambda: monkeypatch.setattr(logging.getLogger(), "handlers", [])
    log.logger.setLevel(level)


def test_numbers_read_naturally():
    assert log.duration(42) == "42s"
    assert log.duration(725) == "12m 05s"
    assert log.duration(3 * 3600 + 120) == "3h 02m"
    assert log.duration(3 * 86400) == "3d 0h"
    assert log.tokens(950) == "950" and log.tokens(12_345) == "12.3k" and log.tokens(65_536_000) == "65.5M"
    assert log.plural(1, "tensor") == "1 tensor" and log.plural(2, "pass", "passes") == "2 passes"


def test_progress_reports_now_and_then(caplog):
    caplog.set_level(logging.DEBUG, logger="wsplab")
    now = [0.0]
    p = log.Progress("baseline evaluation", 2000, clock=lambda: now[0])
    for t, done in ((30, 100), (90, 120), (100, 130), (200, 400), (900, 410)):
        now[0] = t
        p.update(done, 1300.0)
    now[0] = 1000
    p.finish()
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    # nothing before a minute; then at 90s; not again until 5% more is done (200s) or 10 minutes pass (900s)
    assert lines[0] == "baseline evaluation: 120/2,000 answers (6%), 1,300 tokens/s, about 23m 30s left"
    assert lines[1].startswith("baseline evaluation: 400/2,000 answers (20%)")
    assert lines[2].startswith("baseline evaluation: 410/2,000 answers (20%)")
    assert lines[3] == "baseline evaluation: 2,000 answers in 16m 40s"
    assert len(lines) == 4


def test_short_generations_stay_out_of_the_way(caplog):
    caplog.set_level(logging.DEBUG, logger="wsplab")
    now = [0.0]
    p = log.Progress("forward pass 1/25", 8, clock=lambda: now[0])
    now[0] = 20
    p.update(8)
    p.finish()
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


def test_a_run_prints_and_writes_run_log(tmp_path, clean_logging, capsys):
    clean_logging()
    with log.run_log(tmp_path):
        log.info("baseline: 0.1000")
    assert "baseline: 0.1000" in capsys.readouterr().err
    assert "INFO    baseline: 0.1000" in (tmp_path / "run.log").read_text()
    assert log.logger.handlers == [] and log.logger.level == logging.NOTSET  # left as it was found


def test_quiet_only_writes_the_file(tmp_path, clean_logging, capsys):
    clean_logging()
    with log.run_log(tmp_path, quiet=True):
        log.info("turn 1: Python run 1")
    assert capsys.readouterr().err == ""
    assert "turn 1: Python run 1" in (tmp_path / "run.log").read_text()


def test_your_own_logging_setup_wins(tmp_path, clean_logging, capsys):
    seen = []

    class Mine(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    clean_logging()
    logging.getLogger().addHandler(Mine())
    with log.run_log(tmp_path):
        log.info("verdict: confirmed")
    assert seen == ["verdict: confirmed"]  # reached your handler, and was not printed a second time by wsplab
    assert capsys.readouterr().err == ""
    assert "verdict: confirmed" in (tmp_path / "run.log").read_text()
    log.logger.setLevel(logging.WARNING)  # and you can turn progress off
    with log.run_log(tmp_path):
        log.info("not wanted")
    assert "not wanted" not in seen


def test_nested_runs_and_council_members(tmp_path, clean_logging):
    outer, inner = tmp_path / "search", tmp_path / "search" / "run_0"
    inner.mkdir(parents=True)
    with log.run_log(outer, quiet=True):
        log.info("search: 2 sessions")
        with log.run_log(outer, quiet=True):  # run() handing over to run_lm(): no second copy of each line
            log.info("once")
        parent = log.run_context()

        def member():
            with log.run_log(inner, quiet=True), log.labelled("0: anthropic:claude-opus-5-5"):
                log.info("turn 1: Python run 1")

        threads = [threading.Thread(target=lambda: parent.copy().run(member)),
                   threading.Thread(target=lambda: log.info("from a thread that was not given the context"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    outer_log, inner_log = (outer / "run.log").read_text(), (inner / "run.log").read_text()
    assert outer_log.count("once") == 1
    assert "[0: anthropic:claude-opus-5-5] turn 1: Python run 1" in outer_log
    assert "turn 1: Python run 1" in inner_log and "search: 2 sessions" not in inner_log
    assert "not given the context" not in outer_log


def test_the_cli_sets_how_much_is_printed(clean_logging):
    log.configure(logging.WARNING)
    assert log.logger.handlers[0].level == logging.WARNING
    log.configure(logging.DEBUG)
    assert len(log.logger.handlers) == 1 and log.logger.level == logging.DEBUG
    log.logger.handlers.clear()


def test_the_vllm_worker_writes_its_progress_to_a_file(tmp_path):
    path = tmp_path / "progress.json"
    bar = progress_bar(str(path), total=8, every=0)(total=2, desc="Processed prompts")
    assert json.loads(path.read_text()) == {"done": 0, "total": 8, "out_tokens_per_s": None}
    bar.postfix = "est. speed input: 10.00 toks/s, output: 1234.50 toks/s"
    bar.update(4)
    assert json.loads(path.read_text()) == {"done": 4, "total": 8, "out_tokens_per_s": 1234.5}
    bar.update(4)
    bar.close()
    assert json.loads(path.read_text())["done"] == 8


def test_the_runner_follows_a_worker_and_keeps_its_output(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="wsplab")
    monkeypatch.setattr(lm, "POLL_SECONDS", 0.05)
    progress = tmp_path / "progress.json"
    script = (f"import json, time\nfor d in (1, 2):\n    json.dump({{'done': d, 'total': 2}}, open({str(progress)!r}, 'w'))\n"
              f"    time.sleep(0.2)\nprint('engine says hello')\n")
    with log.activity("baseline evaluation"):
        code, output = lm._run_worker([sys.executable, "-c", script], {}, str(progress), 2, str(tmp_path / "w.log"))
    assert code == 0 and "engine says hello" in output
    assert any("baseline evaluation: 2 answers to generate" in r.getMessage() for r in caplog.records)
