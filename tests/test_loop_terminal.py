"""Loop-terminal generation + loop-failure grading (ruling 2026-07-09).

A zero-content draw whose reasoning (or content) LOOPS is not a transient
error: it is the result. generate stores it flagged instead of retrying;
grade converts it to a mechanical all-FAIL (no judge call) so the prompt
stays in the aggregate as a failure instead of being excluded. Non-looping
empties remain retriable. Mocked streams/clients only.
"""
import threading
import time
from types import SimpleNamespace

import pytest

import config
import pipeline.generate as generate
import pipeline.grade as grade
from conftest import make_cfg
from pipeline._io import read_jsonl
from pipeline.run_config import JudgeSpec

LOOP_UNIT = "the same reasoning going around again. "   # 39 chars
FABLE = JudgeSpec.from_value("claude-fable-5")           # panel-spec plan entry


def _chunk(content=None, reasoning=None, finish=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


class _Stream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def close(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise StopIteration


def _router(chunks_factory):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: _Stream(chunks_factory()))))


def test_empty_looping_completion_returns_terminal_record(monkeypatch):
    monkeypatch.setattr(generate, "retry", lambda fn, label: fn())
    monkeypatch.setattr(generate, "router", _router(lambda: [
        _chunk(reasoning=LOOP_UNIT * 300),
        _chunk(reasoning=LOOP_UNIT * 100, finish="length"),
    ]))
    captured = []
    fields = generate._generate_one(make_cfg(), "prov/m", "p",
                                    on_reject=captured.append)
    assert fields["response"] == ""
    assert fields["finish_reason"] == "length"
    assert fields["loop_failure"]["channel"] == "reasoning"
    assert fields["loop_failure"]["period"] == len(LOOP_UNIT)
    assert LOOP_UNIT in fields["reasoning"]
    assert captured == []            # stored as THE result, not a rejected draft


def test_empty_nonlooping_completion_still_retries(monkeypatch):
    monkeypatch.setattr(generate, "retry", lambda fn, label: fn())
    varied = " ".join(f"w{i}" for i in range(2000))
    monkeypatch.setattr(generate, "router", _router(lambda: [
        _chunk(reasoning=varied, finish="length"),
    ]))
    captured = []
    with pytest.raises(RuntimeError, match="empty completion"):
        generate._generate_one(make_cfg(), "prov/m", "p",
                               on_reject=captured.append)
    assert len(captured) == 1
    assert captured[0]["error"] == "empty_completion"


def _mon():
    return SimpleNamespace(item_start=lambda **k: None,
                           item_done=lambda **k: None,
                           record_error=lambda *a, **k: None)


def _loop_rec(rid):
    return {"id": rid, "response": "", "finish_reason": "length",
            "reasoning": LOOP_UNIT * 400,
            "loop_failure": {"channel": "reasoning", "period": 39, "onset": 0}}


def test_sequential_grade_writes_mechanical_all_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(grade, "call_json",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("judge must not be called")))
    cfg = make_cfg()
    by_id = {"p1": {"id": "p1", "prompt": "p", "criteria": ["c1", "c2", "c3"]}}
    plan = [(FABLE, "m1", [_loop_rec("p1")])]
    grade._run_sequential(cfg, _mon(), by_id, plan)
    rows = read_jsonl(cfg.grades_path("claude-fable-5", "m1"))
    assert len(rows) == 1
    verdicts = rows[0]["verdicts"]
    assert len(verdicts) == 3
    assert all(v["verdict"] == "FAIL" for v in verdicts)
    assert all(v["reason"].startswith("loop_failure") for v in verdicts)


def test_batch_grade_never_submits_loop_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    exploding = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(
        create=lambda **k: (_ for _ in ()).throw(
            AssertionError("no batch may be created for loop failures")))))
    monkeypatch.setattr(grade, "anthropic", exploding)
    cfg = make_cfg()
    by_id = {"p1": {"id": "p1", "prompt": "p", "criteria": ["c1"]}}
    plan = [(FABLE, "m1", [_loop_rec("p1")])]
    mon = SimpleNamespace(item_start=lambda **k: None, item_done=lambda **k: None,
                          record_error=lambda *a, **k: None,
                          set_batch_counts=lambda **k: None,
                          note=lambda *a, **k: None)
    grade._run_batch(cfg, mon, by_id, plan)
    rows = read_jsonl(cfg.grades_path("claude-fable-5", "m1"))
    assert rows[0]["verdicts"][0]["reason"].startswith("loop_failure")
