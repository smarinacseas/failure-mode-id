"""Rejected-attempt capture: doomed generation drafts (empty completion,
deadline abort) must be handed to on_reject BEFORE the raise, so retries no
longer discard the runaway text — it is training signal (loops-are-failures
ruling, 2026-07-09). Mocked streams only; zero real API calls.
"""
import threading
import time
from types import SimpleNamespace

import pytest

import config
import pipeline.generate as generate
from conftest import make_cfg
from pipeline._io import read_jsonl
from pipeline.generate import GenerationDeadlineExceeded


def _chunk(content=None, reasoning=None, finish=None):
    delta = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


class _Stream:
    def __init__(self, chunks=(), endless=0.0):
        self._chunks = list(chunks)
        self._endless = endless
        self._closed = threading.Event()

    def close(self):
        self._closed.set()

    def __iter__(self):
        return self

    def __next__(self):
        if self._chunks:
            return self._chunks.pop(0)
        if self._endless and not self._closed.is_set():
            time.sleep(self._endless)
            return _chunk(reasoning="loop loop loop ")
        raise StopIteration


def _router(stream_factory):
    def create(**kwargs):
        return stream_factory()
    return SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))


def _single_attempt(monkeypatch):
    monkeypatch.setattr(generate, "retry", lambda fn, label: fn())


def test_empty_completion_invokes_on_reject_then_raises(monkeypatch):
    _single_attempt(monkeypatch)
    monkeypatch.setattr(config, "router", _router(lambda: _Stream(chunks=[
        _chunk(reasoning="ruminating forever "),
        _chunk(reasoning="ruminating forever ", finish="length"),
    ])))
    captured = []
    with pytest.raises(RuntimeError, match="empty completion"):
        generate._generate_one(make_cfg(), "prov/m", "p",
                               on_reject=captured.append)
    assert len(captured) == 1
    assert captured[0]["error"] == "empty_completion"
    assert captured[0]["finish_reason"] == "length"
    assert "ruminating" in captured[0]["reasoning"]
    assert captured[0]["response"] == ""


def test_deadline_abort_invokes_on_reject_with_partial(monkeypatch):
    _single_attempt(monkeypatch)
    monkeypatch.setattr(config, "router",
                        _router(lambda: _Stream(endless=0.02)))
    captured = []
    with pytest.raises(GenerationDeadlineExceeded):
        generate._generate_one(make_cfg(), "prov/m", "p", deadline_s=0.1,
                               on_reject=captured.append)
    assert len(captured) == 1
    assert captured[0]["error"] == "deadline"
    assert "loop" in captured[0]["reasoning"]


def test_run_item_writes_rejected_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()

    def fake_generate(cfg_, model_id, prompt, deadline_s=None, on_reject=None):
        on_reject({"error": "empty_completion", "finish_reason": "length",
                   "response": "", "reasoning": "xyz " * 10})
        raise RuntimeError("empty completion content (fake)")

    monkeypatch.setattr(generate, "_generate_one", fake_generate)
    mon = SimpleNamespace(item_start=lambda **k: None,
                          item_done=lambda **k: None,
                          record_error=lambda *a, **k: None)
    generate._run_item(cfg, mon, "m1", "prov/m", {"id": "p1", "prompt": "p"})
    rows = read_jsonl(cfg.run_dir / "rejected" / "m1.jsonl")
    assert len(rows) == 1
    assert rows[0]["id"] == "p1"
    assert rows[0]["error"] == "empty_completion"
