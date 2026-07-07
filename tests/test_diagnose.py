"""Tests for the diagnose stage (pipeline/_taxonomy.py + pipeline/diagnose.py).

Self-contained: fixtures build a tiny run directory under tmp_path; the
Anthropic Message Batches endpoint is a scripted fake (mirrors
tests/test_concurrency.py's harness — duplicated deliberately so each test
file stands alone)."""

import json
from types import SimpleNamespace

import config
import pipeline._taxonomy as taxonomy


def test_taxonomy_reserved_labels_always_present():
    """judge_suspect and other are a-priori labels (spec §3): they must be
    offered in BOTH trace modes, before and after Pass-1 population."""
    for trace_present in (True, False):
        keys = taxonomy.allowed_keys(trace_present)
        assert "judge_suspect" in keys
        assert "other" in keys


def test_taxonomy_no_trace_mode_collapses_trace_dependent_categories():
    """Without a trace, 'never noticed' vs 'noticed-but-dropped' are
    indistinguishable (spec §3): trace-requiring categories are withheld and
    the collapse category constraint_unaddressed is offered instead."""
    with_trace = taxonomy.allowed_keys(True)
    without = taxonomy.allowed_keys(False)
    assert "constraint_unaddressed" in without
    assert "constraint_unaddressed" not in with_trace
    for cat in taxonomy.DERIVED:
        if cat["requires_trace"]:
            assert cat["key"] not in without


def test_diagnose_system_prompt_is_analyst_not_grader():
    """Spec §4 blinding rule 4: distinct analyst role, no JUDGE_SYSTEM reuse,
    evidence-first instruction, and every offered category documented."""
    text = taxonomy.diagnose_system(True)
    assert "PASS" not in text and "FAIL" not in text.replace("failed", "")
    assert "evidence" in text.lower()
    for key in taxonomy.allowed_keys(True):
        assert key in text


import pipeline.diagnose as diagnose
from conftest import make_cfg
from pipeline._io import read_jsonl, write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan

OPUS = "claude-opus-4-8"


def _cfg_opus(**kw):
    """Backfill-shaped cfg: canonical judge Opus, Fable second (like E03–E05)."""
    return make_cfg(judges=(OPUS, "claude-fable-5"), **kw)


def _seed_run(tmp_path, monkeypatch, *, reasoning_trace="I will check c2… it fails."):
    """One prompt, one model, 3 criteria: c1 PASS, c2 real FAIL, c3 FAIL from a
    grading artifact (judge_truncated) that must NOT become a diagnose target."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = _cfg_opus()
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "Write a plan. Never mention cats.",
                        "use_case": "A", "instruction_type": "Negative",
                        "prompt_style": "Direct",
                        "criteria": ["c1 text", "c2 text", "c3 text"]}])
    monkeypatch.setattr(diagnose, "DATA_JSONL", data)
    resp = {"id": "p1", "response": "A plan featuring cats."}
    if reasoning_trace is not None:
        resp["reasoning"] = reasoning_trace
    write_jsonl(cfg.responses_path("m1"), [resp])
    # MARKER_* strings exist so blinding tests can assert judge-reason text
    # never reaches an analyst payload.
    write_jsonl(cfg.grades_path(OPUS, "m1"), [{
        "id": "p1",
        "verdicts": [
            {"index": 1, "verdict": "PASS", "reason": "MARKER_PASS_REASON fine"},
            {"index": 2, "verdict": "FAIL", "reason": "MARKER_FAIL_REASON cats"},
            {"index": 3, "verdict": "FAIL",
             "reason": "judge_truncated: stop_reason=max_tokens"},
        ],
    }])
    return cfg


def test_failed_cells_target_real_fails_only(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch)
    records = read_jsonl(diagnose.DATA_JSONL)
    cells = diagnose._failed_cells(cfg, records)
    assert len(cells) == 1
    cell = cells[0]
    assert (cell["key"], cell["rid"]) == ("m1", "p1")
    assert cell["failed_indices"] == [2]        # 1 passed; 3 is an artifact
    assert cell["criteria"] == ["c1 text", "c2 text", "c3 text"]
    assert cell["reasoning"] == "I will check c2… it fails."


def test_failed_cells_skip_already_diagnosed_and_all_pass(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch)
    write_jsonl(cfg.diagnosis_path("m1"),
                [{"id": "p1", "trace_status": "present", "diagnoses": []}])
    records = read_jsonl(diagnose.DATA_JSONL)
    assert diagnose._failed_cells(cfg, records) == []
