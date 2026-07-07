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


def test_payload_is_blind(tmp_path, monkeypatch):
    """Spec §4 rules 1–3: no judge reason text, no candidate model name, no
    judge names anywhere in the analyst payload. The MARKER_* strings the
    fixture plants in grade reasons must not leak."""
    cfg = _seed_run(tmp_path, monkeypatch)
    records = read_jsonl(diagnose.DATA_JSONL)
    (cell,) = diagnose._failed_cells(cfg, records)
    payload, trace_status = diagnose._user_message(cell)
    for forbidden in ("MARKER_FAIL_REASON", "MARKER_PASS_REASON", "m1",
                      "claude-opus-4-8", "claude-fable-5", "PASS", "FAIL"):
        assert forbidden not in payload
    assert trace_status == "present"
    # What it MUST contain: prompt, every criterion, the unmet index, both texts.
    assert "Never mention cats" in payload
    for c in ("c1 text", "c2 text", "c3 text"):
        assert c in payload
    assert "2" in payload.split("UNMET")[1].splitlines()[0]
    assert "A plan featuring cats." in payload
    assert "I will check c2" in payload


def test_payload_trace_absent_and_truncated(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch, reasoning_trace=None)
    records = read_jsonl(diagnose.DATA_JSONL)
    (cell,) = diagnose._failed_cells(cfg, records)
    _, trace_status = diagnose._user_message(cell)
    assert trace_status == "absent"

    long = "x" * (config.DIAGNOSE_MAX_FIELD_CHARS + 5000)
    clipped, was_clipped = diagnose._clip(long)
    assert was_clipped and len(clipped) < len(long)
    assert clipped.startswith("x" * 100) and clipped.endswith("x" * 100)
    assert "TRUNCATED" in clipped
    short, was_clipped = diagnose._clip("short")
    assert short == "short" and not was_clipped


def test_user_message_truncated_and_absent_beats_truncation(tmp_path, monkeypatch):
    """Characterization of the trace_status contract through _user_message
    (locks it before Task 4 builds on the pair): an oversized trace yields
    "truncated"; with no trace, absence beats truncation even though the
    oversized response still gets clipped in the payload."""
    # Case A: oversized reasoning trace → "truncated", marker in payload.
    cfg = _seed_run(tmp_path, monkeypatch,
                    reasoning_trace="x" * (config.DIAGNOSE_MAX_FIELD_CHARS + 5000))
    records = read_jsonl(diagnose.DATA_JSONL)
    (cell,) = diagnose._failed_cells(cfg, records)
    payload, trace_status = diagnose._user_message(cell)
    assert trace_status == "truncated"
    assert "TRUNCATED" in payload

    # Case B: reasoning absent + oversized response → "absent" wins the flag,
    # but the response is still clipped (marker present).
    cfg = _seed_run(tmp_path, monkeypatch, reasoning_trace=None)
    write_jsonl(cfg.responses_path("m1"),
                [{"id": "p1",
                  "response": "y" * (config.DIAGNOSE_MAX_FIELD_CHARS + 5000)}])
    records = read_jsonl(diagnose.DATA_JSONL)
    (cell,) = diagnose._failed_cells(cfg, records)
    payload, trace_status = diagnose._user_message(cell)
    assert trace_status == "absent"
    assert "TRUNCATED" in payload


def _diag_json(index=2, root="judge_suspect"):
    return json.dumps([{"index": index, "evidence": "quote", "root_cause": root,
                        "secondary": None, "confidence": "high",
                        "rationale": "because"}])


def test_text_to_diagnoses_happy_path_and_unknown_label():
    rows, err = diagnose._text_to_diagnoses(_diag_json(), "end_turn", [2], True)
    assert err == "" and rows[0]["root_cause"] == "judge_suspect"

    # Unknown label → parse error (spec §9), NOT silent coercion.
    rows, err = diagnose._text_to_diagnoses(
        _diag_json(root="not_a_category"), "end_turn", [2], True)
    assert rows is None and err.startswith("diagnose_parse_error")

    # A no-trace cell must reject trace-dependent keys it wasn't offered:
    # constraint_unaddressed IS allowed, garbage is not.
    rows, err = diagnose._text_to_diagnoses(
        _diag_json(root="constraint_unaddressed"), "end_turn", [2], False)
    assert err == "" and rows[0]["root_cause"] == "constraint_unaddressed"


def test_text_to_diagnoses_refusal_truncation_and_missing_index():
    rows, err = diagnose._text_to_diagnoses("", "refusal", [2], True)
    assert rows is None and err.startswith("diagnose_refusal")
    rows, err = diagnose._text_to_diagnoses("[{", "max_tokens", [2], True)
    assert rows is None and err.startswith("diagnose_truncated")
    # Missing an assigned index → parse error (the cell retries whole).
    rows, err = diagnose._text_to_diagnoses(_diag_json(index=2), "end_turn", [2, 3], True)
    assert rows is None and err.startswith("diagnose_parse_error")


def test_error_rows_are_terminal_other_low():
    rows = diagnose._error_rows([2, 3], "diagnose_parse_error: boom")
    assert [r["index"] for r in rows] == [2, 3]
    assert all(r["root_cause"] == "other" and r["confidence"] == "low"
               and r["rationale"] == "diagnose_parse_error: boom" for r in rows)


def test_text_to_diagnoses_rejects_duplicate_indices():
    """Two diagnoses for the same criterion index = internally inconsistent
    analyst output -> parse error (whole cell retries), never a silent
    last-one-wins overwrite."""
    raw = json.dumps([
        {"index": 2, "evidence": "a", "root_cause": "judge_suspect",
         "secondary": None, "confidence": "high", "rationale": "r1"},
        {"index": 2, "evidence": "b", "root_cause": "other",
         "secondary": None, "confidence": "low", "rationale": "r2"},
    ])
    rows, err = diagnose._text_to_diagnoses(raw, "end_turn", [2], True)
    assert rows is None and err.startswith("diagnose_parse_error")
