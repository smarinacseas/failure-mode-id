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


def test_failed_cells_include_diagnosed_ignores_resume(tmp_path, monkeypatch):
    """Pass-1 open coding samples from ALL failed cells (spec §3) — the
    diagnosed-cell resume filter is Pass-2 behavior only."""
    cfg = _seed_run(tmp_path, monkeypatch)
    write_jsonl(cfg.diagnosis_path("m1"),
                [{"id": "p1", "trace_status": "present", "diagnoses": []}])
    records = read_jsonl(diagnose.DATA_JSONL)
    assert diagnose._failed_cells(cfg, records) == []              # resume default
    cells = diagnose._failed_cells(cfg, records, include_diagnosed=True)
    assert [c["rid"] for c in cells] == ["p1"]


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


def _succeeded(cid, text, stop_reason="end_turn"):
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                          stop_reason=stop_reason)
    return SimpleNamespace(custom_id=cid,
                           result=SimpleNamespace(type="succeeded", message=msg))


def _errored(cid, etype="api_error"):
    return SimpleNamespace(custom_id=cid,
                           result=SimpleNamespace(type="errored",
                                                  error=SimpleNamespace(type=etype)))


class FakeBatches:
    """Scripted Message Batches endpoint: submission N consumes script[N] =
    {"polls": [status, ...], "results": callable(requests) -> [result]}."""

    def __init__(self, script):
        self.script = script
        self.created: list[list[dict]] = []
        self._by_id: dict[str, dict] = {}

    def create(self, requests):
        idx = len(self.created)
        self.created.append(list(requests))
        bid = f"batch_{idx}"
        entry = dict(self.script[idx])
        entry["polls"] = list(entry.get("polls") or ["ended"])
        entry["requests"] = list(requests)
        self._by_id[bid] = entry
        return SimpleNamespace(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        entry = self._by_id[bid]
        status = entry["polls"].pop(0) if len(entry["polls"]) > 1 else entry["polls"][0]
        rc = SimpleNamespace(processing=0, succeeded=0, errored=0)
        return SimpleNamespace(id=bid, processing_status=status, request_counts=rc)

    def results(self, bid):
        entry = self._by_id[bid]
        return iter(entry["results"](entry["requests"]))


def _batch_setup(tmp_path, monkeypatch, script, **seed_kw):
    cfg = _seed_run(tmp_path, monkeypatch, **seed_kw)
    fake = FakeBatches(script)
    monkeypatch.setattr(diagnose, "anthropic",
                        SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    monkeypatch.setattr(diagnose, "_sleep", lambda s: None)
    monkeypatch.setattr(diagnose, "_synthesize", lambda cfg, mon: None)
    return cfg, fake


def _diag_monitor():
    return RunMonitor(WorkPlan.for_step("diagnose", 1, 1), sinks=[RecordingSink()])


def test_diagnose_batch_submit_collect_and_request_shape(tmp_path, monkeypatch):
    script = [{"polls": ["in_progress", "ended"],
               "results": lambda reqs: [_succeeded(reqs[0]["custom_id"], _diag_json())]}]
    cfg, fake = _batch_setup(tmp_path, monkeypatch, script)

    m = _diag_monitor()
    with m:
        diagnose.run(cfg, monitor=m)

    (req,) = fake.created[0]
    assert req["custom_id"] == "m1__p1"
    p = req["params"]
    assert p["model"] == config.DIAGNOSE_JUDGE
    assert p["max_tokens"] == config.DIAGNOSE_MAX_TOKENS
    assert p["thinking"] == {"type": "adaptive"}
    assert "failure analyst" in p["system"]
    assert "MARKER" not in p["messages"][0]["content"]      # blind end-to-end

    (row,) = read_jsonl(cfg.diagnosis_path("m1"))
    assert row["id"] == "p1" and row["trace_status"] == "present"
    assert row["diagnoses"][0]["root_cause"] == "judge_suspect"
    snap = m.snapshot()
    assert snap["errors"] == 0 and snap["stages"][0]["done"] == 1


def test_diagnose_refusal_is_terminal_without_resubmission(tmp_path, monkeypatch):
    script = [{"polls": ["ended"],
               "results": lambda reqs: [_succeeded(reqs[0]["custom_id"], "", "refusal")]}]
    cfg, fake = _batch_setup(tmp_path, monkeypatch, script)
    m = _diag_monitor()
    with m:
        diagnose.run(cfg, monitor=m)
    assert len(fake.created) == 1                            # no second round
    (row,) = read_jsonl(cfg.diagnosis_path("m1"))
    assert row["diagnoses"][0]["rationale"].startswith("diagnose_refusal")


def test_diagnose_parse_error_retries_once_then_error_rows(tmp_path, monkeypatch):
    script = [
        {"polls": ["ended"],
         "results": lambda reqs: [_succeeded(reqs[0]["custom_id"], "not json")]},
        {"polls": ["ended"],
         "results": lambda reqs: [_errored(reqs[0]["custom_id"])]},
    ]
    cfg, fake = _batch_setup(tmp_path, monkeypatch, script)
    m = _diag_monitor()
    with m:
        diagnose.run(cfg, monitor=m)
    assert len(fake.created) == 2
    (row,) = read_jsonl(cfg.diagnosis_path("m1"))
    assert row["diagnoses"][0]["root_cause"] == "other"
    assert m.snapshot()["errors"] == 1


def test_diagnose_noop_when_no_failures(tmp_path, monkeypatch):
    cfg, fake = _batch_setup(tmp_path, monkeypatch, [])
    write_jsonl(cfg.grades_path(OPUS, "m1"), [{
        "id": "p1", "verdicts": [{"index": i, "verdict": "PASS", "reason": ""}
                                 for i in (1, 2, 3)],
    }])
    m = _diag_monitor()
    with m:
        diagnose.run(cfg, monitor=m)
    assert fake.created == [] and not cfg.diagnosis_path("m1").exists()


import pipeline.aggregate as aggregate


def test_failure_analysis_block_joins_concurrence_and_rolls_up(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch)
    # Fable disagrees on criterion 2 (PASS) → opus_only concurrence.
    write_jsonl(cfg.grades_path("claude-fable-5", "m1"), [{
        "id": "p1",
        "verdicts": [
            {"index": 1, "verdict": "PASS", "reason": ""},
            {"index": 2, "verdict": "PASS", "reason": ""},
            {"index": 3, "verdict": "FAIL", "reason": ""},
        ],
    }])
    write_jsonl(cfg.diagnosis_path("m1"), [{
        "id": "p1", "trace_status": "present",
        "diagnoses": [{"index": 2, "evidence": "q", "root_cause": "judge_suspect",
                       "secondary": None, "confidence": "high", "rationale": "r"}],
    }])
    records = read_jsonl(diagnose.DATA_JSONL)
    grades_by_judge = {
        j: {"m1": {g["id"]: g["verdicts"]
                   for g in read_jsonl(cfg.grades_path(j, "m1"))}}
        for j in cfg.judges
    }
    block = aggregate._failure_analysis_block(cfg, records, grades_by_judge)
    assert block["taxonomy_version"] >= 1
    assert block["diagnose_judge"] == config.DIAGNOSE_JUDGE
    assert block["verdict_basis"] == OPUS
    assert block["counts"] == {"failed_criteria": 1, "diagnosed": 1, "cells": 1}
    (row,) = block["rows"]
    assert row["id"] == "p1" and row["model"] == "m1" and row["criterion_index"] == 2
    assert row["judge_concurrence"] == "opus_only"
    assert row["trace_status"] == "present"
    rc = block["by_root_cause"]["judge_suspect"]
    assert rc["total"] == 1
    assert rc["by_model"]["m1"] == 1
    assert rc["by_instruction_type"]["Negative"] == 1
    assert rc["by_use_case"]["A"] == 1
    # judge_suspect appears in the taxonomy echo (dashboard legend).
    assert any(c["key"] == "judge_suspect" for c in block["taxonomy"])


def test_failure_analysis_concurrence_both_fail(tmp_path, monkeypatch):
    """Characterization test (behavior shipped in the fold-in commit, not TDD):
    the majority case — second judge independently FAILs the same criterion
    with a normal reason — must map to both_fail, not opus_only/fable_refused."""
    cfg = _seed_run(tmp_path, monkeypatch)
    write_jsonl(cfg.grades_path("claude-fable-5", "m1"), [{
        "id": "p1",
        "verdicts": [
            {"index": 1, "verdict": "PASS", "reason": ""},
            {"index": 2, "verdict": "FAIL", "reason": "mentions cats"},
            {"index": 3, "verdict": "FAIL", "reason": ""},
        ],
    }])
    write_jsonl(cfg.diagnosis_path("m1"), [{
        "id": "p1", "trace_status": "present",
        "diagnoses": [{"index": 2, "evidence": "q", "root_cause": "other",
                       "secondary": None, "confidence": "high", "rationale": "r"}],
    }])
    records = read_jsonl(diagnose.DATA_JSONL)
    grades_by_judge = {
        j: {"m1": {g["id"]: g["verdicts"]
                   for g in read_jsonl(cfg.grades_path(j, "m1"))}}
        for j in cfg.judges
    }
    block = aggregate._failure_analysis_block(cfg, records, grades_by_judge)
    assert block["rows"][0]["judge_concurrence"] == "both_fail"


def test_failure_analysis_concurrence_refused_and_single_judge(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch)
    write_jsonl(cfg.diagnosis_path("m1"), [{
        "id": "p1", "trace_status": "present",
        "diagnoses": [{"index": 2, "evidence": "q", "root_cause": "other",
                       "secondary": None, "confidence": "low", "rationale": "r"}],
    }])
    records = read_jsonl(diagnose.DATA_JSONL)
    # Fable refused this cell → fable_refused.
    fable = {"m1": {"p1": [
        {"index": i, "verdict": "FAIL",
         "reason": "judge_refusal: model declined to grade this cell"}
        for i in (1, 2, 3)]}}
    opus = {"m1": {g["id"]: g["verdicts"]
                   for g in read_jsonl(cfg.grades_path(OPUS, "m1"))}}
    block = aggregate._failure_analysis_block(
        cfg, records, {OPUS: opus, "claude-fable-5": fable})
    assert block["rows"][0]["judge_concurrence"] == "fable_refused"

    # Single-judge cfg → no_second_judge. Distinct slug: E99-test's run dir
    # already holds this test's dual-judge artifacts.
    solo = make_cfg(slug="E98-solo", judges=(OPUS,))
    write_jsonl(solo.diagnosis_path("m1"), [{
        "id": "p1", "trace_status": "absent",
        "diagnoses": [{"index": 2, "evidence": "", "root_cause": "other",
                       "secondary": None, "confidence": "low", "rationale": "r"}],
    }])
    write_jsonl(solo.grades_path(OPUS, "m1"),
                [{"id": "p1", "verdicts": [
                    {"index": 2, "verdict": "FAIL", "reason": "x"}]}])
    block = aggregate._failure_analysis_block(solo, records, {OPUS: {
        "m1": {"p1": [{"index": 2, "verdict": "FAIL", "reason": "x"}]}}})
    assert block["rows"][0]["judge_concurrence"] == "no_second_judge"


def test_failure_analysis_absent_when_no_diagnosis_dir(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch)
    records = read_jsonl(diagnose.DATA_JSONL)
    grades_by_judge = {OPUS: {"m1": {}}}
    assert aggregate._failure_analysis_block(cfg, records, grades_by_judge) is None


def _index_and_results(tmp_path, monkeypatch, entries):
    """Fake outputs/experiments/: entries = [(slug, number, has_fa)]."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    idx = {"schema_version": "3.1", "experiments": [
        {"slug": s, "number": n, "results_path": f"experiments/{s}.json"}
        for s, n, _ in entries]}
    (exp_dir / "index.json").write_text(json.dumps(idx))
    for s, _n, has_fa in entries:
        doc = {"schema_version": "3.1"}
        if has_fa:
            doc["failure_analysis"] = {
                "counts": {"failed_criteria": 9, "diagnosed": 9, "cells": 3},
                "by_root_cause": {"other": {"total": 9, "by_model": {"m1": 9},
                                            "by_instruction_type": {},
                                            "by_use_case": {}}},
                "synthesis": {"recommendations": [
                    {"category": "coverage", "action": "run more prompts",
                     "rationale": "small n", "expected_signal": "tighter cells"}]},
            }
        (exp_dir / f"{s}.json").write_text(json.dumps(doc))
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", exp_dir)
    monkeypatch.setattr(config, "EXPERIMENT_INDEX_PATH", exp_dir / "index.json")


def test_predecessor_skips_analysis_less_experiments(tmp_path, monkeypatch):
    """E99 asks: E98 exists but has no failure_analysis (like E01/E02) → the
    chain skips it transparently and lands on E97."""
    _index_and_results(tmp_path, monkeypatch, [
        ("E96-old", 96, False), ("E97-prior", 97, True), ("E98-skip", 98, False)])
    slug, doc = diagnose._predecessor(make_cfg())          # E99-test → 99
    assert slug == "E97-prior"
    assert "failure_analysis" in doc

    _index_and_results(tmp_path, monkeypatch, [])
    assert diagnose._predecessor(make_cfg()) == (None, None)


def test_synthesize_writes_validates_and_resumes(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch)
    write_jsonl(cfg.diagnosis_path("m1"), [{
        "id": "p1", "trace_status": "present",
        "diagnoses": [{"index": 2, "evidence": "q", "root_cause": "other",
                       "secondary": None, "confidence": "low", "rationale": "r"}],
    }])
    _index_and_results(tmp_path, monkeypatch, [("E97-prior", 97, True)])
    calls = {"n": 0}

    def fake_call_json(model, system, user_msg, label):
        calls["n"] += 1
        # Blind-side check of the INPUT: rollups only — no response text,
        # no criterion text, and the predecessor's data made it in.
        assert "A plan featuring cats." not in user_msg
        assert "c2 text" not in user_msg
        assert "E97-prior" in user_msg and "run more prompts" in user_msg
        return json.dumps({
            "comparison": ["other-rate flat vs E97"],
            "prior_recommendations_review": ["coverage rec not yet addressed"],
            "recommendations": [
                {"category": "robustness", "action": "same config, limit 75",
                 "rationale": "cells too small", "expected_signal": "stable Pareto"},
                {"category": "bogus_category", "action": "x", "rationale": "y",
                 "expected_signal": "z"},
                {"category": "failure_mode_depth", "action": "hand-check 20 rows",
                 "rationale": "judge_suspect present", "expected_signal": "validated labels"},
                {"category": "coverage", "action": "4th rec", "rationale": "over cap",
                 "expected_signal": "dropped"},
            ],
            "iteration_note": "one more eval round before training",
        }), "end_turn"
    monkeypatch.setattr(diagnose, "call_json", fake_call_json)

    mon = SimpleNamespace(note=lambda *a, **k: None,
                          record_error=lambda *a, **k: None)
    diagnose._synthesize(cfg, mon)
    out = json.loads(cfg.synthesis_path.read_text())
    assert out["predecessor"] == "E97-prior"
    cats = [r["category"] for r in out["recommendations"]]
    assert cats == ["robustness", "failure_mode_depth", "coverage"]  # bogus dropped, capped later
    assert len(out["recommendations"]) <= 3
    assert out["iteration_note"]

    diagnose._synthesize(cfg, mon)                    # file exists → skip
    assert calls["n"] == 1


def test_synthesize_nonfatal_on_failure_and_skips_without_rows(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch)
    _index_and_results(tmp_path, monkeypatch, [])
    errors = []
    mon = SimpleNamespace(note=lambda *a, **k: None, record_error=errors.append)

    # No diagnosis rows on disk → nothing to synthesize, no call attempted.
    def boom(*a, **k):
        raise AssertionError("must not be called")
    monkeypatch.setattr(diagnose, "call_json", boom)
    diagnose._synthesize(cfg, mon)
    assert not cfg.synthesis_path.exists() and errors == []

    # Rows exist but both attempts fail → recorded error, no file, no raise.
    write_jsonl(cfg.diagnosis_path("m1"), [{
        "id": "p1", "trace_status": "present",
        "diagnoses": [{"index": 2, "evidence": "q", "root_cause": "other",
                       "secondary": None, "confidence": "low", "rationale": "r"}],
    }])
    def raiser(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr(diagnose, "call_json", raiser)
    diagnose._synthesize(cfg, mon)
    assert not cfg.synthesis_path.exists()
    assert errors and "synthesis" in errors[0]


def test_aggregate_folds_synthesis_block(tmp_path, monkeypatch):
    cfg = _seed_run(tmp_path, monkeypatch)
    write_jsonl(cfg.diagnosis_path("m1"), [{
        "id": "p1", "trace_status": "present",
        "diagnoses": [{"index": 2, "evidence": "q", "root_cause": "other",
                       "secondary": None, "confidence": "low", "rationale": "r"}],
    }])
    cfg.synthesis_path.write_text(json.dumps({
        "predecessor": "E97-prior", "comparison": [],
        "prior_recommendations_review": [], "recommendations": [],
        "iteration_note": "n"}))
    records = read_jsonl(diagnose.DATA_JSONL)
    opus = {"m1": {g["id"]: g["verdicts"]
                   for g in read_jsonl(cfg.grades_path(OPUS, "m1"))}}
    block = aggregate._failure_analysis_block(cfg, records, {OPUS: opus})
    assert block["synthesis"]["predecessor"] == "E97-prior"


def test_synthesize_nonfatal_on_malformed_artifacts(tmp_path, monkeypatch):
    """Spec §4b: synthesis failure is NON-fatal, including while preparing
    inputs — a corrupt-but-parseable diagnosis row (no root_cause) must
    record an error and skip synthesis, never raise into run()."""
    cfg = _seed_run(tmp_path, monkeypatch)
    write_jsonl(cfg.diagnosis_path("m1"), [{
        "id": "p1", "trace_status": "present",
        "diagnoses": [{"index": 2, "evidence": "q",
                       "confidence": "low", "rationale": "r"}],  # no root_cause
    }])
    _index_and_results(tmp_path, monkeypatch, [])
    errors = []
    mon = SimpleNamespace(note=lambda *a, **k: None, record_error=errors.append)
    monkeypatch.setattr(diagnose, "call_json",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")))
    diagnose._synthesize(cfg, mon)          # must NOT raise
    assert not cfg.synthesis_path.exists()
    assert errors and "synthesis" in errors[0]


def test_derived_taxonomy_is_populated_and_well_formed():
    """RED until the Pass-1 consolidation lands. Guards: non-empty enum, all
    required fields non-empty, keys unique + disjoint from reserved, at least
    one trace-dependent category (the CoT-vs-answer split is the taxonomy's
    reason to exist)."""
    assert len(taxonomy.DERIVED) >= 4
    seen = set()
    for c in taxonomy.DERIVED:
        for field in ("key", "label", "description", "training_implication"):
            assert isinstance(c[field], str) and c[field].strip()
        assert isinstance(c["requires_trace"], bool)
        assert c["key"] not in seen
        seen.add(c["key"])
    assert seen.isdisjoint({"judge_suspect", "other", "constraint_unaddressed"})
    assert any(c["requires_trace"] for c in taxonomy.DERIVED)
