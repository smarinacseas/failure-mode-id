"""Task 9: diagnose fallback chain + consensus-FAIL targeting + analyst provenance.

Self-contained (like tests/test_diagnose.py): the Anthropic Message Batches
endpoint is a scripted fake so the member-loop transport is exercised without
network. DIAGNOSE_CHAIN's default head is Fable, fallback Opus — the analyst
chain is independent of the grading panel."""

import json
from types import SimpleNamespace

import config
import pipeline.diagnose as diagnose
from conftest import make_cfg
from pipeline._io import read_jsonl, write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan

FABLE = "claude-fable-5"
OPUS = "claude-opus-4-8"


# --- consensus targeting ---------------------------------------------------

def test_failed_cells_use_consensus(tmp_path, monkeypatch):
    """_failed_cells targets PANEL-CONSENSUS FAILs, not first-judge FAILs.

    Two judges, Fable first (== canonical) and Opus second:
      index 1: Fable PASS, Opus FAIL(real) → 2-judge tie → FAIL panel_tie →
               TARGETED. This is the RED-making case: the pre-Task-9 code read
               ONLY the first judge's grades, saw Fable's PASS, and skipped it.
      index 2: Fable FAIL(judge_refusal → abstains), Opus PASS → one cast vote
               < quorum(2) → panel_no_quorum → NOT targeted.
    Exactly index 1 survives."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(judges=(FABLE, OPUS))
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "Do it.", "use_case": "A",
                        "instruction_type": "Neg", "prompt_style": "Direct",
                        "criteria": ["c1 text", "c2 text"]}])
    monkeypatch.setattr(diagnose, "DATA_JSONL", data)
    write_jsonl(cfg.responses_path("m1"), [{"id": "p1", "response": "resp"}])
    write_jsonl(cfg.grades_path(FABLE, "m1"), [{"id": "p1", "verdicts": [
        {"index": 1, "verdict": "PASS", "reason": ""},
        {"index": 2, "verdict": "FAIL", "reason": "judge_refusal: declined"}]}])
    write_jsonl(cfg.grades_path(OPUS, "m1"), [{"id": "p1", "verdicts": [
        {"index": 1, "verdict": "FAIL", "reason": "missed the constraint"},
        {"index": 2, "verdict": "PASS", "reason": ""}]}])

    records = read_jsonl(diagnose.DATA_JSONL)
    (cell,) = diagnose._failed_cells(cfg, records)
    assert (cell["key"], cell["rid"]) == ("m1", "p1")
    assert cell["failed_indices"] == [1]


def test_failed_cells_single_judge_behaves_identically(tmp_path, monkeypatch):
    """A single-judge panel degrades quorum to 1, so consensus targeting
    reproduces the pre-panel first-judge behavior exactly: a real FAIL is
    targeted, an artifact FAIL abstains → panel_no_quorum → not targeted, and a
    PASS is not targeted."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(judges=(OPUS,))
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "Do it.", "use_case": "A",
                        "instruction_type": "Neg", "prompt_style": "Direct",
                        "criteria": ["c1 text", "c2 text", "c3 text"]}])
    monkeypatch.setattr(diagnose, "DATA_JSONL", data)
    write_jsonl(cfg.responses_path("m1"), [{"id": "p1", "response": "resp"}])
    write_jsonl(cfg.grades_path(OPUS, "m1"), [{"id": "p1", "verdicts": [
        {"index": 1, "verdict": "PASS", "reason": ""},
        {"index": 2, "verdict": "FAIL", "reason": "real miss"},
        {"index": 3, "verdict": "FAIL", "reason": "judge_truncated: max_tokens"}]}])

    records = read_jsonl(diagnose.DATA_JSONL)
    (cell,) = diagnose._failed_cells(cfg, records)
    assert cell["failed_indices"] == [2]


# --- scripted Message Batches fake (mirrors tests/test_diagnose.py) --------

def _diag_json(index=2, root="judge_suspect"):
    return json.dumps([{"index": index, "evidence": "quote", "root_cause": root,
                        "secondary": None, "confidence": "high",
                        "rationale": "because"}])


def _succeeded(cid, text, stop_reason="end_turn"):
    msg = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                          stop_reason=stop_reason)
    return SimpleNamespace(custom_id=cid,
                           result=SimpleNamespace(type="succeeded", message=msg))


class FakeBatches:
    """Submission N consumes script[N] = {"polls":[...], "results": callable}."""

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


def _seed_consensus_fail(tmp_path, monkeypatch):
    """One consensus-targeted cell (m1/p1): both panel judges FAIL index 2 with
    real reasons (index 1 PASS). The DIAGNOSE_CHAIN (Fable→Opus) supplies the
    analyst, independent of the panel."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(judges=(FABLE, OPUS))
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "Do it. Never mention cats.",
                        "use_case": "A", "instruction_type": "Neg",
                        "prompt_style": "Direct", "criteria": ["c1 text", "c2 text"]}])
    monkeypatch.setattr(diagnose, "DATA_JSONL", data)
    write_jsonl(cfg.responses_path("m1"),
                [{"id": "p1", "response": "A plan featuring cats.",
                  "reasoning": "hmm"}])
    for j in (FABLE, OPUS):
        write_jsonl(cfg.grades_path(j, "m1"), [{"id": "p1", "verdicts": [
            {"index": 1, "verdict": "PASS", "reason": ""},
            {"index": 2, "verdict": "FAIL", "reason": "mentions cats"}]}])
    return cfg


def _install_fake(monkeypatch, script):
    fake = FakeBatches(script)
    monkeypatch.setattr(diagnose, "anthropic",
                        SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    monkeypatch.setattr(diagnose, "_sleep", lambda s: None)
    monkeypatch.setattr(diagnose, "_synthesize", lambda cfg, mon: None)
    return fake


def _diag_monitor():
    return RunMonitor(WorkPlan.for_step("diagnose", 1, 1), sinks=[RecordingSink()])


def test_refused_cell_advances_to_next_chain_member(tmp_path, monkeypatch):
    """Member 1 (Fable) refuses the cell → it is NOT written terminally and NOT
    resubmitted to Fable; it advances to member 2 (Opus), which succeeds. The
    resulting diagnosis is real (not an _error_rows row) and carries
    analyst == Opus. Exactly two batches — a same-member refusal resubmission
    would make it three."""
    cfg = _seed_consensus_fail(tmp_path, monkeypatch)
    script = [
        {"polls": ["ended"],                       # member 1 (Fable): refusal
         "results": lambda reqs: [_succeeded(reqs[0]["custom_id"], "", "refusal")]},
        {"polls": ["ended"],                       # member 2 (Opus): success
         "results": lambda reqs: [_succeeded(reqs[0]["custom_id"], _diag_json())]},
    ]
    fake = _install_fake(monkeypatch, script)
    m = _diag_monitor()
    with m:
        diagnose.run(cfg, monitor=m)

    assert len(fake.created) == 2
    assert fake.created[0][0]["params"]["model"] == FABLE
    assert fake.created[1][0]["params"]["model"] == OPUS
    (row,) = read_jsonl(cfg.diagnosis_path("m1"))
    assert row["analyst"] == OPUS
    assert row["diagnoses"][0]["root_cause"] == "judge_suspect"    # real diagnosis
    assert m.snapshot()["errors"] == 0


def test_chain_exhaustion_writes_error_rows(tmp_path, monkeypatch):
    """Both chain members refuse → terminal _error_rows carrying analyst == the
    LAST member (Opus). Two batches only: each refusal is sticky per member, so
    neither member resubmits round 2."""
    cfg = _seed_consensus_fail(tmp_path, monkeypatch)
    refuse = {"polls": ["ended"],
              "results": lambda reqs: [_succeeded(reqs[0]["custom_id"], "", "refusal")]}
    fake = _install_fake(monkeypatch, [dict(refuse), dict(refuse)])
    m = _diag_monitor()
    with m:
        diagnose.run(cfg, monitor=m)

    assert len(fake.created) == 2
    (row,) = read_jsonl(cfg.diagnosis_path("m1"))
    assert row["analyst"] == OPUS
    assert row["diagnoses"][0]["root_cause"] == "other"           # terminal row
    assert row["diagnoses"][0]["rationale"].startswith("diagnose_refusal")
    assert m.snapshot()["errors"] == 1


def test_run_member_stream_serial_fallback(tmp_path, monkeypatch):
    """OpenRouter chain members walk pending cells serially via call_json with
    the same one-retry rule as the batch path: a retriable parse failure retries
    once with the SAME member, then a clean parse writes analyst == member.key
    and drops the cell from `pending`."""
    cfg = _seed_consensus_fail(tmp_path, monkeypatch)
    records = read_jsonl(diagnose.DATA_JSONL)
    (cell,) = diagnose._failed_cells(cfg, records)
    user_msg, trace_status = diagnose._user_message(cell)
    pending = {"m1__p1": {**cell, "user_msg": user_msg, "trace_status": trace_status,
                          "last_err": "", "refused_by": set()}}
    member = SimpleNamespace(key="gpt-5", client="openrouter", model="openai/gpt-5.2")

    calls = {"n": 0}

    def fake_call_json(spec, system, user_msg, label):
        calls["n"] += 1
        assert spec is member and label == "openrouter:gpt-5:diagnose"
        return ("not json", "stop") if calls["n"] == 1 else (_diag_json(), "stop")

    monkeypatch.setattr(diagnose, "call_json", fake_call_json)

    errors: list = []
    mon = SimpleNamespace(item_start=lambda **k: None, item_done=lambda **k: None,
                          record_error=lambda *a, **k: errors.append(a),
                          note=lambda *a, **k: None)
    diagnose._run_member_stream(cfg, mon, member, pending)

    assert calls["n"] == 2 and pending == {} and errors == []
    (row,) = read_jsonl(cfg.diagnosis_path("m1"))
    assert row["analyst"] == "gpt-5"
    assert row["diagnoses"][0]["root_cause"] == "judge_suspect"
