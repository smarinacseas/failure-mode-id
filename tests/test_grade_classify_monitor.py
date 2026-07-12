import config
import pipeline.classify as classify
import pipeline.grade as grade
from conftest import make_cfg
from pipeline._io import write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan
from pipeline.run_config import JudgeSpec


def test_grade_emits_events(tmp_path, monkeypatch):
    # judge_mode="sequential": these tests exercise the per-cell call path
    # (_grade_one); the batch transport has its own tests in test_concurrency.py.
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(judge_mode="sequential")
    write_jsonl(tmp_path / "prompts.jsonl",
                [{"id": "p1", "prompt": "a", "criteria": ["c1"]}])
    write_jsonl(cfg.responses_path("m1"), [{"id": "p1", "response": "r"}])
    monkeypatch.setattr(grade, "DATA_JSONL", tmp_path / "prompts.jsonl")
    monkeypatch.setattr(grade, "_grade_one",
                        lambda cfg, p, r, c: [{"index": 1, "verdict": "PASS", "reason": ""}])

    m = RunMonitor(WorkPlan.for_step("grade", 1, 1), sinks=[RecordingSink()])
    with m:
        grade.run(cfg, monitor=m)
    assert m.snapshot()["stages"][0]["done"] == 1
    assert m.snapshot()["errors"] == 0


def test_classify_emits_events(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()
    write_jsonl(tmp_path / "prompts.jsonl",
                [{"id": "p1", "prompt": "a", "criteria": ["c1", "c2"]}])
    monkeypatch.setattr(classify, "DATA_JSONL", tmp_path / "prompts.jsonl")
    monkeypatch.setattr(classify, "_classify_one",
                        lambda cfg, criteria: ([{"index": 1, "verifiability": "auto",
                                           "gameable": False, "reward_hack": "",
                                           "ambiguous": False}], "claude-fable-5"))
    m = RunMonitor(WorkPlan.for_step("classify", 1, 1), sinks=[RecordingSink()])
    with m:
        classify.run(cfg, monitor=m)
    assert m.snapshot()["stages"][0]["done"] == 1
    assert m.snapshot()["errors"] == 0


def test_grade_records_error_on_judge_parse_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(judge_mode="sequential")
    write_jsonl(tmp_path / "prompts.jsonl", [{"id": "p1", "prompt": "a", "criteria": ["c1"]}])
    write_jsonl(cfg.responses_path("m1"), [{"id": "p1", "response": "r"}])
    monkeypatch.setattr(grade, "DATA_JSONL", tmp_path / "prompts.jsonl")
    monkeypatch.setattr(grade, "_grade_one",
                        lambda cfg, p, r, c: [{"index": 1, "verdict": "FAIL", "reason": "judge_parse_error: boom"}])
    m = RunMonitor(WorkPlan.for_step("grade", 1, 1), sinks=[RecordingSink()])
    with m:
        grade.run(cfg, monitor=m)
    assert m.snapshot()["errors"] == 1
    assert m.snapshot()["stages"][0]["done"] == 1


def test_grade_one_records_refusal_distinctly(monkeypatch):
    """A judge-level refusal (stop_reason=refusal, empty text) is not a parse
    error: record it as judge_refusal and don't burn a second attempt —
    refusals proved sticky (7/7 on E05 Fable × CIF-006)."""
    calls = {"n": 0}

    def fake_call_json(judge, system, user_msg, label):
        calls["n"] += 1
        return "", "refusal"
    monkeypatch.setattr(grade, "call_json", fake_call_json)

    out = grade._grade_one(JudgeSpec.from_value("claude-fable-5"),
                           "prompt", "resp", ["c1", "c2"])
    assert len(out) == 2
    assert all(v["verdict"] == "FAIL" for v in out)
    assert all(v["reason"].startswith("judge_refusal") for v in out)
    assert calls["n"] == 1


def test_classify_records_error_on_parse_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()
    write_jsonl(tmp_path / "prompts.jsonl", [{"id": "p1", "prompt": "a", "criteria": ["c1", "c2"]}])
    monkeypatch.setattr(classify, "DATA_JSONL", tmp_path / "prompts.jsonl")
    # Patch call_json to always fail parsing -> chain exhausts -> note_error
    from pipeline import _judge_llm
    monkeypatch.setattr(_judge_llm, "call_json", lambda *a, **k: ("not json at all", "stop"))
    m = RunMonitor(WorkPlan.for_step("classify", 1, 1), sinks=[RecordingSink()])
    with m:
        classify.run(cfg, monitor=m)
    assert m.snapshot()["errors"] == 1
    assert m.snapshot()["stages"][0]["done"] == 1
