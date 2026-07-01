import pipeline.classify as classify
import pipeline.grade as grade
from pipeline._io import write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_grade_emits_events(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "prompts.jsonl",
                [{"id": "p1", "prompt": "a", "criteria": ["c1"]}])
    write_jsonl(tmp_path / "m1.jsonl", [{"id": "p1", "response": "r"}])
    monkeypatch.setattr(grade, "DATA_JSONL", tmp_path / "prompts.jsonl")
    monkeypatch.setattr(grade, "RESPONSES_DIR", tmp_path)
    monkeypatch.setattr(grade, "GRADES_DIR", tmp_path / "grades")
    monkeypatch.setattr(grade, "CANDIDATES", {"m1": "model-1"})
    monkeypatch.setattr(grade, "_grade_one",
                        lambda p, r, c: [{"index": 1, "verdict": "PASS", "reason": ""}])

    m = RunMonitor(WorkPlan.for_step("grade", 1, 1), sinks=[RecordingSink()])
    with m:
        grade.run(limit=None, monitor=m)
    assert m.snapshot()["stages"][0]["done"] == 1


def test_classify_emits_events(tmp_path, monkeypatch):
    write_jsonl(tmp_path / "prompts.jsonl",
                [{"id": "p1", "prompt": "a", "criteria": ["c1", "c2"]}])
    monkeypatch.setattr(classify, "DATA_JSONL", tmp_path / "prompts.jsonl")
    monkeypatch.setattr(classify, "CRITERIA_TAGS_PATH", tmp_path / "tags.jsonl")
    monkeypatch.setattr(classify, "_classify_one",
                        lambda criteria: [{"index": 1, "verifiability": "auto",
                                           "gameable": False, "reward_hack": "",
                                           "ambiguous": False}])
    m = RunMonitor(WorkPlan.for_step("classify", 1, 1), sinks=[RecordingSink()])
    with m:
        classify.run(limit=None, monitor=m)
    assert m.snapshot()["stages"][0]["done"] == 1
