import pipeline.generate as generate
from pipeline._io import write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_generate_continue_on_error(tmp_path, monkeypatch):
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "a", "criteria": []},
                       {"id": "p2", "prompt": "b", "criteria": []}])
    monkeypatch.setattr(generate, "DATA_JSONL", data)
    monkeypatch.setattr(generate, "RESPONSES_DIR", tmp_path)
    monkeypatch.setattr(generate, "CANDIDATES", {"m1": "model-1"})

    def fake_one(model_id, prompt):
        if prompt == "b":
            raise RuntimeError("boom after retries")
        return "RESP"
    monkeypatch.setattr(generate, "_generate_one", fake_one)

    m = RunMonitor(WorkPlan.for_step("generate", 2, 1), sinks=[RecordingSink()])
    with m:
        generate.run(limit=None, monitor=m)

    from pipeline._io import read_jsonl
    written = read_jsonl(tmp_path / "m1.jsonl")
    assert [r["id"] for r in written] == ["p1"]      # p2 skipped, not written
    snap = m.snapshot()
    assert snap["errors"] == 1
    assert snap["stages"][0]["done"] == 2            # both attempted → bar completes


def test_generate_seeds_already_done(tmp_path, monkeypatch):
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "a", "criteria": []},
                       {"id": "p2", "prompt": "b", "criteria": []}])
    write_jsonl(tmp_path / "m1.jsonl", [{"id": "p1", "response": "done earlier"}])
    monkeypatch.setattr(generate, "DATA_JSONL", data)
    monkeypatch.setattr(generate, "RESPONSES_DIR", tmp_path)
    monkeypatch.setattr(generate, "CANDIDATES", {"m1": "model-1"})
    monkeypatch.setattr(generate, "_generate_one", lambda m, p: "RESP")

    m = RunMonitor(WorkPlan.for_step("generate", 2, 1), sinks=[RecordingSink()])
    with m:
        generate.run(limit=None, monitor=m)
    # 1 pre-existing + 1 processed == 2 total
    assert m.snapshot()["stages"][0]["done"] == 2
