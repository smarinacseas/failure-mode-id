import config
import pipeline.generate as generate
from conftest import make_cfg
from pipeline._io import read_jsonl, write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def _setup(tmp_path, monkeypatch):
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "a", "criteria": []},
                       {"id": "p2", "prompt": "b", "criteria": []}])
    monkeypatch.setattr(generate, "DATA_JSONL", data)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    return make_cfg()


def test_generate_continue_on_error(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch)

    def fake_one(cfg_, model_id, prompt):
        if prompt == "b":
            raise RuntimeError("boom after retries")
        return "RESP"
    monkeypatch.setattr(generate, "_generate_one", fake_one)

    m = RunMonitor(WorkPlan.for_step("generate", 2, 1), sinks=[RecordingSink()])
    with m:
        generate.run(cfg, monitor=m)

    written = read_jsonl(cfg.responses_path("m1"))
    assert [r["id"] for r in written] == ["p1"]      # p2 skipped, not written
    snap = m.snapshot()
    assert snap["errors"] == 1
    assert snap["stages"][0]["done"] == 2            # both attempted → bar completes


def test_generate_seeds_already_done(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch)
    write_jsonl(cfg.responses_path("m1"), [{"id": "p1", "response": "done earlier"}])
    monkeypatch.setattr(generate, "_generate_one", lambda c, m, p: "RESP")

    m = RunMonitor(WorkPlan.for_step("generate", 2, 1), sinks=[RecordingSink()])
    with m:
        generate.run(cfg, monitor=m)
    # 1 pre-existing + 1 processed == 2 total
    assert m.snapshot()["stages"][0]["done"] == 2


def test_generate_respects_cfg_limit(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch)
    cfg = make_cfg(limit=1)
    monkeypatch.setattr(generate, "_generate_one", lambda c, m, p: "RESP")
    m = RunMonitor(WorkPlan.for_step("generate", 1, 1), sinks=[RecordingSink()])
    with m:
        generate.run(cfg, monitor=m)
    assert [r["id"] for r in read_jsonl(cfg.responses_path("m1"))] == ["p1"]


def test_generate_empty_completion_is_error_not_data(tmp_path, monkeypatch):
    """A provider returning empty content must NOT be stored as a valid response
    (it would silently grade 0/N downstream). It's an error: recorded, skipped,
    resumable. Regression test for the E03 qwen-397b/CIF-002 empty cell."""
    cfg = _setup(tmp_path, monkeypatch)

    class _EmptyChoice:
        message = type("M", (), {"content": "   "})()
    class _EmptyRouter:
        class chat:
            class completions:
                @staticmethod
                def create(**_k):
                    return type("R", (), {"choices": [_EmptyChoice()]})()

    monkeypatch.setattr(generate, "router", _EmptyRouter)
    # retry() must treat empty completions as retriable, then give up loudly.
    monkeypatch.setattr(generate, "retry",
                        lambda fn, label: fn())  # single attempt: surfacing the raise is what we test

    m = RunMonitor(WorkPlan.for_step("generate", 2, 1), sinks=[RecordingSink()])
    with m:
        generate.run(cfg, monitor=m)

    assert read_jsonl(cfg.responses_path("m1")) == []   # nothing written
    snap = m.snapshot()
    assert snap["errors"] == 2                          # both prompts recorded as errors
    assert snap["stages"][0]["done"] == 2               # bar still completes
