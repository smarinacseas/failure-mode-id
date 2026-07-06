import json
from types import SimpleNamespace

import config
import pipeline.aggregate as aggregate
import pipeline.validate as validate
from conftest import make_cfg
from pipeline._io import append_jsonl, write_jsonl
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_validate_marks_stage_done(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()
    monkeypatch.setattr(validate, "sample", lambda cfg, mon=None: mon.note("sampled 0"))
    m = RunMonitor(WorkPlan.for_step("validate", 1, 1), sinks=[RecordingSink()])
    with m:
        validate.run(cfg, mode="sample", monitor=m)
    assert m.snapshot()["stages"][0]["state"] == "done"


def test_aggregate_uses_seeded_sample_not_first_n(tmp_path, monkeypatch):
    """With sample_seed frozen, aggregate must compose results over the SAME
    stratified subset the other stages used — slicing records[:limit] would
    aggregate the wrong prompts. p4 is a singleton use case at the end of the
    file: the sampled subset must include it; first-N never would."""
    data = tmp_path / "prompts.jsonl"
    recs = [
        {"id": pid, "prompt": pid, "use_case": uc,
         "instruction_type": "Negative", "prompt_style": "Direct",
         "criteria": ["does the thing"]}
        for pid, uc in (("p1", "A"), ("p2", "A"), ("p3", "A"), ("p4", "B"))
    ]
    write_jsonl(data, recs)
    monkeypatch.setattr(aggregate, "DATA_JSONL", data)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg(limit=2, sample_seed=1)

    # Complete artifacts for ALL 4 prompts so eligibility can't mask selection.
    for pid in ("p1", "p2", "p3", "p4"):
        append_jsonl(cfg.responses_path("m1"), {"id": pid, "response": "resp"})
        append_jsonl(cfg.grades_path("claude-fable-5", "m1"),
                     {"id": pid, "verdicts": [{"index": 1, "verdict": "PASS", "reason": ""}]})
        append_jsonl(cfg.criteria_tags_path,
                     {"id": pid, "tags": [{"index": 1, "verifiability": "auto",
                                           "gameable": False, "reward_hack": "",
                                           "ambiguous": False}]})

    results_path = tmp_path / "results.json"
    monkeypatch.setattr(aggregate, "RESULTS_PATH", results_path)
    monkeypatch.setattr(aggregate, "write_experiment_copy", lambda slug, results: tmp_path / "e.json")
    monkeypatch.setattr(aggregate, "update_index", lambda slug, meta: tmp_path / "i.json")
    monkeypatch.setattr(aggregate, "_sync_dashboard", lambda mon: None)
    monkeypatch.setattr(aggregate, "_merge_manifest", lambda cfg, update: None)
    monkeypatch.setattr(aggregate, "build_meta", lambda cfg, rr, rd, prompts: {
        "counts": {"n_prompts": len(prompts), "n_criteria": 1},
        "experiment": {}, "models": [], "judges": [], "judge": "",
        "git": {}, "config": {}, "validation": {},
    })

    aggregate._run(cfg, None, SimpleNamespace(note=lambda *a, **k: None))

    results = json.loads(results_path.read_text())
    ids = {p["id"] for p in results["prompts"]}
    assert "p4" in ids and len(ids) == 2


def test_aggregate_marks_stage_done(tmp_path, monkeypatch):
    # Stub the whole extracted body so no real git/file I/O runs; we only
    # assert the monitor stage lifecycle here.
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()
    called = {}
    monkeypatch.setattr(aggregate, "_run",
                        lambda cfg, run_report, mon: called.setdefault("ok", True))
    m = RunMonitor(WorkPlan.for_step("aggregate", 1, 1), sinks=[RecordingSink()])
    with m:
        aggregate.run(cfg, monitor=m)
    assert called["ok"] and m.snapshot()["stages"][0]["state"] == "done"
