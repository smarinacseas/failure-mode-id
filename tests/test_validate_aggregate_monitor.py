import config
import pipeline.aggregate as aggregate
import pipeline.validate as validate
from conftest import make_cfg
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_validate_marks_stage_done(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()
    monkeypatch.setattr(validate, "sample", lambda cfg, mon=None: mon.note("sampled 0"))
    m = RunMonitor(WorkPlan.for_step("validate", 1, 1), sinks=[RecordingSink()])
    with m:
        validate.run(cfg, mode="sample", monitor=m)
    assert m.snapshot()["stages"][0]["state"] == "done"


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
