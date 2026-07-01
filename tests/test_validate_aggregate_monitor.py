import pipeline.aggregate as aggregate
import pipeline.validate as validate
from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan


def test_validate_marks_stage_done(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "sample", lambda mon=None: mon.note("sampled 0"))
    m = RunMonitor(WorkPlan.for_step("validate", 1, 1), sinks=[RecordingSink()])
    with m:
        validate.run(mode="sample", monitor=m)
    assert m.snapshot()["stages"][0]["state"] == "done"


def test_aggregate_marks_stage_done(monkeypatch):
    # Stub the whole extracted body so no real git/file I/O runs; we only
    # assert the monitor stage lifecycle here.
    called = {}
    monkeypatch.setattr(aggregate, "_run",
                        lambda limit, exp, desc, rr, mon: called.setdefault("ok", True))
    m = RunMonitor(WorkPlan.for_step("aggregate", 1, 1), sinks=[RecordingSink()])
    with m:
        aggregate.run(limit=0, monitor=m)
    assert called["ok"] and m.snapshot()["stages"][0]["state"] == "done"
