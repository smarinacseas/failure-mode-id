import json

from pipeline.monitor import LogSink, RunMonitor, StatusSink, WorkPlan


def test_status_sink_writes_valid_snapshot(tmp_path):
    path = tmp_path / "progress.json"
    m = RunMonitor(WorkPlan.for_step("grade", 2, 1), experiment="E99-test",
                   sinks=[StatusSink(path, min_interval=0.0)])
    with m:
        m.start_stage("grade", total=2)
        m.item_start(model="m1", prompt_id="p1")
        m.item_done()
    snap = json.loads(path.read_text())
    assert snap["experiment"] == "E99-test"
    assert snap["stages"][0]["name"] == "grade"
    assert snap["state"] == "done"                 # forced write on close
    assert set(snap) >= {"overall", "current", "retries", "errors", "eta_s"}


def test_status_sink_throttles(tmp_path):
    path = tmp_path / "progress.json"
    sink = StatusSink(path, min_interval=999.0)     # effectively "never again"
    m = RunMonitor(WorkPlan.for_step("grade", 5, 1), sinks=[sink])
    with m:                                          # __enter__ writes once
        first = path.read_text()
        m.start_stage("grade", total=5)
        m.item_start(); m.item_done()               # throttled — file unchanged
        assert path.read_text() == first
    # close() forces a final write regardless of throttle
    assert json.loads(path.read_text())["state"] == "done"


def test_log_sink_records_events(tmp_path):
    path = tmp_path / "run.log"
    m = RunMonitor(WorkPlan.for_step("grade", 1, 1), sinks=[LogSink(path)])
    with m:
        m.start_stage("grade", total=1)
        m.record_error("grade m1 p1: boom")
        m.note("skipped 2 incomplete prompts")
    text = path.read_text()
    assert "stage grade started" in text
    assert "ERROR" in text and "boom" in text
    assert "skipped 2 incomplete prompts" in text
