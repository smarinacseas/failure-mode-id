import pytest

from pipeline.monitor import RecordingSink, RunMonitor, WorkPlan, render_lines
import pipeline.monitor as monitor


def _mon(step="all", limit=2, n=1):
    return RunMonitor(WorkPlan.for_step(step, limit, n), experiment="E99-test",
                      sinks=[RecordingSink()])


def test_item_events_advance_stage_and_overall():
    rec = RecordingSink()
    m = RunMonitor(WorkPlan.for_step("generate", limit=2, n_candidates=1), sinks=[rec])
    with m:
        m.start_stage("generate", total=2, already_done=0)
        for pid in ("p1", "p2"):
            m.item_start(model="m1", prompt_id=pid)
            m.item_done()
        m.end_stage()
    snap = m.snapshot()
    assert snap["stages"][0]["done"] == 2
    assert snap["stages"][0]["state"] == "done"
    assert snap["overall"]["done"] == 2
    assert snap["state"] == "done"
    assert rec.snapshots, "sink received updates"


def test_already_done_seeds_resumed_progress():
    m = _mon(step="generate", limit=10, n=1)
    with m:
        m.start_stage("generate", total=10, already_done=6)
        assert m.snapshot()["stages"][0]["done"] == 6


def test_retry_and_error_counters_via_module_helpers():
    m = _mon()
    with m:
        assert monitor.ACTIVE is m
        monitor.note_retry("openrouter:x")
        monitor.note_error("generate m1 p1: boom")
    snap = m.snapshot()
    assert snap["retries"] == 1 and snap["errors"] == 1
    assert monitor.ACTIVE is None            # cleared on exit


def test_eta_none_before_data_then_exact():
    m = RunMonitor(WorkPlan.for_step("all", 1, 1), sinks=[])
    with m:
        m.start_stage("generate", total=1)
        assert m.snapshot()["eta_s"] is None            # no duration recorded yet
        m.plan.get("generate").record_duration(10.0)    # 10s/item observed
        # remaining items across all not-done stages, all at the 10s fallback rate:
        # connectivity2 + load1 + generate1 + grade1 + classify1 + diagnose1 + validate1
        # + aggregate1 = 9 -> 90.0
        assert m.snapshot()["eta_s"] == 90.0


def test_error_state_and_active_cleared_on_exception():
    rec = RecordingSink()
    m = RunMonitor(WorkPlan.for_step("all", 2, 1), sinks=[rec])
    with pytest.raises(RuntimeError):
        with m:
            raise RuntimeError("boom")
    assert m.snapshot()["state"] == "error"
    assert monitor.ACTIVE is None
    assert rec.closes >= 1


def test_note_logs_to_sink():
    rec = RecordingSink()
    m = RunMonitor(WorkPlan.for_step("all", 2, 1), sinks=[rec])
    with m:
        m.note("hello")
    assert ("NOTE", "hello") in rec.logs


def test_multi_sink_fan_out():
    rec1, rec2 = RecordingSink(), RecordingSink()
    m = RunMonitor(WorkPlan.for_step("generate", limit=2, n_candidates=1),
                   sinks=[rec1, rec2])
    with m:
        m.start_stage("generate", total=2)
        m.item_start(model="m1", prompt_id="p1")
        m.item_done()
    assert rec1.snapshots and rec2.snapshots


def test_render_lines_contains_key_fields():
    snap = {
        "experiment": "E02-v1-75p", "state": "running",
        "elapsed_s": 4340, "eta_s": 10800,
        "current": {"stage": "generate", "in_flight": 2,
                    "in_flight_items": [
                        {"model": "qwen-397b", "prompt_id": "prompt-50"},
                        {"model": "qwen-9b", "prompt_id": "prompt-51"},
                    ]},
        "overall": {"done": 227, "total": 606, "pct": 37.5},
        "stages": [
            {"name": "load", "state": "done", "done": 75, "total": 75, "pct": 100.0},
            {"name": "generate", "state": "running", "done": 148, "total": 225, "pct": 65.8},
            {"name": "grade", "state": "pending", "done": 0, "total": 225, "pct": 0.0},
        ],
        "retries": 3, "errors": 0,
    }
    out = render_lines(snap)
    assert "E02-v1-75p" in out
    assert "148/225" in out
    assert "65%" in out                # int() truncation of 65.8
    assert "37%" in out                # overall
    assert "2 in flight" in out
    assert "qwen-397b" in out and "prompt-50" in out
    assert "retries: 3" in out and "errors: 0" in out
    assert "✓ done" in out
    assert "pending" in out


def test_render_lines_tolerates_old_single_item_shape():
    """`status` must still render a progress.json written by pre-concurrency
    code (a live run's heartbeat survives the upgrade)."""
    snap = {
        "experiment": "E05-reasoning-rand20p", "state": "running",
        "elapsed_s": 100, "eta_s": None,
        "current": {"stage": "generate", "model": "qwen-397b", "prompt_id": "CIF-050"},
        "overall": {"done": 1, "total": 2, "pct": 50.0},
        "stages": [
            {"name": "generate", "state": "running", "done": 1, "total": 2, "pct": 50.0},
        ],
        "retries": 0, "errors": 0,
    }
    out = render_lines(snap)
    assert "qwen-397b" in out and "CIF-050" in out
    assert "1 in flight" in out


def test_render_lines_shows_batch_counts():
    snap = {
        "experiment": "E99-test", "state": "running",
        "elapsed_s": 10, "eta_s": None,
        "current": {"stage": "grade", "in_flight": 0, "in_flight_items": []},
        "overall": {"done": 0, "total": 6, "pct": 0.0},
        "stages": [
            {"name": "grade", "state": "running", "done": 0, "total": 6, "pct": 0.0},
        ],
        "batch": {"submitted": 6, "pending": 4, "collected": 2, "errored": 1},
        "retries": 0, "errors": 0,
    }
    out = render_lines(snap)
    assert "submitted 6" in out and "pending 4" in out
    assert "collected 2" in out and "errored 1" in out
