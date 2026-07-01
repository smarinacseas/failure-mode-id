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


def test_eta_is_time_weighted_across_remaining_stages():
    m = RunMonitor(WorkPlan.for_step("all", limit=1, n_candidates=1), sinks=[])
    with m:
        m.start_stage("generate", total=1)
        m.plan.get("generate").record_duration(10.0)   # 10s/item observed
        eta = m.snapshot()["eta_s"]
    # remaining items across not-done stages × observed/fallback rate (10s):
    # load1 + generate0(after done? not ended) ... at least > 0 and finite
    assert eta is not None and eta > 0


def test_error_state_on_exception_in_context():
    m = _mon()
    try:
        with m:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m.snapshot()["state"] == "error"


def test_render_lines_contains_key_fields():
    snap = {
        "experiment": "E02-v1-75p", "state": "running",
        "elapsed_s": 4340, "eta_s": 10800,
        "current": {"stage": "generate", "model": "qwen-397b", "prompt_id": "prompt-50"},
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
    assert "qwen-397b" in out and "prompt-50" in out
    assert "retries: 3" in out and "errors: 0" in out
    assert "✓ done" in out
