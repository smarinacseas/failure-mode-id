import pipeline.monitor as monitor
from pipeline._io import retry
from pipeline.monitor import RunMonitor, WorkPlan


def test_retry_reports_to_active_monitor():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 rate limited")
        return "ok"

    m = RunMonitor(WorkPlan.for_step("generate", 1, 1), sinks=[])
    with m:
        assert retry(flaky, label="test", base_delay=0.0) == "ok"
    assert m.snapshot()["retries"] == 2          # two backoffs before success


def test_retry_without_active_monitor_is_safe():
    monitor.ACTIVE = None
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("429 rate limited")
        return "ok"

    # Raises once -> retry() enters its except block and calls monitor.note_retry
    # while ACTIVE is None; must not crash and must still return "ok".
    assert retry(flaky, label="test", base_delay=0.0) == "ok"
