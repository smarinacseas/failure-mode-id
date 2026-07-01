import io

from rich.console import Console

from pipeline.monitor import (
    ConsoleSink, RunMonitor, WorkPlan, build_monitor, default_sinks, stage_ctx,
)


def test_console_sink_runs_without_error_and_prints_notices():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    m = RunMonitor(WorkPlan.for_step("grade", 2, 1), experiment="E99-test",
                   sinks=[ConsoleSink(console=console)])
    with m:
        m.start_stage("grade", total=2)
        m.item_start(model="m1", prompt_id="p1")
        m.item_done()
        m.record_error("grade m1 p2: boom")        # ERROR notice -> printed
        m.note("heads up")                          # NOTE notice -> printed
    out = buf.getvalue()
    assert "boom" in out
    assert "heads up" in out
    assert "INFO" not in out


def test_default_sinks_shape(tmp_path, monkeypatch):
    import pipeline.monitor as monitor
    monkeypatch.setattr(monitor, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(monitor, "PROGRESS_PATH", tmp_path / "progress.json")
    sinks = default_sinks("E02-v1-75p")
    kinds = [type(s).__name__ for s in sinks]
    assert kinds == ["ConsoleSink", "LogSink", "StatusSink"]


def test_build_monitor_uses_workplan_and_candidate_count(monkeypatch):
    import pipeline.monitor as monitor
    monkeypatch.setattr(monitor, "CANDIDATES", {"a": "x", "b": "y"})
    m = build_monitor("generate", limit=4, experiment=None, sinks=[])
    assert m.plan.get("generate").total == 8      # 4 * 2 candidates


def test_stage_ctx_reuses_passed_monitor():
    m = RunMonitor(WorkPlan.for_step("grade", 1, 1), sinks=[])
    with stage_ctx(m, "grade", 1) as got:
        assert got is m


def test_stage_ctx_builds_when_no_monitor(monkeypatch):
    import pipeline.monitor as monitor
    # Stub default_sinks so no real ConsoleSink/LogSink/StatusSink (and no log file) is created.
    monkeypatch.setattr(monitor, "default_sinks", lambda experiment: [])
    ctx = stage_ctx(None, "grade", 2)
    assert isinstance(ctx, RunMonitor)
    assert ctx.plan.get("grade").total == 2 * len(monitor.CANDIDATES)
