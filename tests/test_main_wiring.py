import json

import main


def test_all_requires_limit(capsys):
    assert main.main(["all"]) == 2
    assert "requires --limit" in capsys.readouterr().err


def test_single_step_builds_monitor_and_passes_it(monkeypatch):
    captured = {}

    def fake_grade_run(limit=None, monitor=None):
        captured["limit"] = limit
        captured["has_monitor"] = monitor is not None
    monkeypatch.setattr(main.grade, "run", fake_grade_run)
    # keep the console quiet in tests: no-op sinks
    monkeypatch.setattr(main, "build_monitor",
                        lambda step, limit, experiment=None: _NoopMonitor(step))

    assert main.main(["grade", "--limit", "5"]) == 0
    assert captured == {"limit": 5, "has_monitor": True}


def test_status_survives_naive_updated_at(tmp_path, monkeypatch, capsys):
    snap = {
        "experiment": "E", "state": "running", "pid": 1,
        "started_at": "x", "updated_at": "2026-07-01T10:00:00",  # naive (no tz) -> TypeError on subtract
        "elapsed_s": 1, "eta_s": None,
        "current": {"stage": None, "model": None, "prompt_id": None},
        "overall": {"done": 0, "total": 1, "pct": 0.0},
        "stages": [{"name": "load", "state": "running", "done": 0, "total": 1, "pct": 0.0}],
        "retries": 0, "errors": 0,
    }
    (tmp_path / "progress.json").write_text(json.dumps(snap))
    monkeypatch.setattr(main, "PROGRESS_PATH", tmp_path / "progress.json")
    main._print_status()                    # must not raise
    assert "E" in capsys.readouterr().out   # rendered fine, no stale warning, no crash


class _NoopMonitor:
    def __init__(self, step): self.step = step
    def __enter__(self): return self
    def __exit__(self, *a): return False
