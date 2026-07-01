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


class _NoopMonitor:
    def __init__(self, step): self.step = step
    def __enter__(self): return self
    def __exit__(self, *a): return False
