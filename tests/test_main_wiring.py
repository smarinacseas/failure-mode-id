import json

import config
import main


def _noop_monitor_factory():
    return lambda step, limit, experiment=None, n_candidates=None: _NoopMonitor(step)


def test_data_step_requires_experiment(capsys):
    assert main.main(["generate"]) == 2
    assert "--experiment" in capsys.readouterr().err


def test_all_requires_limit_for_new_experiment(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    assert main.main(["all", "--experiment", "E90-x"]) == 2
    assert "--limit" in capsys.readouterr().err


def test_conflicting_flag_exits_2_with_diff(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())
    monkeypatch.setattr(main.grade, "run", lambda cfg, monitor=None: None)
    assert main.main(["grade", "--experiment", "E91-y", "--max-tokens", "8000"]) == 0
    assert main.main(["grade", "--experiment", "E91-y", "--max-tokens", "4000"]) == 2
    assert "max_tokens" in capsys.readouterr().err


def test_single_step_gets_cfg_and_monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    captured = {}

    def fake_grade_run(cfg, monitor=None):
        captured["slug"] = cfg.slug
        captured["limit"] = cfg.limit
        captured["has_monitor"] = monitor is not None
    monkeypatch.setattr(main.grade, "run", fake_grade_run)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())

    assert main.main(["grade", "--experiment", "E92-z", "--limit", "5"]) == 0
    assert captured == {"slug": "E92-z", "limit": 5, "has_monitor": True}


def test_bad_judge_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    assert main.main(["grade", "--experiment", "E93-j", "--judge", "gpt-5"]) == 2
    assert "claude-" in capsys.readouterr().err


def test_status_survives_naive_updated_at(tmp_path, monkeypatch, capsys):
    snap = {
        "experiment": "E", "state": "running", "pid": 1,
        "started_at": "x", "updated_at": "2026-07-01T10:00:00",
        "elapsed_s": 1, "eta_s": None,
        "current": {"stage": None, "model": None, "prompt_id": None},
        "overall": {"done": 0, "total": 1, "pct": 0.0},
        "stages": [{"name": "load", "state": "running", "done": 0, "total": 1, "pct": 0.0}],
        "retries": 0, "errors": 0,
    }
    (tmp_path / "progress.json").write_text(json.dumps(snap))
    monkeypatch.setattr(main, "PROGRESS_PATH", tmp_path / "progress.json")
    main._print_status()
    assert "E" in capsys.readouterr().out


class _NoopMonitor:
    def __init__(self, step): self.step = step
    def __enter__(self): return self
    def __exit__(self, *a): return False
