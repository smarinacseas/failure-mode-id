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
    # Erroring out before doing anything must not freeze the slug's params —
    # otherwise a retry with --limit hits a manufactured ConfigConflictError.
    assert not (tmp_path / "E90-x" / "experiment.json").exists()

    captured = {}

    def fake_run_all(cfg, run_report, diagnose_enabled=True):
        captured["cfg"] = cfg
    monkeypatch.setattr(main, "_run_all", fake_run_all)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())

    assert main.main(["all", "--experiment", "E90-x", "--limit", "3"]) == 0
    assert captured["cfg"].limit == 3


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


def test_sample_seed_flag_flows_to_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    captured = {}

    def fake_grade_run(cfg, monitor=None):
        captured["sample_seed"] = cfg.sample_seed
    monkeypatch.setattr(main.grade, "run", fake_grade_run)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())

    assert main.main(["grade", "--experiment", "E94-s", "--limit", "5",
                      "--sample-seed", "42"]) == 0
    assert captured["sample_seed"] == 42


def test_judge_mode_flag_flows_to_cfg_and_defaults_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    captured = {}

    def fake_grade_run(cfg, monitor=None):
        captured["judge_mode"] = cfg.judge_mode
    monkeypatch.setattr(main.grade, "run", fake_grade_run)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())

    assert main.main(["grade", "--experiment", "E98-seq", "--limit", "5",
                      "--judge-mode", "sequential"]) == 0
    assert captured["judge_mode"] == "sequential"

    assert main.main(["grade", "--experiment", "E98-batch", "--limit", "5"]) == 0
    assert captured["judge_mode"] == "batch"        # frozen default

    # judge_mode is frozen like every other param: a conflicting re-pass errors.
    assert main.main(["grade", "--experiment", "E98-seq",
                      "--judge-mode", "batch"]) == 2


def test_bad_judge_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    assert main.main(["grade", "--experiment", "E93-j", "--judge", "gpt-5"]) == 2
    assert "claude-" in capsys.readouterr().err


def test_all_on_slug_frozen_without_limit_gives_distinct_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())
    # Freeze E97-nolimit via a non-`all` step, without ever passing --limit.
    monkeypatch.setattr(main.grade, "run", lambda cfg, monitor=None: None)
    assert main.main(["grade", "--experiment", "E97-nolimit"]) == 0

    # `all` on that already-frozen, limit-less slug must NOT hit the
    # pre-resolve "new run" message — it hits the post-resolve one instead.
    assert main.main(["all", "--experiment", "E97-nolimit"]) == 2
    err = capsys.readouterr().err
    assert "was frozen without a limit" in err
    # The pre-resolve ("new run") wording is distinct — must not appear here.
    assert "a new `all` run requires --limit" not in err


def test_connectivity_bare_pings_defaults_and_freezes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())
    captured = {}

    def fake_connectivity_run(cfg, monitor=None):
        captured["cfg"] = cfg
    monkeypatch.setattr(main.connectivity, "run", fake_connectivity_run)

    assert main.main(["connectivity"]) == 0
    assert captured["cfg"] is None
    assert list(tmp_path.iterdir()) == []


def test_connectivity_unfrozen_slug_errors_without_freezing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    assert main.main(["connectivity", "--experiment", "E95-new"]) == 2
    assert not (tmp_path / "E95-new" / "experiment.json").exists()


def test_connectivity_frozen_slug_pings_frozen_models(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())
    monkeypatch.setattr(main.grade, "run", lambda cfg, monitor=None: None)
    assert main.main(["grade", "--experiment", "E96-frozen"]) == 0

    captured = {}

    def fake_connectivity_run(cfg, monitor=None):
        captured["cfg"] = cfg
    monkeypatch.setattr(main.connectivity, "run", fake_connectivity_run)

    assert main.main(["connectivity", "--experiment", "E96-frozen"]) == 0
    assert captured["cfg"].slug == "E96-frozen"


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


def test_diagnose_step_gets_cfg_and_monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    captured = {}

    def fake_diagnose_run(cfg, monitor=None):
        captured["slug"] = cfg.slug
        captured["has_monitor"] = monitor is not None
    monkeypatch.setattr(main.diagnose, "run", fake_diagnose_run)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())

    assert main.main(["diagnose", "--experiment", "E88-d", "--limit", "5"]) == 0
    assert captured == {"slug": "E88-d", "has_monitor": True}


def test_diagnose_toggle_is_not_frozen_and_gates_all(tmp_path, monkeypatch):
    """--diagnose is post-hoc analysis config (spec §2.7): toggling it across
    invocations must never raise ConfigConflictError, and `all --diagnose off`
    must skip the stage while `all` (default on) runs it."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "build_monitor", _noop_monitor_factory())
    ran = []
    for step in ("connectivity", "generate", "grade", "classify", "validate", "aggregate"):
        mod = getattr(main, step)
        monkeypatch.setattr(mod, "run",
                            lambda *a, _s=step, **k: ran.append(_s))
    monkeypatch.setattr(main.load, "run", lambda *a, **k: ran.append("load"))
    monkeypatch.setattr(main.diagnose, "run", lambda *a, **k: ran.append("diagnose"))

    assert main.main(["all", "--experiment", "E87-t", "--limit", "2",
                      "--diagnose", "off"]) == 0
    assert "diagnose" not in ran

    ran.clear()
    # Same slug, opposite toggle: must NOT conflict, and must run the stage
    # between classify and validate.
    assert main.main(["all", "--experiment", "E87-t"]) == 0
    assert ran.index("classify") < ran.index("diagnose") < ran.index("validate")


class _NoopMonitor:
    def __init__(self, step): self.step = step
    def __enter__(self): return self
    def __exit__(self, *a): return False
