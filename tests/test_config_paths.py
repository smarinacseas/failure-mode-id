from pathlib import Path

import config


def test_new_monitoring_paths_exist():
    assert isinstance(config.LOGS_DIR, Path)
    assert isinstance(config.PROGRESS_PATH, Path)
    assert config.LOGS_DIR == config.OUTPUTS_DIR / "logs"
    assert config.PROGRESS_PATH == config.OUTPUTS_DIR / "progress.json"


def test_rich_and_pytest_available():
    import rich  # noqa: F401


def test_runs_dir_and_no_dead_constants():
    assert config.RUNS_DIR == config.ROOT / "runs"
    for dead in ("RESPONSES_DIR", "GRADES_DIR", "CRITERIA_TAGS_PATH",
                 "JUDGE_VALIDATION_PATH", "RUN_MANIFEST_PATH", "CANDIDATE_EXTRA_BODY"):
        assert not hasattr(config, dead), f"config.{dead} should be removed"
