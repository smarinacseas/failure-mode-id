from pathlib import Path

import config


def test_new_monitoring_paths_exist():
    assert isinstance(config.LOGS_DIR, Path)
    assert isinstance(config.PROGRESS_PATH, Path)
    assert config.LOGS_DIR == config.OUTPUTS_DIR / "logs"
    assert config.PROGRESS_PATH == config.OUTPUTS_DIR / "progress.json"


def test_rich_and_pytest_available():
    import rich  # noqa: F401
