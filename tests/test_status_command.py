import json
from datetime import datetime, timedelta, timezone

import main


def _snap(state, updated):
    return {
        "experiment": "E02-v1-75p", "state": state, "pid": 4242,
        "started_at": "2026-07-02T14:30:00+00:00", "updated_at": updated,
        "elapsed_s": 4340, "eta_s": 10800,
        "current": {"stage": "generate", "model": "qwen-397b", "prompt_id": "prompt-50"},
        "overall": {"done": 227, "total": 606, "pct": 37.5},
        "stages": [{"name": "generate", "state": "running",
                    "done": 148, "total": 225, "pct": 65.8}],
        "retries": 3, "errors": 0,
    }


def test_status_renders_heartbeat(tmp_path, monkeypatch, capsys):
    path = tmp_path / "progress.json"
    now = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(_snap("running", now)))
    monkeypatch.setattr(main, "PROGRESS_PATH", path)
    main._print_status()
    out = capsys.readouterr().out
    assert "E02-v1-75p" in out and "148/225" in out
    assert "stalled" not in out                       # fresh heartbeat


def test_status_flags_stale_running_run(tmp_path, monkeypatch, capsys):
    path = tmp_path / "progress.json"
    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    path.write_text(json.dumps(_snap("running", old)))
    monkeypatch.setattr(main, "PROGRESS_PATH", path)
    main._print_status()
    assert "possibly stalled" in capsys.readouterr().out


def test_status_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main, "PROGRESS_PATH", tmp_path / "nope.json")
    main._print_status()
    assert "no run found" in capsys.readouterr().out
