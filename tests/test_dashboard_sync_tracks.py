import json
from pathlib import Path
import scripts.dashboard_sync as ds


def _fake_experiment(src: Path, slug: str):
    payload = {"meta": {"experiment": {"slug": slug, "run_date": "2026-07-12T00:00:00+00:00",
                                       "description": f"{slug} desc."},
                        "counts": {"n_prompts": 2, "n_models": 2}}}
    (src / f"{slug}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_sync_partitions_by_track(tmp_path):
    src = tmp_path / "experiments"; src.mkdir()
    dst = tmp_path / "dashboard"; dst.mkdir()
    _fake_experiment(src, "E07-reasoning-full75")
    _fake_experiment(src, "T01-ihrlvr")

    ds.sync(src, dst)

    runs = json.loads((dst / "runs.json").read_text())
    training = json.loads((dst / "training.json").read_text())
    assert [r["id"] for r in runs["runs"]] == ["E07-reasoning-full75"]
    assert [r["id"] for r in training["runs"]] == ["T01-ihrlvr"]


def test_sync_preserves_indices_and_reference_from_pruning(tmp_path):
    src = tmp_path / "experiments"; src.mkdir()
    dst = tmp_path / "dashboard"; dst.mkdir()
    (dst / "runs.json").write_text("{}", encoding="utf-8")
    (dst / "training.json").write_text("{}", encoding="utf-8")
    (dst / "reference.json").write_text("{}", encoding="utf-8")
    _fake_experiment(src, "E01-smoke-3p")

    ds.sync(src, dst)  # must not delete tracked files while pruning stale files

    assert (dst / "training.json").exists()
    assert (dst / "reference.json").exists()
