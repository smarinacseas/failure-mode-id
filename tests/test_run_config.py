import json

import config
from pipeline.run_config import InvalidSlugError, RunConfig, parse_slug


def _cfg(**kw):
    params = dict(
        slug="E99-test",
        candidates={"m1": "prov/model-1"},
        judge="claude-fable-5",
        max_tokens=100,
        temperature=0.0,
        reasoning=False,
        timeout_s=5.0,
        limit=3,
        description="test run",
    )
    params.update(kw)
    return RunConfig(**params)


def test_parse_slug_valid_and_invalid():
    assert parse_slug("E01-smoke-3p") == (1, "smoke-3p")
    import pytest
    with pytest.raises(InvalidSlugError):
        parse_slug("smoke")


def test_paths_derive_from_runs_dir_and_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    cfg = _cfg()
    assert cfg.run_dir == tmp_path / "E99-test"
    assert cfg.experiment_json_path == tmp_path / "E99-test" / "experiment.json"
    assert cfg.responses_path("m1") == tmp_path / "E99-test" / "responses" / "m1.jsonl"
    assert cfg.grades_path("m1") == tmp_path / "E99-test" / "grades" / "m1.jsonl"
    assert cfg.criteria_tags_path == tmp_path / "E99-test" / "criteria_tags.jsonl"
    assert cfg.judge_validation_path == tmp_path / "E99-test" / "judge_validation.json"
    assert cfg.run_manifest_path == tmp_path / "E99-test" / "run_manifest.json"


def test_extra_body_reflects_reasoning():
    assert _cfg(reasoning=False).extra_body == {"reasoning": {"enabled": False}}
    assert _cfg(reasoning=True).extra_body == {"reasoning": {"enabled": True}}


def test_json_round_trip():
    cfg = _cfg(reasoning=True, limit=None)
    d = json.loads(json.dumps(cfg.to_json_dict()))
    assert RunConfig.from_json_dict("E99-test", d) == cfg


def test_runs_dir_default_and_judge_default():
    assert config.RUNS_DIR == config.ROOT / "runs"
    assert config.JUDGE == "claude-fable-5"
