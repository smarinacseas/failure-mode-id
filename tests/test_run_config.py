import json

import pytest

import config
from pipeline.run_config import (
    ConfigConflictError, InvalidSlugError, RunConfig, parse_candidates,
    parse_slug, resolve, validate_judge,
)


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


def test_parse_candidates_registry_keys_and_pairs(monkeypatch):
    monkeypatch.setattr(config, "CANDIDATES", {"qwen-9b": "qwen/qwen3.5-9b"})
    assert parse_candidates("qwen-9b") == {"qwen-9b": "qwen/qwen3.5-9b"}
    assert parse_candidates("qwen-9b,deepseek=deepseek/deepseek-v4") == {
        "qwen-9b": "qwen/qwen3.5-9b",
        "deepseek": "deepseek/deepseek-v4",
    }


def test_parse_candidates_unknown_key_lists_registry(monkeypatch):
    monkeypatch.setattr(config, "CANDIDATES", {"qwen-9b": "qwen/qwen3.5-9b"})
    with pytest.raises(ValueError, match="qwen-9b"):
        parse_candidates("nope")


def test_validate_judge_prefix():
    assert validate_judge("claude-fable-5") == "claude-fable-5"
    with pytest.raises(ValueError, match="claude-"):
        validate_judge("gpt-5")


def test_resolve_freezes_on_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    cfg = resolve("E10-frozen", {"max_tokens": 1234, "limit": 5})
    assert cfg.max_tokens == 1234 and cfg.limit == 5
    assert cfg.judge == config.JUDGE                     # default filled in
    frozen = json.loads((tmp_path / "E10-frozen" / "experiment.json").read_text())
    assert frozen["params"]["max_tokens"] == 1234
    assert "created_at" in frozen


def test_resolve_reloads_frozen_params(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    resolve("E11-reload", {"reasoning": True})
    cfg2 = resolve("E11-reload", {})                     # bare re-invocation
    assert cfg2.reasoning is True


def test_resolve_conflicting_flag_errors_with_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    resolve("E12-conflict", {"max_tokens": 8000})
    with pytest.raises(ConfigConflictError, match="max_tokens"):
        resolve("E12-conflict", {"max_tokens": 4000})


def test_resolve_identical_repass_is_fine(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    resolve("E13-same", {"max_tokens": 8000})
    cfg = resolve("E13-same", {"max_tokens": 8000})      # same value → no error
    assert cfg.max_tokens == 8000


def test_resolve_rejects_bad_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    with pytest.raises(InvalidSlugError):
        resolve("not-a-slug", {})


def test_from_json_dict_missing_field_raises_readable_value_error():
    d = _cfg().to_json_dict()
    del d["judge"]
    with pytest.raises(ValueError, match=r"missing field 'judge'.*hand-edited.*runs/E99-test/"):
        RunConfig.from_json_dict("E99-test", d)
