import json

import pytest

import config
from pipeline.run_config import (
    ConfigConflictError, InvalidSlugError, JudgeSpec, RunConfig, family_overlaps,
    parse_candidates, parse_judges, parse_slug, resolve, resolve_judge,
)


def _cfg(**kw):
    params = dict(
        slug="E99-test",
        candidates={"m1": "prov/model-1"},
        judges=("claude-fable-5",),
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
    assert cfg.grades_path("claude-fable-5", "m1") == (
        tmp_path / "E99-test" / "grades" / "claude-fable-5" / "m1.jsonl"
    )
    assert cfg.judge.key == "claude-fable-5"  # first judge is canonical
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
    assert config.JUDGES == list(config.JUDGE_REGISTRY)
    assert config.JUDGE == config.JUDGES[0]


def test_parse_judges_dedups():
    js = parse_judges("claude-opus-4-8, claude-fable-5, claude-opus-4-8")
    assert [s.key for s in js] == ["claude-opus-4-8", "claude-fable-5"]


def test_judges_default_filled_and_json_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    cfg = resolve("E14-judges", {"limit": 3})
    default = tuple(resolve_judge(k) for k in config.JUDGES)
    assert cfg.judges == default            # default panel resolved to specs
    # re-passing the same judges must NOT trip the frozen-conflict guard
    cfg2 = resolve("E14-judges", {"judges": default})
    assert cfg2.judges == default


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


def test_resolve_freezes_on_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    cfg = resolve("E10-frozen", {"max_tokens": 1234, "limit": 5})
    assert cfg.max_tokens == 1234 and cfg.limit == 5
    assert cfg.judges == tuple(resolve_judge(k) for k in config.JUDGES)  # default filled in
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


def test_sample_seed_freezes_reloads_and_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    cfg = resolve("E15-sampled", {"limit": 20, "sample_seed": 20260706})
    assert cfg.sample_seed == 20260706
    frozen = json.loads((tmp_path / "E15-sampled" / "experiment.json").read_text())
    assert frozen["params"]["sample_seed"] == 20260706
    # bare re-invocation reloads the frozen seed
    assert resolve("E15-sampled", {}).sample_seed == 20260706
    # a different seed would silently select different prompts — must conflict
    with pytest.raises(ConfigConflictError, match="sample_seed"):
        resolve("E15-sampled", {"sample_seed": 1})


def test_config_block_records_sampling_knobs():
    """meta.config is 'every knob that could differ across experiments' —
    the prompt subset (limit + sample_seed) is exactly such a knob."""
    from pipeline._experiment import config_block
    block = config_block(_cfg(limit=20, sample_seed=20260706))
    assert block["limit"] == 20
    assert block["sample_seed"] == 20260706


def test_sample_seed_backcompat_defaults_none():
    """Freezes older than the sample_seed knob (E01–E04) must load as None."""
    d = _cfg().to_json_dict()
    d.pop("sample_seed", None)
    assert RunConfig.from_json_dict("E99-test", d).sample_seed is None


def test_from_json_dict_missing_field_raises_readable_value_error():
    d = _cfg().to_json_dict()
    del d["judges"]
    with pytest.raises(ValueError, match=r"missing field 'judges'.*hand-edited.*runs/E99-test/"):
        RunConfig.from_json_dict("E99-test", d)


def test_judges_hydrate_from_strings_and_dicts():
    cfg = _cfg(judges=("claude-fable-5",
                       {"key": "gpt-5", "client": "openrouter", "model": "openai/gpt-5.2"}))
    assert cfg.judges[0] == JudgeSpec("claude-fable-5", "anthropic", "claude-fable-5")
    assert cfg.judges[1].client == "openrouter"
    assert cfg.judge == cfg.judges[0]
    assert cfg.judge_keys == ("claude-fable-5", "gpt-5")


def test_classifier_chain_defaults_to_first_judge():
    cfg = _cfg(judges=("claude-fable-5", "claude-opus-4-8"))
    assert cfg.classifier_chain == (cfg.judges[0],)
    cfg2 = _cfg(classifier_chain=("claude-opus-4-8",))
    assert cfg2.classifier_chain[0].key == "claude-opus-4-8"


def test_freeze_round_trip_with_specs():
    cfg = _cfg(judges=("claude-fable-5",
                       {"key": "gpt-5", "client": "openrouter", "model": "openai/gpt-5.2"}))
    d = json.loads(json.dumps(cfg.to_json_dict()))
    assert d["judges"][1] == {"key": "gpt-5", "client": "openrouter", "model": "openai/gpt-5.2"}
    assert RunConfig.from_json_dict("E99-test", d) == cfg


def test_old_freeze_hydrates_byte_identical():
    # pre-panel freeze: judges as plain strings, no classifier_chain
    d = _cfg(judges=("claude-opus-4-8", "claude-fable-5")).to_json_dict()
    d["judges"] = ["claude-opus-4-8", "claude-fable-5"]
    del d["classifier_chain"]
    cfg = RunConfig.from_json_dict("E99-test", d)
    assert all(s.client == "anthropic" and s.key == s.model for s in cfg.judges)
    assert cfg.classifier_chain == (cfg.judges[0],)
    # pre-multi-judge freeze: scalar judge
    d2 = dict(d); del d2["judges"]; d2["judge"] = "claude-fable-5"
    assert RunConfig.from_json_dict("E99-test", d2).judge.key == "claude-fable-5"


def test_refreezing_same_judges_not_a_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    judges = parse_judges("claude-fable-5,gpt-5")
    resolve("E97-panel", {"limit": 2, "judges": judges})
    cfg = resolve("E97-panel", {"judges": judges})   # must NOT raise
    assert cfg.judges == judges
    with pytest.raises(ConfigConflictError):
        resolve("E97-panel", {"judges": parse_judges("claude-fable-5")})


def test_family_overlap_detection():
    cfg = _cfg(candidates={"qwen-9b": "qwen/qwen3.5-9b"},
               judges=("claude-fable-5", {"key": "qwen-judge", "client": "openrouter",
                                          "model": "qwen/qwen3.5-397b-a17b"}))
    assert family_overlaps(cfg) == [("qwen-judge", "qwen")]
