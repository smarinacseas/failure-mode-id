import pytest

import config
from pipeline.run_config import (
    JudgeSpec, candidate_family, family_of, parse_judges, resolve_judge,
)


def test_registry_key_resolves():
    s = resolve_judge("gpt-5")
    assert s == JudgeSpec(key="gpt-5", client="openrouter",
                          model=config.JUDGE_REGISTRY["gpt-5"]["model"])


def test_bare_claude_id_resolves_anthropic():
    s = resolve_judge("claude-opus-4-8")
    assert s.client == "anthropic" and s.key == s.model == "claude-opus-4-8"


def test_key_equals_syntax_infers_client():
    s = resolve_judge("kimi=moonshotai/kimi-k3")
    assert s == JudgeSpec(key="kimi", client="openrouter", model="moonshotai/kimi-k3")
    s2 = resolve_judge("opus=claude-opus-4-8")
    assert s2.client == "anthropic"


def test_bare_unknown_entry_errors_with_pointer():
    with pytest.raises(ValueError, match="key=provider/model-id"):
        resolve_judge("gpt-99-unknown")


def test_uninferrable_client_errors():
    with pytest.raises(ValueError, match="cannot infer client"):
        resolve_judge("weird=no-slash-no-claude")


def test_key_must_be_path_safe():
    with pytest.raises(ValueError, match="judge key"):
        JudgeSpec(key="a/b", client="openrouter", model="x/y")
    with pytest.raises(ValueError, match="judge key"):
        JudgeSpec(key="a__b", client="anthropic", model="claude-x")


def test_parse_judges_dedup_and_conflict():
    js = parse_judges("claude-opus-4-8, gpt-5, claude-opus-4-8")
    assert [s.key for s in js] == ["claude-opus-4-8", "gpt-5"]
    with pytest.raises(ValueError, match="different models"):
        parse_judges("gpt-5,gpt-5=openai/gpt-other")
    with pytest.raises(ValueError, match="empty set"):
        parse_judges(" , ")


def test_from_value_hydration():
    assert JudgeSpec.from_value("claude-fable-5") == JudgeSpec(
        key="claude-fable-5", client="anthropic", model="claude-fable-5")
    d = {"key": "gpt-5", "client": "openrouter", "model": "openai/gpt-5.2"}
    assert JudgeSpec.from_value(d) == JudgeSpec(**d)


def test_families():
    assert family_of("anthropic", "claude-fable-5") == "anthropic"
    assert family_of("openrouter", "qwen/qwen3.5-35b-a3b") == "qwen"
    assert candidate_family("qwen/qwen3.5-9b") == "qwen"


def test_default_panel_is_five_families():
    assert list(config.JUDGES) == list(config.JUDGE_REGISTRY)
    assert len(config.JUDGES) % 2 == 1  # odd panel: majority cannot tie at full attendance
