import types

import pytest

import config
from pipeline import _judge_llm
from pipeline._judge_llm import JudgeDeadlineExceeded, call_json, call_json_chain
from pipeline.run_config import JudgeSpec

OR_SPEC = JudgeSpec(key="gpt-5", client="openrouter", model="openai/gpt-5.2")
ANTH_SPEC = JudgeSpec(key="claude-fable-5", client="anthropic", model="claude-fable-5")


def _chunk(content=None, finish=None):
    delta = types.SimpleNamespace(content=content, model_extra=None)
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish)
    return types.SimpleNamespace(choices=[choice])


class FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False
        self.request_kwargs = None
    def __iter__(self):
        return iter(self._chunks)
    def close(self):
        self.closed = True


def _patch_router(monkeypatch, stream):
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return stream
    monkeypatch.setattr(config.router.chat.completions, "create", create)
    return captured


def test_openrouter_branch_streams_and_normalizes_stop(monkeypatch):
    captured = _patch_router(monkeypatch, FakeStream([_chunk("["), _chunk("]"), _chunk(finish="stop")]))
    text, stop = call_json(OR_SPEC, "SYS", "USER", label="openrouter:gpt-5")
    assert (text, stop) == ("[]", "stop")
    assert captured["model"] == "openai/gpt-5.2"
    assert captured["stream"] is True
    assert captured["extra_body"] == {"reasoning": {"enabled": True}}
    assert captured["max_tokens"] == config.JUDGE_MAX_TOKENS
    assert captured["messages"][0] == {"role": "system", "content": "SYS"}


def test_finish_length_maps_to_max_tokens(monkeypatch):
    _patch_router(monkeypatch, FakeStream([_chunk("x"), _chunk(finish="length")]))
    _, stop = call_json(OR_SPEC, "s", "u", label="l")
    assert stop == "max_tokens"


def test_content_filter_and_empty_map_to_refusal(monkeypatch):
    _patch_router(monkeypatch, FakeStream([_chunk("x"), _chunk(finish="content_filter")]))
    assert call_json(OR_SPEC, "s", "u", label="l")[1] == "refusal"
    _patch_router(monkeypatch, FakeStream([_chunk(finish="stop")]))
    assert call_json(OR_SPEC, "s", "u", label="l")[1] == "refusal"


def test_deadline_abort_raises_retriable(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_DEADLINE_S", 0.0)
    _patch_router(monkeypatch, FakeStream([_chunk("a"), _chunk("b")]))
    with pytest.raises(JudgeDeadlineExceeded):
        # attempts=1 via retry: patch retry to call through once
        monkeypatch.setattr(_judge_llm, "retry", lambda fn, **kw: fn())
        call_json(OR_SPEC, "s", "u", label="l")


def test_anthropic_branch_unchanged(monkeypatch):
    calls = {}
    class FakeMsg:
        content = [types.SimpleNamespace(type="text", text="[1]")]
        stop_reason = "end_turn"
    class FakeStreamCtx:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get_final_message(self):
            return FakeMsg()
    def stream(**kwargs):
        calls.update(kwargs)
        return FakeStreamCtx()
    monkeypatch.setattr(config.anthropic.messages, "stream", stream)
    text, stop = call_json(ANTH_SPEC, "SYS", "USER", label="anthropic:claude-fable-5")
    assert (text, stop) == ("[1]", "end_turn")
    assert calls["model"] == "claude-fable-5"
    assert calls["thinking"] == {"type": "adaptive"}
    # plain-string caller hydrates to the same anthropic path
    text2, _ = call_json("claude-fable-5", "SYS", "USER", label="x")
    assert text2 == "[1]"


def test_chain_refusal_advances_without_retry(monkeypatch):
    seen = []
    def fake_call(spec, system, user, label):
        seen.append(spec.key)
        if spec.key == "claude-fable-5":
            return "", "refusal"
        return "ok", "stop"
    monkeypatch.setattr(_judge_llm, "call_json", fake_call)
    result, spec, err = call_json_chain(
        (ANTH_SPEC, OR_SPEC), "s", "u", "test",
        parse=lambda raw, stop: (None, "refused") if stop == "refusal" else (raw, ""))
    assert result == "ok" and spec.key == "gpt-5"
    assert seen == ["claude-fable-5", "gpt-5"]  # exactly ONE attempt on the refuser


def test_chain_parse_error_retries_once_then_advances(monkeypatch):
    seen = []
    def fake_call(spec, system, user, label):
        seen.append(spec.key)
        return "garbage" if spec.key == "claude-fable-5" else "ok", "stop"
    monkeypatch.setattr(_judge_llm, "call_json", fake_call)
    result, spec, _ = call_json_chain(
        (ANTH_SPEC, OR_SPEC), "s", "u", "test",
        parse=lambda raw, stop: (raw, "") if raw == "ok" else (None, "parse"))
    assert result == "ok" and spec.key == "gpt-5"
    assert seen == ["claude-fable-5", "claude-fable-5", "gpt-5"]


def test_chain_exhaustion_returns_none_with_last_err(monkeypatch):
    monkeypatch.setattr(_judge_llm, "call_json",
                        lambda spec, s, u, label: (_ for _ in ()).throw(RuntimeError("boom")))
    result, spec, err = call_json_chain((ANTH_SPEC,), "s", "u", "t",
                                        parse=lambda r, st: (r, ""))
    assert result is None and spec is None and "boom" in err
