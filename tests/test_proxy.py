import json

from fastapi.testclient import TestClient
from training.proxy import build_app


class _FakeSampler:
    def __init__(self, tag): self.tag = tag
    def generate(self, messages, max_tokens, temperature, reasoning):
        return f"{self.tag}:{messages[-1]['content']}"


class _RecordingSampler:
    """Records the `reasoning` kwarg it was called with, for Fix 2's test."""
    def __init__(self):
        self.last_reasoning = None

    def generate(self, messages, max_tokens, temperature, reasoning):
        self.last_reasoning = reasoning
        return "recorded"


def _client():
    app = build_app({"qwen3-8b-base": _FakeSampler("BASE"),
                     "qwen3-8b-ihrlvr": _FakeSampler("FT")}, api_key="local-dev")
    return TestClient(app)


def test_routes_by_model():
    c = _client()
    r = c.post("/v1/chat/completions",
               headers={"Authorization": "Bearer local-dev"},
               json={"model": "qwen3-8b-ihrlvr",
                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "FT:hi"


def test_rejects_bad_key():
    c = _client()
    r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer wrong"},
               json={"model": "qwen3-8b-base", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


def test_unknown_model_is_404():
    c = _client()
    r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer local-dev"},
               json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404


def test_nonstreaming_unchanged():
    """Non-streaming shape (used by connectivity.py's preflight ping) is
    untouched by the SSE branch: a plain chat.completion JSON body."""
    c = _client()
    r = c.post("/v1/chat/completions",
               headers={"Authorization": "Bearer local-dev"},
               json={"model": "qwen3-8b-base",
                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "BASE:hi"}


def test_streaming_returns_sse_frames():
    """Fix 1: stream=True must get real SSE frames, not a plain JSON body —
    the OpenAI SDK's SSEDecoder yields zero events from a plain JSON body,
    which is exactly the false-confidence trap this proxy fell into
    (connectivity.py pings without stream, so preflight passed regardless)."""
    c = _client()
    with c.stream("POST", "/v1/chat/completions",
                   headers={"Authorization": "Bearer local-dev"},
                   json={"model": "qwen3-8b-base",
                         "messages": [{"role": "user", "content": "hi"}],
                         "stream": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = list(r.iter_lines())

    data_lines = [line for line in lines if line.startswith("data: ")]
    assert data_lines, "expected at least one SSE data line"
    assert data_lines[-1] == "data: [DONE]"

    payloads = [json.loads(line.removeprefix("data: ")) for line in data_lines[:-1]]
    assert payloads, "expected at least one JSON chunk before [DONE]"
    for chunk in payloads:
        assert chunk["object"] == "chat.completion.chunk"

    content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in payloads
    )
    assert content == "BASE:hi"

    finish_reasons = [chunk["choices"][0]["finish_reason"] for chunk in payloads]
    assert "stop" in finish_reasons


def test_reasoning_flag_read_from_top_level():
    """Fix 2: the OpenAI SDK merges extra_body into the top-level request
    JSON, so `reasoning` arrives as a top-level key, never under an
    "extra_body" key."""
    sampler = _RecordingSampler()
    app = build_app({"qwen3-8b-base": sampler}, api_key="local-dev")
    c = TestClient(app)

    r = c.post("/v1/chat/completions",
               headers={"Authorization": "Bearer local-dev"},
               json={"model": "qwen3-8b-base",
                     "messages": [{"role": "user", "content": "hi"}],
                     "reasoning": {"enabled": True}})
    assert r.status_code == 200
    assert sampler.last_reasoning is True

    r = c.post("/v1/chat/completions",
               headers={"Authorization": "Bearer local-dev"},
               json={"model": "qwen3-8b-base",
                     "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert sampler.last_reasoning is False
