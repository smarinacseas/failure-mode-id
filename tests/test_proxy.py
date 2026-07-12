from fastapi.testclient import TestClient
from training.proxy import build_app


class _FakeSampler:
    def __init__(self, tag): self.tag = tag
    def generate(self, messages, max_tokens, temperature, reasoning):
        return f"{self.tag}:{messages[-1]['content']}"


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
