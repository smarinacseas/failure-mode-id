import os
import pytest
import config


def test_router_transport_is_default():
    client, model = config.resolve_candidate_transport("qwen/qwen3.5-9b")
    assert client is config.router
    assert model == "qwen/qwen3.5-9b"


def test_proxy_transport_strips_scheme(monkeypatch):
    monkeypatch.setenv("PROXY_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("PROXY_API_KEY", "local-dev")
    config._PROXY_CLIENT = None  # reset the lazy singleton for the test
    client, model = config.resolve_candidate_transport("proxy://qwen3-8b-ft")
    assert model == "qwen3-8b-ft"
    assert str(client.base_url).rstrip("/") == "http://127.0.0.1:8000/v1"


def test_proxy_transport_requires_env(monkeypatch):
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    config._PROXY_CLIENT = None
    with pytest.raises(RuntimeError, match="PROXY_BASE_URL"):
        config.resolve_candidate_transport("proxy://qwen3-8b-ft")
