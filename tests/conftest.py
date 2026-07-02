"""Test setup: dummy API keys so importing `config` never KeyErrors.

`config.py` reads OPENROUTER_API_KEY and builds an Anthropic() client at
import time. Tests never make real calls (network is monkeypatched), but the
imports must succeed, so seed placeholder keys before anything imports config.
"""

import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")


def make_cfg(slug="E99-test", **kw):
    """Small RunConfig for stage tests. Import lazily so key-seeding runs first."""
    from pipeline.run_config import RunConfig
    params = dict(
        candidates={"m1": "model-1"},
        judge="claude-fable-5",
        max_tokens=100,
        temperature=0.0,
        reasoning=False,
        timeout_s=5.0,
        limit=None,
        description="",
    )
    params.update(kw)
    return RunConfig(slug=slug, **params)
