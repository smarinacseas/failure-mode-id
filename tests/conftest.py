"""Test setup: dummy API keys so importing `config` never KeyErrors.

`config.py` reads OPENROUTER_API_KEY and builds an Anthropic() client at
import time. Tests never make real calls (network is monkeypatched), but the
imports must succeed, so seed placeholder keys before anything imports config.
"""

import os

os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
