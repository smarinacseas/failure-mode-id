"""Central config: API clients, model registry, file paths.

Keys are loaded from `.env` only (never hardcoded). Candidate IDs MUST be
verified against https://openrouter.ai/models — the open-model lineup
ships monthly and these strings drift. The connectivity check in
`pipeline/generate.py` will fail loudly if any ID is wrong.
"""

from __future__ import annotations

import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Candidates via OpenRouter (OpenAI-compatible).
router = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Judge + classifier via Anthropic. Non-candidate family → no self-preference bias.
anthropic = Anthropic()

JUDGE: str = "claude-opus-4-8"

# Single family (Qwen) for v1 size ladder. DeepSeek added later for cross-family.
CANDIDATES: dict[str, str] = {
    "qwen-9b":   "qwen/qwen3.5-9b",
    "qwen-35b":  "qwen/qwen3.5-35b-a3b",
    "qwen-397b": "qwen/qwen3.5-397b-a17b",
    # "deepseek": "deepseek/deepseek-v4",  # cross-family robustness check (later)
}

# Paths
ROOT = Path(__file__).resolve().parent
DATA_XLSX = ROOT / "data" / "ComplexConstraints.xlsx"
DATA_JSONL = ROOT / "data" / "complexconstraints.jsonl"
RESPONSES_DIR = ROOT / "responses"
GRADES_DIR = ROOT / "grades"
OUTPUTS_DIR = ROOT / "outputs"
PROMPTS_DIR = ROOT / "prompts"

CRITERIA_TAGS_PATH = OUTPUTS_DIR / "criteria_tags.jsonl"
RESULTS_PATH = OUTPUTS_DIR / "results.json"
RUN_MANIFEST_PATH = OUTPUTS_DIR / "run_manifest.json"
JUDGE_VALIDATION_PATH = OUTPUTS_DIR / "judge_validation.json"
