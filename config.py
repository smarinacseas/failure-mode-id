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

# Judges (graders). A run may carry several — each grades the SAME candidate
# responses, so the dashboard can toggle between them apples-to-apples. The
# first entry doubles as the criterion classifier. JUDGE kept as a back-compat
# alias for any single-judge caller.
JUDGES: list[str] = ["claude-opus-4-8", "claude-fable-5"]
JUDGE: str = JUDGES[0]

# Single family (Qwen) for v1 size ladder. DeepSeek added later for cross-family.
CANDIDATES: dict[str, str] = {
    "qwen-9b":   "qwen/qwen3.5-9b",
    "qwen-35b":  "qwen/qwen3.5-35b-a3b",
    "qwen-397b": "qwen/qwen3.5-397b-a17b",
    # "deepseek": "deepseek/deepseek-v4",  # cross-family robustness check (later)
}

# --- Run-time knobs (defaults only; every experiment freezes its own copy of these
# into runs/<slug>/experiment.json via pipeline/run_config.py, so changing a value here
# only affects new experiments, never ones already frozen) ---

# Candidate generation (see pipeline/generate.py for the reasoning-mode rationale).
CANDIDATE_TEMPERATURE: float = 0.0
CANDIDATE_MAX_TOKENS: int = 8000
CANDIDATE_TIMEOUT_S: float = 300.0

# Judge / classifier calls. Judges run with adaptive thinking on (see
# pipeline/_judge_llm.py) so both Opus and Fable reason before emitting the
# JSON verdict — apples-to-apples. Thinking tokens count against max_tokens,
# so the budget must cover hidden thinking PLUS the verdict array; the call is
# streamed, so this can exceed the ~16k non-streaming SDK timeout guard.
JUDGE_MAX_TOKENS: int = 32000

# Validate sampler.
VALIDATE_SEED: int = 20260101
VALIDATE_SAMPLE_TARGET: int = 60
VALIDATE_RESPONSE_EXCERPT_CHARS: int = 800

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATA_XLSX = ROOT / "data" / "ComplexConstraints.xlsx"
DATA_JSONL = ROOT / "data" / "complexconstraints.jsonl"
OUTPUTS_DIR = ROOT / "outputs"
RUNS_DIR = ROOT / "runs"          # per-experiment isolated data: runs/<slug>/
PROMPTS_DIR = ROOT / "prompts"
LOGS_DIR = OUTPUTS_DIR / "logs"
PROGRESS_PATH = OUTPUTS_DIR / "progress.json"

RESULTS_PATH = OUTPUTS_DIR / "results.json"

# Per-experiment results storage — dashboard reads `index.json` for the dropdown
# and fetches the individual `<slug>.json` files on selection.
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
EXPERIMENT_INDEX_PATH = EXPERIMENTS_DIR / "index.json"
