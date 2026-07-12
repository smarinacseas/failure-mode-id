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

# Optional local proxy for self-hosted / fine-tuned candidates (training track).
# A candidate whose model-id is "proxy://<served-name>" is routed here instead
# of OpenRouter. Base URL + key come from the environment (transport, NOT a
# frozen treatment) so the same experiment.json replays against any proxy host.
_PROXY_CLIENT: "OpenAI | None" = None
PROXY_SCHEME = "proxy://"


def proxy_client() -> "OpenAI":
    """Lazily build the OpenAI-compatible client for the local candidate proxy.
    Raises if the proxy env vars are absent — so a normal OpenRouter-only run
    never touches this path."""
    global _PROXY_CLIENT
    if _PROXY_CLIENT is None:
        base_url = os.environ.get("PROXY_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "a proxy:// candidate needs PROXY_BASE_URL set (e.g. "
                "http://127.0.0.1:8000/v1); start training/proxy.py first.")
        _PROXY_CLIENT = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("PROXY_API_KEY", "local-dev"),
        )
    return _PROXY_CLIENT


def resolve_candidate_transport(model_id: str):
    """(client, model_name) for a candidate model-id. proxy://X → the proxy
    client with model 'X'; anything else → the OpenRouter `router`, unchanged."""
    if model_id.startswith(PROXY_SCHEME):
        return proxy_client(), model_id[len(PROXY_SCHEME):]
    return router, model_id


# Judge + classifier via Anthropic. Non-candidate family → no self-preference bias.
anthropic = Anthropic()

# Judges (graders). A run may carry several — each grades the SAME candidate
# responses. Registry maps a short PATH-SAFE key (used in grades/<key>/ dirs,
# by_judge, custom_ids, dashboard labels) to {client, model}. OpenRouter model
# ids MUST be verified against https://openrouter.ai/models when pinned.
JUDGE_REGISTRY: dict[str, dict] = {
    "claude-opus-4-8": {"client": "anthropic",  "model": "claude-opus-4-8"},
    "claude-fable-5":  {"client": "anthropic",  "model": "claude-fable-5"},
    "gpt-5":           {"client": "openrouter", "model": "openai/gpt-5.2"},
    "gemini-3-pro":    {"client": "openrouter", "model": "google/gemini-3.1-pro-preview"},
    "deepseek-v4":     {"client": "openrouter", "model": "deepseek/deepseek-v4-pro"},
}
# Default panel: every registry key. 5 members (odd) so a full-attendance
# majority vote cannot tie. Qwen judges stay out: Qwen is the candidate
# ladder (family overlap warns, never blocks — see run_config.resolve).
JUDGES: list[str] = list(JUDGE_REGISTRY)
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

# --- Concurrency knobs (recorded in every run manifest; see CONCURRENCY.md) ---

# Worker threads for the generate stage's (candidate × prompt) work items.
GENERATION_WORKERS: int = 4

# Hard wall-clock deadline for a SINGLE candidate generation call. `timeout_s`
# is an httpx read timeout: it resets on every streamed chunk, so it bounds
# dead sockets only — not slow-trickle generations, and not the half-open
# socket a mid-stream client sleep leaves behind (neither bytes nor EOF; the
# E04 run lost 10.3 h to one such call — see meta/2026-07-03-reasoning-smoke.md).
# The deadline is enforced in pipeline/generate.py by streaming the call and
# checking elapsed wall-clock per chunk, plus a watchdog that closes the
# stream if no chunk ever arrives.
#
# Derivation (2026-07-06, scripts kept in CONCURRENCY.md): p99 (nearest-rank)
# of the 69 observed generation-call durations under the frozen reasoning-on
# treatment (48k budget, temp 0.6, provider_sort=throughput) — E04 attempt-6
# log (9 calls) + E05-reasoning-rand20p log (60 calls) — is 1014.6 s
# (qwen-9b/CIF-024, 16.9 min; per-item log deltas, retries included, so an
# upper bound on any single call). Deadline = max(2 × p99, 900 s floor)
# = max(2029.3, 900) → 2030 s. Abandoned-config E04 calls (greedy hangs,
# 32k cap-outs, up to 45.6 min) are excluded: no current freeze can
# reproduce that configuration — and all would have been caught by this guard.
#
# 2026-07-08: raised 2030 → 2700 for E07's max_tokens bump (48k → 64k). A
# legitimate run-to-cap at 64k tokens needs ~2670 s at the same worst-case
# ~24 tok/s the 2030 s figure implied for 48k — and a capped repetition loop
# should be STORED (finish_reason=length, diagnosable) rather than
# deadline-aborted into an error.
GENERATION_DEADLINE_S: float = 2700.0

# Cancel-to-collect backstop for Message Batches (grade + diagnose).
# E07 (2026-07-10): a 187-request diagnose batch showed processing=187/
# succeeded=0 for 15.5h; canceling revealed 184 had already succeeded
# server-side — the counts endpoint was stale, and the poll loop's only
# exit was `ended`. After this many seconds of in_progress with frozen
# request_counts, pipeline/_batch_guard.py cancels the batch: completed
# requests are collected (only they are billed), the remainder resubmits
# via the existing two-attempt loop. Sized ~2.4x the largest healthy drain
# observed (50 min, 220-request Opus grade batch). None disables.
BATCH_CANCEL_COLLECT_AFTER_S: float = 7200.0
GENERATION_DEADLINE_BASIS: str = (
    "2 x p99 (nearest-rank) of 69 frozen-treatment generation calls from "
    "E04/E05 logs (p99=1014.6s, qwen-9b/CIF-024), floor 900s -> 2030s; "
    "scaled by 64k/48k token budget and rounded -> 2700s (2026-07-08)"
)

# Judge / classifier calls. Judges run with adaptive thinking on (see
# pipeline/_judge_llm.py) so both Opus and Fable reason before emitting the
# JSON verdict — apples-to-apples. Thinking tokens count against max_tokens,
# so the budget must cover hidden thinking PLUS the verdict array; the call is
# streamed, so this can exceed the ~16k non-streaming SDK timeout guard.
JUDGE_MAX_TOKENS: int = 32000

# OpenRouter judge calls: streamed worker pool (Anthropic judges use Message
# Batches / sequential per the frozen judge_mode; OpenRouter has no batch API).
# Concurrency is transport, not treatment — not frozen (like GENERATION_WORKERS).
JUDGE_WORKERS: int = 4

# Hard wall-clock deadline for ONE OpenRouter judge call — same failure surface
# as candidate generation (half-open sockets, slow trickle; see
# GENERATION_DEADLINE_S). No observed p99 yet, so the basis is the token budget:
# JUDGE_MAX_TOKENS (32k) at the worst-case ~24 tok/s the generation deadline
# implies ≈ 1333 s → 1350 s. Revise from observed judge-call durations.
JUDGE_DEADLINE_S: float = 1350.0
JUDGE_DEADLINE_BASIS: str = (
    "JUDGE_MAX_TOKENS (32k) at worst-case ~24 tok/s (the GENERATION_DEADLINE_S "
    "basis rate) ≈ 1333 s, rounded to 1350 s (2026-07-11); no observed judge "
    "p99 yet — revise from data"
)

# Diagnose fallback chain (spec 2026-07-09 §4). Fable is the preferred analyst
# but demonstrably refuses on CIF-006-class content; per-item fallback to Opus
# means the preferred model does the work and degrades per CELL, not per stage
# (pre-panel this forced Opus-solo for everything). NOT frozen — diagnosis is
# post-hoc, re-runnable measurement. Entries are judge registry keys.
DIAGNOSE_CHAIN: tuple[str, ...] = ("claude-fable-5", "claude-opus-4-8")
DIAGNOSE_MAX_TOKENS: int = 32000
# Head+tail clip threshold for response/trace fields in the analyst payload
# (the E05 qwen-9b × CIF-012 runaway answer was 87.7k chars).
DIAGNOSE_MAX_FIELD_CHARS: int = 60000

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
