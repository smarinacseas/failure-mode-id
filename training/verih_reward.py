"""Native VerIH reward — faithful to ih.py's compute_score, minus the parts
that don't apply once we already hold assistant-only decoded text.

Sources (both under data/verih/, cloned from skai-research/VerIH — gitignored,
see training/data.py's module docstring for the clone command):
  - data/verih/RLVR/verl/utils/reward_score/if_functions.py
    (stdlib-only: json/re/typing) — IF_FUNCTIONS_MAP, the ~25 IFEval-style
    verifiers keyed by gt["func_name"].
  - data/verih/RLVR/verl/utils/reward_score/ih.py — the semantics we
    replicate here: check_format (has a <think>...</think> span) AND
    check_answer's aligned/conflict branch (strip through the LAST
    </think>, call IF_FUNCTIONS_MAP[func_name](answer, **non_none_args)),
    composed into a binary 1.0/0.0 reward exactly like compute_score's
    `answer_score = float(_score == 1.)`.

We do NOT import ih.py itself — it needs the `regex` package and verl's
package context (relative import `from .if_functions import ...`), neither
of which we want as a training dependency. We also skip ih.py's
extract_solution (chat-template splitting on "<|im_start|>assistant" etc.):
that's for solution_str values that still contain the full rendered
conversation. Our caller (training/train_rlvr.py) already decodes only the
model's sampled continuation tokens, so the text handed to rl_reward() here
is already assistant-only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Import if_functions.py directly as a top-level module (no packaging hacks):
# it's pure stdlib (json/re/typing), so adding its directory to sys.path and
# doing a plain `import if_functions` works without needing the surrounding
# verl.utils.reward_score package to be importable.
_REWARD_SCORE_DIR = (
    Path(__file__).resolve().parent.parent
    / "data" / "verih" / "RLVR" / "verl" / "utils" / "reward_score"
)
if str(_REWARD_SCORE_DIR) not in sys.path:
    sys.path.insert(0, str(_REWARD_SCORE_DIR))

import if_functions  # type: ignore  # noqa: E402

IF_FUNCTIONS_MAP = if_functions.IF_FUNCTIONS_MAP

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_CLOSE_TAG = "</think>"


def strip_think(text: str) -> str:
    """Return the text after the LAST `</think>`, stripped — ih.py's
    `answer_str.split("</think>")[-1].strip()`, but via rfind (equivalent,
    avoids splitting the whole string on every occurrence). Text with no
    `</think>` at all is returned unchanged."""
    idx = text.rfind(_THINK_CLOSE_TAG)
    if idx == -1:
        return text
    return text[idx + len(_THINK_CLOSE_TAG):].strip()


def has_think_block(text: str) -> bool:
    """ih.py's check_format: does the text contain a <think>...</think>
    span (DOTALL, so the block may contain newlines)?"""
    return _THINK_BLOCK_RE.search(text) is not None


def native_score(text: str, gt: dict) -> float:
    """Run the native VerIH verifier named by gt["func_name"] against the
    post-</think> answer text. 0.0 if func_name is missing/unknown, or if
    the verifier raises (raw model text can crash a verifier — e.g.
    verify_letter_frequency requires a single-character `letter`; VerIH's
    own check_answer has no try/except around this call, but ih.py's
    compute_score is invoked from verl's reward worker which already
    swallows scorer exceptions per-sample and prints+returns 0. We inline
    that same swallow-and-zero behavior here since we call the verifier
    directly, with no such wrapper upstream)."""
    func_name = gt.get("func_name")
    if not func_name or func_name not in IF_FUNCTIONS_MAP:
        return 0.0
    func = IF_FUNCTIONS_MAP[func_name]
    non_none_args = {k: v for k, v in gt.items() if v is not None and k != "func_name"}
    answer = strip_think(text)
    try:
        result = func(answer, **non_none_args)
    except Exception:
        return 0.0
    return float(bool(result))


def rl_reward(text: str, gt: dict) -> float:
    """Binary composition matching ih.py's compute_score: 1.0 iff the text
    has a think block AND the native verifier scores it 1.0, else 0.0."""
    return 1.0 if has_think_block(text) and native_score(text, gt) == 1.0 else 0.0
