"""GRPO reward wrapper (plan T1.1). Dense partial credit — the fraction of a
prompt's constraints its response satisfies — minus penalties that stop the
model from gaming the signal:

  * malformed (empty/whitespace) → a negative score, so collapsing is strictly
    worse than any real attempt (guards the degenerate_output mode E08 surfaced);
  * over-length → docked, so padding text to sneak past constraints can't win
    (the plan's "length caps ... so verbosity can't be the win").

Importing this module registers both verifier pools, so callers only need
`from reward import constraint_reward`. This is the reward fed to TRL's
GRPOTrainer in Phase T1.3 (identical for the RA and RB arms — only the data
differs). Penalty magnitudes are tunable knobs, frozen per the training config
in T1.3, not silently (PREREG §9).
"""
from __future__ import annotations

import coverage_pool  # noqa: F401 — registers coverage verifiers
import precision_pool  # noqa: F401 — registers precision verifiers
from base import check


def constraint_reward(
    response: str,
    specs: list[dict],
    *,
    malformed_penalty: float = 1.0,
    max_chars: int | None = None,
    length_penalty: float = 0.5,
) -> float:
    """Reward in roughly [-malformed_penalty, 1]. `specs` is the prompt's full
    constraint list; each is graded by base.check and partial credit is the
    pass fraction."""
    if not response.strip():
        return -malformed_penalty
    frac = (sum(check(response, s).passed for s in specs) / len(specs)) if specs else 0.0
    reward = frac
    if max_chars is not None and len(response) > max_chars:
        reward -= length_penalty
    return reward
