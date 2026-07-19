"""GRPO reward adapter (PREREG amendment 2026-07-16 (c)).

Turns the T1.1 verifier reward `constraint_reward(response, specs) -> float`
(verifiers/reward.py) into TRL's GRPO reward signature
`(prompts, completions, **kwargs) -> list[float]`:

  * the GRPO dataset carries a per-prompt `specs` column (JSON string);
  * each rollout is graded on its *final answer* only — `extract_final(c)` —
    so the reasoning scaffold before ===FINAL=== is never graded;
  * `max_chars` activates the pre-registered length cap (verbosity can't be the win).

Identical reward for the RA and RB arms — only the training data differs.
"""
from __future__ import annotations

import json
from typing import Callable

from reward import constraint_reward   # importing registers both verifier pools

from common import extract_final


def _completion_text(completion) -> str:
    """TRL hands back a plain string (standard dataset) or a list of message
    dicts (conversational dataset). Normalise to the assistant text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def _as_specs(spec) -> list[dict]:
    return json.loads(spec) if isinstance(spec, str) else spec


def make_constraint_reward(
    max_chars: int | None = None,
    *,
    malformed_penalty: float = 1.0,
    length_penalty: float = 0.5,
) -> Callable[..., list[float]]:
    """Build the TRL reward function with the length cap / penalties frozen in."""

    def constraint_reward_fn(prompts=None, completions=None, specs=None,
                             log_metric=None, **kwargs):
        rewards: list[float] = []
        malformed = 0
        ans_chars = 0
        for comp, spec in zip(completions, specs):
            answer = extract_final(_completion_text(comp))
            if not answer.strip():
                malformed += 1
            ans_chars += len(answer)
            rewards.append(constraint_reward(
                answer, _as_specs(spec),
                malformed_penalty=malformed_penalty,
                max_chars=max_chars,
                length_penalty=length_penalty,
            ))
        # surface format health + answer-length drift into the TRL logs (→ CSV);
        # reviewer asks LR selection be judged on stability, not final reward.
        if log_metric is not None and rewards:
            n = len(rewards)
            log_metric("format_ok", 1.0 - malformed / n)   # frac non-malformed rollouts
            log_metric("answer_chars_mean", ans_chars / n)  # length-drift artifact tell (PREREG §8)
        return rewards

    constraint_reward_fn.__name__ = "constraint_reward"   # readable metric name in TRL logs
    return constraint_reward_fn
