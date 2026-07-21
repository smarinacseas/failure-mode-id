"""Pure, tinker-free helpers for RLVR checkpoint resume + data-stream
fast-forward.

Kept in a SEPARATE module (no tinker/torch import) so they can be unit-tested
offline: training/train_rlvr.py imports tinker at module level (it's a runbook
script, per the Task 8 brief), so importing THAT into a test would drag tinker
in. These functions live here to keep tests import-light and network-free
(tests/test_train_rlvr_resume.py imports only this module).
"""

from __future__ import annotations


def data_offset(step: int, prompts_per_step: int) -> int:
    """Index into the (seeded, fixed-order) pool where `step`'s batch begins,
    before wraparound. The pool is shuffled once from a fixed SEED, so this is
    a pure function of (step, prompts_per_step): on resume, starting the loop
    at the recorded next_step and calling this reproduces exactly the batch
    offset the interrupted run would have used; the data stream continues
    where it left off with no per-lap reshuffle."""
    return step * prompts_per_step


def batch_indices(pool_len: int, step: int, prompts_per_step: int) -> list[int]:
    """Pool indices for `step`'s batch, wrapping around a pool of `pool_len`.
    Deterministic in `step`, so a run that completes steps [0, k) then resumes
    at step k yields the same overall index stream as an uninterrupted run;
    that equivalence is the resume correctness guarantee (tested offline)."""
    if pool_len <= 0:
        raise ValueError("pool_len must be positive")
    start = data_offset(step, prompts_per_step) % pool_len
    return [(start + i) % pool_len for i in range(prompts_per_step)]


def resume_start_step(checkpoints: dict, resume: bool) -> int:
    """Loop start step given a loaded checkpoints.json dict and the RESUME
    flag. Returns the recorded `next_step` when resuming from a durable state
    checkpoint (RESUME set AND both resume_state_path + next_step present),
    else 0 (fresh run)."""
    if resume and checkpoints.get("resume_state_path") and "next_step" in checkpoints:
        return int(checkpoints["next_step"])
    return 0
