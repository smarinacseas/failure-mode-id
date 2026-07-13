"""Offline tests for the pure RLVR resume/data-stream helpers. Imports ONLY
training._rlvr_resume (tinker-free) — train_rlvr.py itself imports tinker at
module level, so it is deliberately not imported here."""

import pytest

from training._rlvr_resume import batch_indices, data_offset, resume_start_step


def test_data_offset_is_step_times_batch():
    assert data_offset(0, 32) == 0
    assert data_offset(151, 32) == 151 * 32
    assert data_offset(3, 128) == 384


def test_batch_indices_wraps_around_pool():
    # pool_len 10, batch 4, step 2 -> starts at (2*4)%10 = 8, wraps: 8,9,0,1
    assert batch_indices(10, 2, 4) == [8, 9, 0, 1]
    # step 0 is the identity slice
    assert batch_indices(100, 0, 32) == list(range(32))


def test_batch_indices_rejects_empty_pool():
    with pytest.raises(ValueError, match="pool_len must be positive"):
        batch_indices(0, 0, 4)


def test_resume_continues_the_exact_batch_stream():
    # The resume correctness guarantee: completing steps [0, k) then resuming
    # at step k yields the SAME overall index stream as an uninterrupted run.
    pool_len, pps, total = 137, 32, 12  # pool_len deliberately not a multiple of pps
    k = 5  # crash after step 4, next_step = 5

    uninterrupted = [batch_indices(pool_len, s, pps) for s in range(total)]

    before_crash = [batch_indices(pool_len, s, pps) for s in range(k)]
    after_resume = [batch_indices(pool_len, s, pps) for s in range(k, total)]

    assert before_crash + after_resume == uninterrupted


def test_resume_start_step_fresh_run_is_zero():
    assert resume_start_step({}, resume=True) == 0
    # RESUME flag off -> always start fresh even if a state path is recorded
    assert resume_start_step({"resume_state_path": "tinker://x", "next_step": 50}, resume=False) == 0


def test_resume_start_step_reads_recorded_next_step():
    checkpoints = {"base": "tinker://base", "resume_state_path": "tinker://state", "next_step": 150}
    assert resume_start_step(checkpoints, resume=True) == 150


def test_resume_start_step_requires_both_path_and_next_step():
    # A half-written checkpoint (path but no next_step, or vice versa) must not
    # be treated as resumable — fall back to a fresh run rather than guess.
    assert resume_start_step({"resume_state_path": "tinker://state"}, resume=True) == 0
    assert resume_start_step({"next_step": 25}, resume=True) == 0
