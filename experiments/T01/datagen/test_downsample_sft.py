"""Tests for SFT parity down-sampling (PREREG amendment DRAFT 2026-07-16).
Asserts determinism, count parity across cells, sorted-before-sample, min-count
identity, and zero overlap with data/holdout/."""
import json
import random
from pathlib import Path

from downsample_sft import (ARM_CAUSE, SEED, _select, accepted_ids,
                            build_manifest, select_ids, target_n)

_T01 = Path(__file__).resolve().parents[1]


def test_target_n_is_min_accepted_across_causes():
    assert target_n() == min(len(accepted_ids(c)) for c in ARM_CAUSE.values())


def test_selection_is_deterministic():
    assert select_ids("precision", 100) == select_ids("precision", 100)


def test_sorted_before_sample_is_order_invariant():
    ids = accepted_ids("precision")
    shuffled = ids[:]
    random.Random(1).shuffle(shuffled)
    # same n, same seed, different input order -> identical selection
    assert _select(shuffled, 100, SEED) == _select(sorted(ids), 100, SEED)


def test_count_parity_across_both_cells():
    n = target_n()
    for arm in ARM_CAUSE:
        m = build_manifest(arm, n)
        assert m["n_selected"] == n == len(m["selected_ids"])
    # the two cells train on equal n
    assert len({build_manifest(a, n)["n_selected"] for a in ARM_CAUSE}) == 1


def test_min_count_cell_is_identity():
    # coverage is the min-count cell: selecting target_n of it keeps the whole set
    n = target_n()
    cov = "coverage"
    assert len(accepted_ids(cov)) == n            # coverage IS the min
    assert set(select_ids(cov, n)) == set(accepted_ids(cov))


def test_selected_ids_are_subset_of_accepted():
    n = target_n()
    for arm, cause in ARM_CAUSE.items():
        assert set(build_manifest(arm, n)["selected_ids"]) <= set(accepted_ids(cause))


def test_zero_overlap_with_reserved_eval_pool():
    n = target_n()
    reserved = set()
    for cause in ARM_CAUSE.values():
        for line in (_T01 / "data" / "holdout" / f"{cause}.jsonl").open(encoding="utf-8"):
            reserved.add(json.loads(line)["id"])
    for arm in ARM_CAUSE:
        assert not (set(build_manifest(arm, n)["selected_ids"]) & reserved)
