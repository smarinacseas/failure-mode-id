"""T1.2 pool-builder tests (written before build_pools.py). A pool is
n_train + n_hold unique prompts, split-labelled, with no prompt shared between
train and holdout, all specs grade-able, reproducible under a seed, and any
externally-contaminated prompt screened out.
"""
from build_pools import build_pool
from contamination import word_ngrams
from base import check


def test_pool_sizes():
    train, hold = build_pool("coverage", n_train=12, n_hold=6, seed=1)
    assert (len(train), len(hold)) == (12, 6)


def test_no_prompt_shared_between_train_and_holdout():
    train, hold = build_pool("coverage", 20, 10, seed=2)
    assert not ({r["prompt"] for r in train} & {r["prompt"] for r in hold})


def test_all_prompts_unique():
    train, hold = build_pool("precision", 20, 10, seed=3)
    prompts = [r["prompt"] for r in train + hold]
    assert len(prompts) == len(set(prompts))


def test_specs_gradeable_and_records_labelled():
    train, hold = build_pool("precision", 8, 4, seed=4)
    for r in train + hold:
        assert r["split"] in ("train", "holdout")
        assert r["cause"] == "precision"
        for s in r["specs"]:
            assert isinstance(check("x", s).passed, bool)


def test_ids_unique_and_prefixed():
    train, hold = build_pool("coverage", 10, 5, seed=5)
    ids = [r["id"] for r in train + hold]
    assert len(ids) == len(set(ids))
    assert all(i.startswith("COV-") for i in ids)


def test_deterministic():
    a = build_pool("coverage", 8, 4, seed=6)
    b = build_pool("coverage", 8, 4, seed=6)
    assert [r["prompt"] for r in a[0]] == [r["prompt"] for r in b[0]]


def test_external_contamination_screens_out_the_overlapping_prompt():
    train, _ = build_pool("precision", 5, 2, seed=7)
    target = train[0]["prompt"]
    ref = word_ngrams(target)                      # pretend `target` is in an eval set
    t2, h2 = build_pool("precision", 5, 2, seed=7, external_ngrams=ref)
    assert target not in {r["prompt"] for r in t2 + h2}
