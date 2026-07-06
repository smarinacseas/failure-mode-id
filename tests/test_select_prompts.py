"""select_prompts: seeded, stratified prompt selection.

Every stage recomputes the selection independently from the same frozen
(limit, sample_seed), so the function must be pure and deterministic.
"""

from pipeline._select import select_prompts


def _rec(i, use_case, itype, style):
    return {
        "id": f"P{i:03d}",
        "prompt": f"prompt {i}",
        "use_case": use_case,
        "instruction_type": itype,
        "prompt_style": style,
        "criteria": [],
    }


def _skewed_dataset():
    """Mimics the real set's shape: dominant stratum first, singleton last —
    so first-N selection provably misses the tail use case."""
    recs = []
    i = 0
    for _ in range(12):
        recs.append(_rec(i, "logistics", "Negative", "Direct")); i += 1
    for _ in range(6):
        recs.append(_rec(i, "data-math", "Multistep", "Context")); i += 1
    for _ in range(3):
        recs.append(_rec(i, "comms", "Conditional", "Rambling")); i += 1
    recs.append(_rec(i, "creative", "Implicit", "Direct"))
    return recs


def test_no_seed_is_first_n_backcompat():
    recs = _skewed_dataset()
    assert select_prompts(recs, 3, None) == recs[:3]
    assert select_prompts(recs, None, None) == recs


def test_limit_none_returns_all_even_with_seed():
    recs = _skewed_dataset()
    assert select_prompts(recs, None, 42) == recs


def test_limit_at_or_above_len_returns_all():
    recs = _skewed_dataset()
    assert select_prompts(recs, len(recs), 42) == recs
    assert select_prompts(recs, len(recs) + 5, 42) == recs


def test_same_seed_same_selection():
    recs = _skewed_dataset()
    a = select_prompts(recs, 8, 20260706)
    b = select_prompts(recs, 8, 20260706)
    assert a == b
    assert len(a) == 8
    ids = [r["id"] for r in a]
    assert len(set(ids)) == 8


def test_selection_preserves_file_order():
    recs = _skewed_dataset()
    chosen = select_prompts(recs, 8, 7)
    positions = [recs.index(r) for r in chosen]
    assert positions == sorted(positions)


def test_every_use_case_represented_when_limit_allows():
    """First-N would return 8x logistics; stratified must reach the tail,
    including the singleton 'creative' stratum."""
    recs = _skewed_dataset()
    chosen = select_prompts(recs, 8, 1)
    assert {r["use_case"] for r in chosen} == {
        "logistics", "data-math", "comms", "creative"
    }


def test_allocation_balances_across_strata():
    """Equal-size strata + limit divisible by stratum count → equal counts."""
    recs = []
    i = 0
    for uc in ("a", "b", "c"):
        for _ in range(10):
            recs.append(_rec(i, uc, "Negative", "Direct")); i += 1
    chosen = select_prompts(recs, 6, 99)
    counts = {}
    for r in chosen:
        counts[r["use_case"]] = counts.get(r["use_case"], 0) + 1
    assert counts == {"a": 2, "b": 2, "c": 2}


def test_covers_instruction_types_and_styles_greedily():
    """Within a stratum the pick prefers unseen instruction types / prompt
    styles: 3 styles x 4 types all present in one use case → any 4 picks
    must cover all 4 types and all 3 styles, whatever the seed."""
    styles = ("Direct", "Context", "Rambling")
    types = ("Negative", "Multistep", "Implicit", "Conditional")
    recs = [
        _rec(i, "logistics", types[i % 4], styles[i % 3]) for i in range(12)
    ]
    for seed in (0, 1, 2, 20260706):
        chosen = select_prompts(recs, 4, seed)
        assert {r["instruction_type"] for r in chosen} == set(types), seed
        assert {r["prompt_style"] for r in chosen} == set(styles), seed
