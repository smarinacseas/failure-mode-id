"""T1.2 composer tests (written before compose.py). A composed prompt must draw
the right count from the right archetype pool, keep constraint types distinct,
carry grade-able specs, and be reproducible under a seed.
"""
import random

from compose import compose_prompt
from archetypes import COVERAGE_TYPES, PRECISION_TYPES
from base import check


def test_coverage_prompt_draws_from_coverage_pool():
    r = compose_prompt("coverage", random.Random(0), "COV-001")
    assert r["cause"] == "coverage"
    assert 5 <= len(r["specs"]) <= len(COVERAGE_TYPES)  # 5-6 after GT2 yield tune
    assert all(s["type"] in COVERAGE_TYPES for s in r["specs"])


def test_precision_prompt_draws_from_precision_pool():
    r = compose_prompt("precision", random.Random(0), "PRE-001")
    assert r["cause"] == "precision"
    assert 2 <= len(r["specs"]) <= 4
    assert all(s["type"] in PRECISION_TYPES for s in r["specs"])


def test_constraint_types_are_distinct():
    r = compose_prompt("coverage", random.Random(1), "x")
    types = [s["type"] for s in r["specs"]]
    assert len(types) == len(set(types))


def test_prompt_text_nonempty_and_id_and_domain_set():
    r = compose_prompt("precision", random.Random(2), "PRE-042")
    assert r["id"] == "PRE-042"
    assert r["prompt"].strip()
    assert r["domain"]


def test_composition_is_deterministic():
    assert (compose_prompt("coverage", random.Random(3), "z")
            == compose_prompt("coverage", random.Random(3), "z"))


def test_all_specs_are_gradeable():
    r = compose_prompt("precision", random.Random(4), "y")
    for s in r["specs"]:
        assert isinstance(check("a short response", s).passed, bool)


def test_unknown_archetype_raises():
    import pytest
    with pytest.raises(ValueError):
        compose_prompt("comprehension", random.Random(0), "q")
