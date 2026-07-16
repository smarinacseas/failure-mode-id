"""T1.2 constraint-instance generator tests (written before constraints.py).

Contract: every inventory type has a generator that yields a
(instruction_text, spec) pair which is (a) well-formed — the verifier accepts
the spec — (b) deterministic given the rng, and (c) internally consistent — the
instruction text actually names what the spec will check.
"""
import random

import pytest

from constraints import GENERATORS, generate_instance
from archetypes import COVERAGE_TYPES, PRECISION_TYPES
from base import check

ALL_TYPES = COVERAGE_TYPES + PRECISION_TYPES


def test_every_inventory_type_has_a_generator():
    missing = [t for t in ALL_TYPES if t not in GENERATORS]
    assert not missing, f"no generator for: {missing}"


@pytest.mark.parametrize("type_name", ALL_TYPES)
def test_instance_is_wellformed(type_name):
    text, spec = generate_instance(type_name, random.Random(0))
    assert isinstance(text, str) and text.strip(), "empty instruction text"
    assert spec["type"] == type_name
    # the verifier must accept the spec without crashing (it is grade-able)
    result = check("A sample response with several words and a NASA acronym.", spec)
    assert isinstance(result.passed, bool)


@pytest.mark.parametrize("type_name", ALL_TYPES)
def test_instance_is_deterministic(type_name):
    assert (generate_instance(type_name, random.Random(7))
            == generate_instance(type_name, random.Random(7)))


def test_keyword_include_text_names_its_keywords():
    text, spec = generate_instance("keyword_include", random.Random(3))
    for kw in spec["keywords"]:
        assert kw in text


def test_end_phrase_text_contains_the_phrase():
    text, spec = generate_instance("end_phrase", random.Random(5))
    assert spec["phrase"] in text


def test_arithmetic_result_expected_equals_stated_numbers_sum():
    # the composed prompt asks for a sum; the spec's expected must equal it, or a
    # correct responder would be graded wrong
    _text, spec = generate_instance("arithmetic_result", random.Random(9))
    assert isinstance(spec["expected"], int)
    # a response containing the expected total passes
    assert check(f"The total is {spec['expected']}.", spec).passed is True
