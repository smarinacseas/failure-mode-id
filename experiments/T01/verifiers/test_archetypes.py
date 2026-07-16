"""The archetype inventory must stay disjoint and complete: every verifier is in
exactly one pool. Disjointness IS the mechanical guarantee of the coverage-vs-
precision contrast the PREREG requires (mixing collapses the interaction toward
zero); completeness stops a new verifier from silently escaping archetype
assignment."""
import archetypes
from archetypes import COVERAGE_TYPES, PRECISION_TYPES, archetype_of
from base import VERIFIERS


def test_pools_are_disjoint():
    assert set(COVERAGE_TYPES).isdisjoint(PRECISION_TYPES)


def test_all_pool_types_are_registered():
    missing = [t for t in COVERAGE_TYPES + PRECISION_TYPES if t not in VERIFIERS]
    assert not missing, f"listed but not registered: {missing}"


def test_every_registered_verifier_is_in_exactly_one_pool():
    assigned = set(COVERAGE_TYPES) | set(PRECISION_TYPES)
    registered = set(VERIFIERS)
    assert assigned == registered, (
        f"unassigned verifiers: {registered - assigned}; "
        f"phantom (listed, not registered): {assigned - registered}")


def test_archetype_of_routes_by_pool():
    assert archetype_of("keyword_include") == "coverage"
    assert archetype_of("arithmetic_result") == "precision"
