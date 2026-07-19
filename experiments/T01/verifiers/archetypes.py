"""Archetype inventory — which registered verifier type belongs to which pool.

The T1.2 composition samples constraint types from these lists. Keeping them
DISJOINT is the mechanical enforcement of the "sortable at a glance" contrast
the PREREG requires (coverage vs precision): a coverage prompt draws only from
COVERAGE_TYPES, a precision prompt only from PRECISION_TYPES. Mixing would
collapse the interaction estimand toward zero by construction.

Importing this module also imports both pools, so `import archetypes` leaves the
verifier registry fully populated.
"""
import coverage_pool  # noqa: F401 — registers coverage verifiers
import precision_pool  # noqa: F401 — registers precision verifiers

# CAUSE_A — noticing/tracking independent, easily-dropped requirements.
# Full archetype membership: every registered coverage verifier is listed here
# (disjointness/completeness guard in test_archetypes).
COVERAGE_TYPES = [
    "keyword_include",
    "keyword_exclude",
    "required_sections",
    "no_placeholders",
    "casing",
    "no_commas",
    "title",
    "end_phrase",
    "start_phrase",
]

# Recalibration 2026-07-16: `no_placeholders` (base 3B ~98% pass) and `title`
# (~93%) are RETIRED FROM SAMPLING — near-free binaries that held coverage above
# the 30-70% band — but stay registered + archetype-assigned. `start_phrase` (a
# stricter exact-opening constraint) was added. Composition draws only from the
# active set below; the interaction estimand is unaffected (coverage stays
# coverage, disjoint from precision).
_RETIRED_FROM_SAMPLING = {"no_placeholders", "title"}
COVERAGE_SAMPLE_TYPES = [t for t in COVERAGE_TYPES if t not in _RETIRED_FROM_SAMPLING]

# CAUSE_B — exact execution (counts, arithmetic, ordering, exact repetition).
PRECISION_TYPES = [
    "word_count",
    "sentence_count",
    "paragraph_count",
    "item_count",
    "keyword_frequency",
    "caps_word_frequency",
    "arithmetic_result",
    "ordering",
    "exact_repetition",
]

_ARCHETYPE = {
    **{t: "coverage" for t in COVERAGE_TYPES},
    **{t: "precision" for t in PRECISION_TYPES},
}


def archetype_of(type_name: str) -> str:
    """'coverage' or 'precision' for a registered verifier type."""
    return _ARCHETYPE[type_name]
