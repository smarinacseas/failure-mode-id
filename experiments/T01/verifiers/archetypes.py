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
COVERAGE_TYPES = [
    "keyword_include",
    "keyword_exclude",
    "required_sections",
    "no_placeholders",
    "casing",
    "no_commas",
    "title",
    "end_phrase",
]

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
