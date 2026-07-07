"""Failure root-cause taxonomy + the diagnose stage's analyst prompt.

Single source of truth for the enum the diagnose stage labels against and
for the results-JSON `failure_analysis.taxonomy` echo. DERIVED is populated
once from the Pass-1 open-coding consolidation (spec §3, run report
documents the derivation); RESERVED and COLLAPSED are a-priori.

TAXONOMY_VERSION is deliberately NOT a frozen experiment param: diagnosis is
re-runnable post-hoc measurement, so a v2 relabel is a code change + re-run,
never a ConfigConflictError (spec §2.8).
"""

from __future__ import annotations

TAXONOMY_VERSION = 1

# Populated by the Pass-1 consolidation (Task 9). Each category:
#   key                  slug used in artifacts + results JSON
#   label                short human name (dashboard legend)
#   description          what the failure looks like (also shown to the analyst)
#   training_implication the Surge-style "what data this buys" line
#   requires_trace       True if diagnosing it needs the reasoning trace
DERIVED: list[dict] = []

# Trace-less collapse category (spec §3): when no reasoning trace exists,
# "never noticed" and "noticed-but-dropped" are indistinguishable — both
# collapse into this answer-observable bucket.
COLLAPSED: dict = {
    "key": "constraint_unaddressed",
    "label": "Constraint unaddressed (no trace)",
    "description": ("The answer neither satisfies nor engages the constraint; "
                    "without a reasoning trace, extraction misses and "
                    "synthesis drops cannot be told apart."),
    "training_implication": ("Coarse signal only — rerun with reasoning "
                             "enabled to split extraction vs synthesis."),
    "requires_trace": False,
}

RESERVED: list[dict] = [
    {
        "key": "judge_suspect",
        "label": "Judge suspect",
        "description": ("You cannot independently locate the asserted failure: "
                        "reading the response (and trace, if present) the "
                        "criterion appears satisfied. Use this instead of "
                        "inventing a root cause."),
        "training_implication": ("None — candidate for judge error; feeds the "
                                 "human-validation priority queue."),
        "requires_trace": False,
    },
    {
        "key": "other",
        "label": "Other",
        "description": ("A real failure that fits none of the categories "
                        "above. Describe it precisely in `rationale`."),
        "training_implication": ("Unclassified — a rising rate here means the "
                                 "taxonomy needs revision."),
        "requires_trace": False,
    },
]


def categories_for(trace_present: bool) -> list[dict]:
    """The category set offered to the analyst for one cell."""
    if trace_present:
        derived = list(DERIVED)
    else:
        derived = [c for c in DERIVED if not c["requires_trace"]] + [COLLAPSED]
    return derived + RESERVED


def allowed_keys(trace_present: bool) -> set[str]:
    return {c["key"] for c in categories_for(trace_present)}


def diagnose_system(trace_present: bool) -> str:
    """Analyst system prompt. Deliberately NOT the judge prompt: the role is
    failure analysis, the verdicts are given, and the output is evidence-first
    (spec §4 blinding rules 4–5)."""
    cats = "\n".join(
        f"- {c['key']}: {c['description']}" for c in categories_for(trace_present)
    )
    return (
        "You are a failure analyst for language-model evaluations. A model "
        "was given a prompt with many requirements; independent grading "
        "determined that the response left specific criteria unmet. Your job "
        "is to find WHY each unmet criterion failed — not to re-grade the "
        "rest of the response.\n\n"
        "For EACH unmet criterion index you are given:\n"
        "1. First locate the failure yourself: quote the shortest span of the "
        "response (or its reasoning trace, when provided) that shows it — "
        "or, for omissions, name what is absent. This is your `evidence`.\n"
        "2. Only then assign `root_cause`: exactly one key from the taxonomy "
        "below. If you cannot independently find the failure, use "
        "judge_suspect rather than inventing one.\n\n"
        "TAXONOMY:\n"
        f"{cats}\n\n"
        "Reply with ONLY a JSON array, one object per unmet criterion index "
        "you were given:\n"
        '[{"index": <int>, "evidence": "<quote or omission note>", '
        '"root_cause": "<taxonomy key>", "secondary": "<taxonomy key or null>", '
        '"confidence": "high|medium|low", "rationale": "<1-2 sentences>"}]'
    )
