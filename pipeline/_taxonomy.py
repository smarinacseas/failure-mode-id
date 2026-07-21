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

# Populated from the Pass-1 open-coding consolidation over E05's failures
# (70 blinded free-text descriptions, seed 20260707 batch; derivation +
# per-category sample support documented in the failure-taxonomy findings
# report). Each category:
#   key                  slug used in artifacts + results JSON
#   label                short human name (dashboard legend)
#   description          what the failure looks like (also shown to the analyst)
#   training_implication the Surge-style "what data this buys" line
#   requires_trace       True if diagnosing it needs the reasoning trace
DERIVED: list[dict] = [
    {
        "key": "constraint_never_surfaced",
        "label": "Never surfaced in reasoning",
        "description": ("The requirement appears nowhere in the reasoning "
                        "trace or the answer — it was never extracted from "
                        "the prompt into the model's working plan (e.g., a "
                        "formatting sub-rule the planning checklist skips "
                        "entirely)."),
        "training_implication": ("Constraint-extraction SFT: reward exhaustive "
                                 "requirement enumeration before drafting; "
                                 "checklist-style supervision."),
        "requires_trace": True,
    },
    {
        "key": "constraint_dropped",
        "label": "Tracked in reasoning, lost from answer",
        "description": ("The trace tracks or even verifies the requirement, "
                        "but iterative replanning, patchwork corrections, or "
                        "final transcription loses it — the answer omits or "
                        "contradicts what the trace established."),
        "training_implication": ("CoT→answer faithfulness rewards: penalize "
                                 "final answers that drop or contradict "
                                 "constraints their own trace satisfied; "
                                 "long-horizon state tracking across plan "
                                 "revisions."),
        "requires_trace": True,
    },
    {
        "key": "constraint_overridden",
        "label": "Noticed, then overridden",
        "description": ("The trace names the constraint, then argues itself "
                        "out of honoring it — habit, source-material "
                        "conventions, helpfulness instincts, or a "
                        "self-invented reinterpretation wins over the "
                        "explicit instruction."),
        "training_implication": ("Instruction-priority preference data: "
                                 "explicit user constraints must outrank "
                                 "priors, source formatting, and the model's "
                                 "own style habits."),
        "requires_trace": True,
    },
    {
        "key": "constraint_sacrificed",
        "label": "Sacrificed to a competing constraint",
        "description": ("Under real or perceived conflict (coverage vs "
                        "ceilings, word caps vs completeness, time budget vs "
                        "task list), the model satisfies one requirement by "
                        "violating another — often asserting compliance "
                        "anyway instead of surfacing the tradeoff."),
        "training_implication": ("Preference data over constraint conflicts: "
                                 "teach surfacing and arbitrating conflicts — "
                                 "or flagging infeasibility — instead of "
                                 "silently sacrificing and overclaiming."),
        "requires_trace": False,
    },
    {
        "key": "constraint_misread",
        "label": "Constraint misread",
        "description": ("The requirement is engaged with but interpreted "
                        "wrongly — scope, boundary semantics ('every week "
                        "after', 'even if zero'), conditional triggers, or "
                        "format words ('brackets', 'beside') resolved to the "
                        "wrong reading."),
        "training_implication": ("Contrastive instruction-semantics SFT on "
                                 "scope/boundary/conditional phrasings with "
                                 "minimal pairs."),
        "requires_trace": False,
    },
    {
        "key": "input_misread",
        "label": "Source data misread",
        "description": ("The failure originates in parsing the prompt's "
                        "source material: column misalignment, false "
                        "equivalences between sections, ambiguity silently "
                        "resolved in the wrong direction."),
        "training_implication": ("Structured-input robustness data (tables, "
                                 "rosters, misaligned columns) with explicit "
                                 "field re-derivation before use; largely "
                                 "auto-checkable, so RLVR-ready."),
        "requires_trace": False,
    },
    {
        "key": "execution_slip",
        "label": "Right intent, wrong execution",
        "description": ("The plan is right but the rendering fails: "
                        "arithmetic/counting errors, format markup omitted, "
                        "per-item rules replaced by one template, "
                        "placeholders never resolved, output never "
                        "cross-checked against its own derived values."),
        "training_implication": ("Verifier-based RL: these criteria are "
                                 "mostly auto-checkable (counts, formats, "
                                 "arithmetic) — reward outputs that pass "
                                 "mechanical validators plus a final "
                                 "self-check pass."),
        "requires_trace": False,
    },
    {
        "key": "degenerate_output",
        "label": "Degenerate generation",
        "description": ("The generation collapses — repetition loops, "
                        "endless self-correction, no committed final answer — "
                        "so whole constraint blocks go unmet at once."),
        "training_implication": ("Decoding/termination training: penalize "
                                 "non-terminating self-correction loops; "
                                 "reward committing to a finalized answer "
                                 "within budget."),
        "requires_trace": False,
    },
]

# Trace-less collapse category (spec §3): when no reasoning trace exists,
# "never noticed" and "noticed-but-dropped" are indistinguishable, both
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
    (spec §4 blinding rules 4 to 5)."""
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
