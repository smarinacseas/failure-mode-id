"""Constraint-instance generators (T1.2). For each verifier type, sample a valid
instance and return (instruction_text, spec): the natural-language requirement
the prompt will state, plus the machine spec the verifier library grades. Every
generator takes a `random.Random` so composition is reproducible under the
PREREG split seed.

The spec a generator emits is exactly what `base.check` (and thus the GRPO
reward + eval) consumes — text and grader can never drift apart.
"""
from __future__ import annotations

import random

# --- content banks (surface topic; the constraints carry the difficulty) -----

KEYWORDS = [
    "budget", "timeline", "client", "revenue", "deadline", "milestone",
    "stakeholder", "objective", "resource", "vendor", "deliverable", "scope",
    "quality", "risk", "feedback", "schedule", "inventory", "forecast",
    "onboarding", "retention", "throughput", "compliance", "roadmap", "backlog",
]
AVOID_WORDS = ["obviously", "very", "just", "basically", "really", "literally",
               "actually", "simply"]
SECTIONS = ["Overview", "Summary", "Background", "Details", "Next Steps",
            "Recommendations", "Timeline", "Budget", "Risks", "Action Items",
            "Objectives", "Conclusion"]
END_PHRASES = [
    "Thank you for your time.",
    "Please let me know if you have any questions.",
    "That concludes the report.",
    "I look forward to your response.",
    "Best regards.",
]
REPEAT_PHRASES = ["for your review", "as discussed", "per our agreement",
                  "to be confirmed", "action required"]
ORDER_SEQUENCES = [
    ["introduction", "analysis", "recommendation", "conclusion"],
    ["context", "options", "decision"],
    ["planning", "execution", "review"],
    ["problem", "approach", "results", "next steps"],
]
FREQ_WORDS = ["team", "goal", "data", "plan", "value", "step"]

TASKS = {
    "professional_comms": [
        "Write an email to a client explaining a two-week delay on their project.",
        "Draft a company-wide announcement about a new remote-work policy.",
        "Compose a memo to the engineering team about an upcoming code freeze.",
        "Write a follow-up note to a prospect after a product demo.",
    ],
    "planning": [
        "Create a one-day schedule for a team offsite.",
        "Draft an itinerary for a three-day client visit.",
        "Write an agenda for a quarterly planning meeting.",
        "Put together a rollout plan for a new internal tool.",
    ],
    "product_content": [
        "Write a product description for a noise-cancelling travel pillow.",
        "Create a marketplace listing for a refurbished standing desk.",
        "Write a short summary of a project-management app's key features.",
        "Draft release notes for a mobile app's latest update.",
    ],
    "structured_data": [
        "Prepare a status report summarizing this sprint's progress.",
        "Write a weekly update covering three key metrics for leadership.",
        "Summarize a vendor comparison across cost, quality, and delivery.",
        "Report on quarterly sales figures for three regions.",
    ],
}


# --- helpers -----------------------------------------------------------------

def _and_list(items: list[str]) -> str:
    items = [str(i) for i in items]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _count_spec(rng: random.Random, type_name: str, lo: int, hi: int,
                exact_p: float = 0.5) -> dict:
    # exact counts are much harder than min/max; exact_p tunes how often we demand
    # exact — calibrated 2026-07-15 to land the base model in the 30-70% band.
    op = "exact" if rng.random() < exact_p else rng.choice(["min", "max"])
    return {"type": type_name, "op": op, "n": rng.randint(lo, hi)}


def _count_phrase(op: str, n: int, unit: str) -> str:
    return {"exact": f"exactly {n} {unit}",
            "min": f"at least {n} {unit}",
            "max": f"no more than {n} {unit}"}[op]


# --- coverage generators (CAUSE_A) -------------------------------------------

def gen_keyword_include(rng):
    kws = rng.sample(KEYWORDS, rng.randint(4, 6))  # more to track (calibration)
    return (f"Make sure to mention {_and_list(kws)} somewhere in the response.",
            {"type": "keyword_include", "keywords": kws})


def gen_keyword_exclude(rng):
    kws = rng.sample(AVOID_WORDS, rng.randint(1, 2))
    return (f"Avoid the word{'s' if len(kws) > 1 else ''} {_and_list(kws)} entirely.",
            {"type": "keyword_exclude", "keywords": kws})


def gen_required_sections(rng):
    secs = rng.sample(SECTIONS, rng.randint(3, 5))  # more to track (calibration)
    return (f"Organize the response under these section headers, each on its own "
            f"line: {_and_list(secs)}.",
            {"type": "required_sections", "sections": secs})


def gen_no_placeholders(rng):
    return ("Do not leave any placeholders such as [NAME] or [DATE] — fill in "
            "concrete, plausible details.",
            {"type": "no_placeholders"})


def gen_casing(rng):
    mode = rng.choice(["upper", "lower"])
    how = {"upper": "ALL CAPITAL LETTERS", "lower": "all lowercase letters"}[mode]
    return (f"Write the entire response in {how}.", {"type": "casing", "mode": mode})


def gen_no_commas(rng):
    return ("Do not use any commas anywhere in the response.",
            {"type": "no_commas"})


def gen_title(rng):
    return ("Give the response a title wrapped in double angle brackets, like "
            "<<Example Title>>.",
            {"type": "title"})


def gen_end_phrase(rng):
    phrase = rng.choice(END_PHRASES)
    return (f"End the response with exactly this sentence: {phrase}",
            {"type": "end_phrase", "phrase": phrase})


# --- precision generators (CAUSE_B) ------------------------------------------

def gen_word_count(rng):
    spec = _count_spec(rng, "word_count", 60, 180, exact_p=0.0)  # exact word count ~impossible
    return (f"The response must contain {_count_phrase(spec['op'], spec['n'], 'words')}.",
            spec)


def gen_sentence_count(rng):
    # ~50% exact (hard), ~50% min/max — the execution weak spot, calibrated
    spec = _count_spec(rng, "sentence_count", 4, 9)
    return (f"Write {_count_phrase(spec['op'], spec['n'], 'sentences')}.", spec)


def gen_paragraph_count(rng):
    spec = _count_spec(rng, "paragraph_count", 2, 5)
    return (f"Structure the response into {_count_phrase(spec['op'], spec['n'], 'paragraphs')}.",
            spec)


def gen_item_count(rng):
    spec = _count_spec(rng, "item_count", 3, 7)
    return (f"Include {_count_phrase(spec['op'], spec['n'], 'bullet points')} "
            f"(lines beginning with '- ' or '1.').",
            spec)


def gen_keyword_frequency(rng):
    kw = rng.choice(FREQ_WORDS)
    spec = _count_spec(rng, "keyword_frequency", 2, 4)
    spec["keyword"] = kw
    return (f"Use the word '{kw}' {_count_phrase(spec['op'], spec['n'], 'times')}.", spec)


def gen_caps_word_frequency(rng):
    op = rng.choice(["exact", "min"])
    n = rng.randint(1, 3)
    return (f"Include {_count_phrase(op, n, 'fully capitalized words')} "
            f"(for example, an acronym like NASA).",
            {"type": "caps_word_frequency", "op": op, "n": n})


def gen_arithmetic_result(rng):
    nums = [rng.randint(10, 99) for _ in range(rng.randint(2, 3))]
    return (f"Compute the sum of {_and_list(nums)} and state the resulting total "
            f"somewhere in the response.",
            {"type": "arithmetic_result", "expected": sum(nums)})


def gen_ordering(rng):
    seq = rng.choice(ORDER_SEQUENCES)
    items = seq[:rng.randint(3, len(seq))]
    return (f"Cover these topics in exactly this order: {_and_list(items)}.",
            {"type": "ordering", "items": items})


def gen_exact_repetition(rng):
    phrase = rng.choice(REPEAT_PHRASES)
    n = rng.randint(2, 3)
    return (f"Include the exact phrase '{phrase}' {n} times.",
            {"type": "exact_repetition", "phrase": phrase, "count": n})


GENERATORS = {
    "keyword_include": gen_keyword_include,
    "keyword_exclude": gen_keyword_exclude,
    "required_sections": gen_required_sections,
    "no_placeholders": gen_no_placeholders,
    "casing": gen_casing,
    "no_commas": gen_no_commas,
    "title": gen_title,
    "end_phrase": gen_end_phrase,
    "word_count": gen_word_count,
    "sentence_count": gen_sentence_count,
    "paragraph_count": gen_paragraph_count,
    "item_count": gen_item_count,
    "keyword_frequency": gen_keyword_frequency,
    "caps_word_frequency": gen_caps_word_frequency,
    "arithmetic_result": gen_arithmetic_result,
    "ordering": gen_ordering,
    "exact_repetition": gen_exact_repetition,
}


def generate_instance(type_name: str, rng: random.Random) -> tuple[str, dict]:
    """Return (instruction_text, spec) for a sampled instance of `type_name`."""
    return GENERATORS[type_name](rng)
