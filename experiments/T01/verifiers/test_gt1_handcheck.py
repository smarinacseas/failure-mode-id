"""Gate GT1 human-agreement check: ~20 realistic (not toy) responses whose
verdict a human can read off at a glance, asserted against the verifier. If the
grader disagrees with the obvious human call on any of these, that's a verifier
bug to fix before it poisons the GRPO reward.
"""
import pytest

import archetypes  # noqa: F401 — importing registers every coverage + precision verifier
from base import check

# (label, response, spec, human_verdict)
CASES = [
    ("sections present as real headers",
     "## Summary\nThe project is on track.\n\n## Next Steps\nWe finalize the budget Friday.",
     {"type": "required_sections", "sections": ["Summary", "Next Steps"]}, True),
    ("sections only mentioned inline, never as headers",
     "In this summary I walk through the next steps we should take together.",
     {"type": "required_sections", "sections": ["Summary", "Next Steps"]}, False),

    ("clean email, no placeholders",
     "Hi Dana, your order ships Monday and arrives by Thursday. Thanks for your patience.",
     {"type": "no_placeholders"}, True),
    ("left an [NAME] template slot unfilled",
     "Hi [NAME], your order ships Monday. Thank you.",
     {"type": "no_placeholders"}, False),
    ("lowercase bracket is a real aside, not a slot",
     "The revenue rose sharply [see the attached chart] over the quarter.",
     {"type": "no_placeholders"}, True),

    ("correct computed total is stated",
     "Adding the three line items (40, 55, and 30) gives a total of 125 dollars.",
     {"type": "arithmetic_result", "expected": 125}, True),
    ("states a wrong total",
     "The three items add up to about 120 dollars in the end.",
     {"type": "arithmetic_result", "expected": 125}, False),
    ("the target number only appears inside a larger number",
     "Revenue reached 1250 dollars, up from last quarter.",
     {"type": "arithmetic_result", "expected": 125}, False),

    ("topics discussed in the required order",
     "First the introduction sets context, the analysis weighs options, and the "
     "conclusion recommends a path.",
     {"type": "ordering", "items": ["introduction", "analysis", "conclusion"]}, True),
    ("topics out of order",
     "The conclusion is clear, but let me back up to the introduction first.",
     {"type": "ordering", "items": ["introduction", "conclusion"]}, False),

    ("all-caps announcement",
     "OFFICE CLOSED FRIDAY FOR MAINTENANCE. PLEASE PLAN ACCORDINGLY.",
     {"type": "casing", "mode": "upper"}, True),
    ("almost all caps but one lowercase word slipped in",
     "OFFICE CLOSED FRIDAY for MAINTENANCE.",
     {"type": "casing", "mode": "upper"}, False),

    ("ends with exactly the required sign-off",
     "Please review the attached summary at your convenience. Best regards.",
     {"type": "end_phrase", "phrase": "Best regards."}, True),
    ("keeps writing after the required sign-off",
     "Best regards. P.S. one more thing about the invoice.",
     {"type": "end_phrase", "phrase": "Best regards."}, False),

    ("uses 'team' exactly three times as a word",
     "The team met today. The team agreed on scope. The team will reconvene Friday.",
     {"type": "keyword_frequency", "keyword": "team", "op": "exact", "n": 3}, True),
    ("'teamwork' must not count toward the 'team' quota",
     "The team met today and the team praised the teamwork afterward.",
     {"type": "keyword_frequency", "keyword": "team", "op": "exact", "n": 3}, False),

    ("short reply within the word cap",
     "Thanks for the update. I will review it and reply by Friday.",
     {"type": "word_count", "op": "max", "n": 12}, True),
    ("reply blows past a tight word cap",
     "I appreciate the detailed report you sent yesterday and I will get back to you soon.",
     {"type": "word_count", "op": "max", "n": 8}, False),

    ("two sentences, with a decimal that is not a boundary",
     "The pi value is 3.14 in our current model. That is precise enough for now.",
     {"type": "sentence_count", "op": "exact", "n": 2}, True),

    ("exactly two blank-line-separated paragraphs",
     "This is the first paragraph of the update.\n\nThis is the second and final paragraph.",
     {"type": "paragraph_count", "op": "exact", "n": 2}, True),

    ("comma-free sentence",
     "The vendor confirmed delivery for Monday and shipping is already scheduled.",
     {"type": "no_commas"}, True),
    ("a single comma breaks the no-comma rule",
     "The vendor confirmed delivery, and shipping is scheduled.",
     {"type": "no_commas"}, False),

    ("all required keywords are present",
     "Our roadmap sets the timeline and the budget for each milestone this quarter.",
     {"type": "keyword_include", "keywords": ["roadmap", "timeline", "budget"]}, True),

    ("an acronym satisfies the fully-capitalized-word requirement",
     "The report references NASA data to support the forecast.",
     {"type": "caps_word_frequency", "op": "min", "n": 1}, True),
]


@pytest.mark.parametrize("label,response,spec,human", CASES, ids=[c[0] for c in CASES])
def test_verifier_agrees_with_human(label, response, spec, human):
    assert check(response, spec).passed is human, (
        f"[{label}] verifier said {check(response, spec).passed}, human says {human}")


def test_handcheck_covers_at_least_20_cases():
    assert len(CASES) >= 20
