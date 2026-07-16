"""Coverage-pool verifiers (CAUSE_A archetype: noticing/tracking independent,
easily-dropped requirements). Importing this module registers them.

Design note: keyword matching is substring + case-insensitive by default — the
coverage failure mode is *dropping* a required word, so "apples" satisfying
"apple" is the right leniency; set case_sensitive=True to demand exact case.
"""
from __future__ import annotations

import re

from base import CheckResult, register

# All-caps bracketed [NAME], mustache {{x}}, or angle <TAG> template slots.
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_ ]*\]|\{\{.*?\}\}|<[A-Z][A-Z0-9_]*>")


def _hay(response: str, spec: dict) -> tuple[str, bool]:
    cs = spec.get("case_sensitive", False)
    return (response if cs else response.lower()), cs


@register("keyword_include")
def keyword_include(response: str, spec: dict) -> CheckResult:
    hay, cs = _hay(response, spec)
    missing = [k for k in spec["keywords"] if (k if cs else k.lower()) not in hay]
    if missing:
        return CheckResult(False, f"missing keywords: {missing}")
    return CheckResult(True, "all required keywords present")


@register("keyword_exclude")
def keyword_exclude(response: str, spec: dict) -> CheckResult:
    hay, cs = _hay(response, spec)
    present = [k for k in spec["keywords"] if (k if cs else k.lower()) in hay]
    if present:
        return CheckResult(False, f"forbidden keywords present: {present}")
    return CheckResult(True, "no forbidden keywords")


def _header_lines(response: str) -> set[str]:
    """Lines that read as section headers: markdown lead (#/*/-) and trailing
    punctuation (: *) stripped, lowercased. Prose lines are included too, but a
    section only counts if its *own* name is a full line — so an inline mention
    inside a paragraph does not satisfy the requirement."""
    out = set()
    for line in response.splitlines():
        s = re.sub(r"[\s:*]+$", "", re.sub(r"^[#*\-\s]+", "", line.strip()))
        if s:
            out.add(s.lower())
    return out


@register("required_sections")
def required_sections(response: str, spec: dict) -> CheckResult:
    headers = _header_lines(response)
    missing = [sec for sec in spec["sections"] if sec.lower() not in headers]
    if missing:
        return CheckResult(False, f"missing section headers: {missing}")
    return CheckResult(True, "all required sections present")


@register("no_placeholders")
def no_placeholders(response: str, spec: dict) -> CheckResult:
    hits = _PLACEHOLDER_RE.findall(response)
    if hits:
        return CheckResult(False, f"unfilled placeholders: {hits}")
    return CheckResult(True, "no unfilled placeholders")
