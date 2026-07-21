"""Deterministic output-constraint checkers: the RLVR reward signal.

No LLM judge: each constraint is a pure function of the response string.
`reward()` returns the mean satisfaction over a sample's constraint set, so
GRPO gets a smooth 0..1 signal (all-or-nothing is the len==1 special case).
The constraint TYPES mirror VerIH's format/quantity/keyword slice; add types
here as the real VerIH schema is confirmed in training/data.py.
"""

from __future__ import annotations

import json
import re


def _words(text: str) -> list[str]:
    # \b\w+\b counts contractions ("don't" -> 2) and hyphenated compounds
    # ("well-known" -> 2) as multiple words, an accepted simplification
    # for quantity constraints.
    return re.findall(r"\b\w+\b", text)


def check_constraint(constraint: dict, response: str) -> bool:
    t = constraint["type"]
    if t == "max_words":
        return len(_words(response)) <= constraint["n"]
    if t == "min_words":
        return len(_words(response)) >= constraint["n"]
    if t == "keyword_include":
        return re.search(rf"\b{re.escape(constraint['word'])}\b", response, re.IGNORECASE) is not None
    if t == "keyword_forbid":
        return re.search(rf"\b{re.escape(constraint['word'])}\b", response, re.IGNORECASE) is None
    if t == "sentence_count":
        # Heuristic: neutralize decimal points (3.14), then count
        # terminal-punctuation runs. Abbreviations (Dr., e.g.) still
        # over-count, a known limitation; dataset-native verifiers supersede
        # these checkers once the real VerIH schema lands (training/data.py).
        no_decimals = re.sub(r"(?<=\d)\.(?=\d)", "", response)
        parts = [s for s in re.split(r"[.!?]+", no_decimals) if s.strip()]
        return len(parts) == constraint["n"]
    if t == "json_parses":
        try:
            json.loads(response.strip())
            return True
        except (ValueError, TypeError):
            return False
    raise ValueError(f"unknown constraint type {t!r}")


def reward(constraints: list[dict], response: str) -> float:
    if not constraints:
        return 1.0
    ok = sum(1 for c in constraints if check_constraint(c, response))
    return ok / len(constraints)
