"""13-gram contamination screen (T1.2, plan step 3). Word-level n-grams, lowercased
and punctuation-stripped, so a composed prompt can be checked for overlap against
the T01 holdout, CC-75, and IFEval originals. Overlap on any 13-gram = a hit
(discard + regenerate). Also used internally to keep train and holdout disjoint.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_WORD = re.compile(r"\w+")


def word_ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Set of consecutive n-word tuples (lowercased). Empty if < n words."""
    words = _WORD.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def shares_ngram(text: str, reference: set[tuple[str, ...]], n: int = 13) -> bool:
    return bool(word_ngrams(text, n) & reference)


def ngrams_of_jsonl(path: Path, fields=("prompt",), n: int = 13) -> set[tuple[str, ...]]:
    """Union of n-grams over the given text fields of every row in a JSONL file."""
    ref: set[tuple[str, ...]] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for field in fields:
                v = row.get(field)
                if isinstance(v, str):
                    ref |= word_ngrams(v, n)
    return ref


def load_reference_ngrams(paths: list[Path], n: int = 13) -> tuple[set, list[str]]:
    """Build the combined reference n-gram set from whatever source files exist.
    Returns (ngrams, report) where report lists which paths were screened vs
    missing — the plan wants contamination screening logged, not silent."""
    ref: set[tuple[str, ...]] = set()
    report = []
    for p in paths:
        if p.exists():
            g = ngrams_of_jsonl(p, n=n)
            ref |= g
            report.append(f"screened vs {p} ({len(g)} 13-grams)")
        else:
            report.append(f"MISSING (skipped): {p}")
    return ref, report
