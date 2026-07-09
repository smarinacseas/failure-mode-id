"""Sidecar degenerate-output probe for E07 (spec 2026-07-08, section 7).

Repeat-draws CIF-012 under E07's frozen treatment on the two models that have
hosted degenerate_output (qwen-9b in E05, qwen-35b in E06), then runs decode
forensics over the draws, E07's own responses, and E05's original CIF-012
response. Writes ONLY sidecar paths under runs/<slug>/probe/ — never the
pipeline-owned responses/ files — so the frozen run and its resume logic are
untouched. Deliberately does not call run_config.resolve(): on a missing slug
resolve() would freeze DEFAULT params; this script must fail instead.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import config
from pipeline._io import append_jsonl, read_jsonl
from pipeline.generate import _generate_one
from pipeline.run_config import RunConfig

SLUG = "E07-reasoning-full75"
PROMPT_ID = "CIF-012"
PROBE_MODELS = ("qwen-9b", "qwen-35b")
DRAWS_PER_MODEL = 5

# Loop-detector heuristic: a text is "looping" when one NGRAM-char shingle
# recurs at least MIN_REPEATS times inside the WINDOW-char tail. 20 repeats
# of one 40-char shingle is ~20% of a 4000-char tail — far beyond anything
# varied prose produces, while catching both long-unit loops (period ~ tens
# of chars) and single-char runaways (period 1).
WINDOW = 4000
NGRAM = 40
MIN_REPEATS = 20


def detect_repetition_loop(text: str, window: int = WINDOW, ngram: int = NGRAM,
                           min_repeats: int = MIN_REPEATS) -> dict:
    """Return {"looping", "period", "onset"} for `text`.

    period = median gap between consecutive occurrences of the dominant
    tail shingle (i.e. the repeat-unit length; 1 for single-char runaways);
    onset = index of that shingle's first occurrence in the FULL text —
    roughly where the loop began.
    """
    tail = text[-window:]
    if len(tail) < ngram * 2:
        return {"looping": False, "period": None, "onset": None}
    counts = Counter(tail[i:i + ngram] for i in range(len(tail) - ngram + 1))
    shingle, n = counts.most_common(1)[0]
    if n < min_repeats:
        return {"looping": False, "period": None, "onset": None}
    hits = []
    start = 0
    while (i := text.find(shingle, start)) != -1:
        hits.append(i)
        start = i + 1
    gaps = sorted(b - a for a, b in zip(hits, hits[1:]))
    period = gaps[len(gaps) // 2] if gaps else None
    return {"looping": True, "period": period, "onset": hits[0]}


def forensic_row(source: str, model: str, rec: dict) -> dict:
    """Mechanical decode forensics for one stored response record."""
    reasoning = rec.get("reasoning", "")
    content = rec.get("response", "")
    return {
        "source": source,
        "model": model,
        "id": rec.get("id", PROMPT_ID),
        "finish_reason": rec.get("finish_reason"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "content_loop": detect_repetition_loop(content),
        "reasoning_loop": detect_repetition_loop(reasoning),
    }
