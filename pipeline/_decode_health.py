"""Mechanical decode-health signals: repetition-loop detection + per-run census.

User ruling (2026-07-09, during E07): ANY repetition loop, in the reasoning
channel or the visible answer, whether the model escaped it or ran the budget
out, counts as a failure. Loops waste tokens, escaping them is unreliable,
and looping is trainable behavior (repetition penalties, process supervision),
so the eval surfaces it as first-class, auto-verifiable signal rather than an
ops footnote. E07 evidence: qwen-9b's largest stored reasoning traces were
loops that escaped (e.g. CIF-055: 113-char period, chars 31k→149k, then a
correct answer); unescaped loops exhausted the shared thinking+answer budget
and surfaced only as retries/errors.

Everything here is pure computation over stored artifacts, no API calls, no
judge cost, so `aggregate` folds it into every run's results unconditionally.
"""

from __future__ import annotations

from collections import Counter

from pipeline._io import read_jsonl

# Loop-detector heuristic: a text is "looping" when one NGRAM-char shingle
# recurs at least MIN_REPEATS times inside the WINDOW-char tail. 20 repeats
# of one 40-char shingle is ~20% of a 4000-char tail, far beyond anything
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
    onset = index of that shingle's first occurrence in the FULL text,
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


def decode_health_block(cfg) -> dict:
    """Per-run loop census over stored responses AND captured rejected
    attempts (runs/<slug>/rejected/<key>.jsonl, written by generate's
    on_reject hook). `rows` carries only flagged responses (a loop in either
    channel, or a non-stop finish_reason) while `by_model` tallies cover
    everything.
    """
    rows: list[dict] = []
    by_model: dict[str, dict] = {}
    for key in cfg.candidates:
        tallies = {
            "n_responses": 0,
            "n_loop_reasoning": 0,
            "n_loop_content": 0,
            "n_loop_any": 0,
            "n_loop_escaped": 0,   # looped but finish==stop, still a failure
            "n_loop_died": 0,      # looped and never finished cleanly
            "n_nonstop_finish": 0,
        }
        for r in read_jsonl(cfg.responses_path(key)):
            tallies["n_responses"] += 1
            reasoning = r.get("reasoning", "")
            content = r.get("response", "")
            r_loop = detect_repetition_loop(reasoning)
            c_loop = detect_repetition_loop(content)
            finish = r.get("finish_reason")
            if finish != "stop":
                tallies["n_nonstop_finish"] += 1
            if r_loop["looping"]:
                tallies["n_loop_reasoning"] += 1
            if c_loop["looping"]:
                tallies["n_loop_content"] += 1
            if r_loop["looping"] or c_loop["looping"]:
                tallies["n_loop_any"] += 1
                bucket = "n_loop_escaped" if finish == "stop" else "n_loop_died"
                tallies[bucket] += 1
            if r_loop["looping"] or c_loop["looping"] or finish != "stop":
                rows.append({
                    "model": key,
                    "id": r["id"],
                    "finish_reason": finish,
                    "reasoning_chars": len(reasoning),
                    "content_chars": len(content),
                    "reasoning_loop": r_loop,
                    "content_loop": c_loop,
                })
        rejected = read_jsonl(cfg.run_dir / "rejected" / f"{key}.jsonl")
        tallies["rejected_attempts"] = {
            "n_captured": len(rejected),
            "n_looping": sum(
                1 for r in rejected
                if detect_repetition_loop(r.get("reasoning", ""))["looping"]
                or detect_repetition_loop(r.get("response", ""))["looping"]
            ),
        }
        by_model[key] = tallies
    return {
        "detector": {"window": WINDOW, "ngram": NGRAM,
                     "min_repeats": MIN_REPEATS},
        "verdict_note": ("any repetition loop counts as a failure "
                         "(user ruling 2026-07-09), including loops the "
                         "model escaped before finishing"),
        "by_model": by_model,
        "rows": rows,
    }
