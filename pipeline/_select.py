"""Seeded, stratified prompt selection shared by generate/grade/classify.

The pipeline never persists a selected-prompt list: every stage re-reads
data/complexconstraints.jsonl and re-derives the subset from the frozen
(limit, sample_seed). Selection is therefore a pure, deterministic function
of its arguments — same inputs, same subset, across stages and resumes.

Strategy (seed set): balanced round-robin over use_case strata — one pick
per stratum per round — so small use cases are never crowded out by the
dominant ones (the real set is 34/22/10/6/1/1/1). Within a stratum, prefer
records covering an instruction_type or prompt_style not yet in the
selection; ties fall to a seed-shuffled order. Output keeps file order.
"""

from __future__ import annotations

import random


def select_prompts(records: list[dict], limit: int | None,
                   seed: int | None = None) -> list[dict]:
    if seed is None:
        return records if limit is None else records[:limit]
    if limit is None or limit >= len(records):
        return list(records)

    strata: dict[str, list[dict]] = {}
    for r in records:
        strata.setdefault(r.get("use_case", ""), []).append(r)
    rng = random.Random(seed)
    # Shuffle in sorted-key order so rng consumption is input-order-agnostic.
    for key in sorted(strata):
        rng.shuffle(strata[key])
    # Round-robin order: biggest strata first, name breaks ties.
    order = sorted(strata, key=lambda k: (-len(strata[k]), k))

    chosen: list[dict] = []
    seen_types: set = set()
    seen_styles: set = set()
    while len(chosen) < limit:
        for key in order:
            group = strata[key]
            if not group or len(chosen) >= limit:
                continue
            best = max(
                range(len(group)),
                key=lambda i: (
                    (group[i].get("instruction_type") not in seen_types)
                    + (group[i].get("prompt_style") not in seen_styles),
                    -i,  # tie → earliest in shuffled order
                ),
            )
            rec = group.pop(best)
            chosen.append(rec)
            seen_types.add(rec.get("instruction_type"))
            seen_styles.add(rec.get("prompt_style"))

    pos = {id(r): i for i, r in enumerate(records)}
    chosen.sort(key=lambda r: pos[id(r)])
    return chosen
