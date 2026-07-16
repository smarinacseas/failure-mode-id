"""Prompt composer (T1.2). Assemble one prompt record per archetype by sampling
distinct constraint types from that archetype's pool and weaving their
instruction texts into a realistic task.

Coverage prompts bury ~half their requirements in prose (the failure mode is
DROPPING an un-obvious requirement) and list the rest; precision prompts state
their few exactness requirements as a clear checklist. A record is:

    {id, cause, domain, prompt, specs, constraint_types}

`specs` is exactly what base.check / the GRPO reward / eval consume, so the
prompt and its grader are generated together and cannot drift.
"""
from __future__ import annotations

import random

from archetypes import COVERAGE_SAMPLE_TYPES, PRECISION_TYPES
from constraints import TASKS, generate_instance


def _assemble(stem: str, texts: list[str], archetype: str) -> str:
    if archetype == "coverage":
        # Bury EVERY requirement in prose (no bullet checklist) — an un-scannable
        # wall of requirements is the coverage failure mode we train against;
        # a bulleted list makes them too easy to track (calibration 2026-07-15).
        return f"{stem} {' '.join(texts)}"
    # precision: state the exact requirements plainly as a checklist
    return f"{stem}\n\nRequirements:\n" + "\n".join(f"- {t}" for t in texts)


def compose_prompt(archetype: str, rng: random.Random, prompt_id: str | None = None) -> dict:
    if archetype == "coverage":
        # 6-7 (recalibration 2026-07-16, was 5-6): the base 3B sat at ~83% mean
        # criterion-pass (out of the 30-70% band) — per-criterion difficulty was
        # too low, so we (a) hardened the count-based generators (keyword_include,
        # required_sections, keyword_exclude), (b) dropped the near-free binaries
        # (no_placeholders, title) and added start_phrase, and (c) raised k here.
        # Teacher-yield (GT2) is re-checked on regeneration; k capped at 7 = |COVERAGE_SAMPLE_TYPES|.
        pool, k = COVERAGE_SAMPLE_TYPES, rng.randint(6, 7)
    elif archetype == "precision":
        pool, k = PRECISION_TYPES, rng.randint(3, 4)
    else:
        raise ValueError(f"unknown archetype: {archetype!r} (coverage|precision)")

    chosen = rng.sample(pool, k)                      # distinct types
    domain = rng.choice(list(TASKS))
    stem = rng.choice(TASKS[domain])
    instances = [generate_instance(t, rng) for t in chosen]
    return {
        "id": prompt_id,
        "cause": archetype,
        "domain": domain,
        "prompt": _assemble(stem, [t for t, _ in instances], archetype),
        "specs": [s for _, s in instances],
        "constraint_types": chosen,
    }
