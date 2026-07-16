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

from archetypes import COVERAGE_TYPES, PRECISION_TYPES
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
        # 5-6 (was 7-8): GT2 teacher-yield kill-switch — a strong teacher can't
        # reliably 100%-pass 7-8 all-or-nothing constraints (correlated fails on
        # no_commas/casing). Per-criterion difficulty is ~k-independent, so the
        # 3B calibration (~75%) holds while the teacher's all-pass yield rises.
        pool, k = COVERAGE_TYPES, rng.randint(5, 6)
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
