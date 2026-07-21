"""Tier-2 CC-75 label helper — reads E08's tracked census (dashboard JSON) and
exposes (a) the base per-criterion pass/fail + panel votes and (b) the
coverage/precision base-failed denominator, using the VERIFIED 1-based
criterion_index convention.

Verified 2026-07-20 (see RUN_TIER2.md):
- failure_analysis.rows `criterion_index` is 1-BASED; prompts[].criteria[] is
  0-based positional. idx0 = criterion_index - 1 gives 0 vote-mismatches over all
  575 coverage+precision rows and base_pass=False for all of them.
- data/complexconstraints.jsonl criteria byte-align with dashboard prompts[]
  (0 mismatches / 1559), so corpus and dashboard indices are interchangeable.

Cause map (E08 archetypes, PREREG-bound):
  constraint_unaddressed -> coverage   (254 base-failed criteria)
  execution_slip         -> precision  (321 base-failed criteria)
"""
from __future__ import annotations

import json
import pathlib
import random

_REPO = pathlib.Path(__file__).resolve().parents[3]
DASH = _REPO / "dashboard" / "E08-llama3-2-3b-cc75.json"
CORPUS = _REPO / "data" / "complexconstraints.jsonl"
CAUSE_MAP = {"constraint_unaddressed": "coverage", "execution_slip": "precision"}
SUBSET_SEED = 20260715


def load_corpus() -> dict:
    """prompt_id -> {prompt, criteria:[str], ...} (the decode/grade source)."""
    out = {}
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def load_e08():
    """Returns (base, labeled):
      base    : (pid, idx0) -> {text, base_pass, votes}   (all 1559 criteria)
      labeled : (pid, idx0) -> "coverage"|"precision"     (575 base-failed denominator)
    """
    d = json.loads(DASH.read_text(encoding="utf-8"))
    base = {}
    for p in d["prompts"]:
        for idx0, c in enumerate(p["criteria"]):
            res = c.get("results", {}).get("llama-3b", {})
            base[(p["id"], idx0)] = {"text": c["text"], "base_pass": res.get("pass"),
                                     "votes": res.get("votes")}
    labeled = {}
    for r in d["failure_analysis"]["rows"]:
        cause = CAUSE_MAP.get(r["root_cause"])
        if cause is None:
            continue
        labeled[(r["id"], r["criterion_index"] - 1)] = cause   # 1-based -> 0-based
    return base, labeled


def select_subset(n: int = 18, seed: int = SUBSET_SEED) -> list[str]:
    """~n CC-75 prompt ids stratified so both causes are represented: half drawn
    from prompts carrying coverage base-failures, half from precision, deduped.
    Deterministic under `seed`."""
    _, labeled = load_e08()
    cov_prompts, prec_prompts = set(), set()
    for (pid, _idx0), cause in labeled.items():
        (cov_prompts if cause == "coverage" else prec_prompts).add(pid)
    rng = random.Random(seed)
    cov = sorted(cov_prompts); prec = sorted(prec_prompts)
    rng.shuffle(cov); rng.shuffle(prec)
    half = n // 2
    picked: list[str] = []
    for pid in cov[:half] + prec[:half]:
        if pid not in picked:
            picked.append(pid)
    # top up to n from the union if dedup shrank it
    pool = [p for p in sorted(cov_prompts | prec_prompts) if p not in picked]
    rng.shuffle(pool)
    while len(picked) < n and pool:
        picked.append(pool.pop())
    return sorted(picked)


if __name__ == "__main__":
    base, labeled = load_e08()
    corpus = load_corpus()
    sub = select_subset()
    cov = sum(1 for (pid, _), c in labeled.items() if pid in sub and c == "coverage")
    prec = sum(1 for (pid, _), c in labeled.items() if pid in sub and c == "precision")
    print(f"E08 base criteria: {len(base)} | labeled base-failed: {len(labeled)} "
          f"(coverage {sum(c=='coverage' for c in labeled.values())}, "
          f"precision {sum(c=='precision' for c in labeled.values())})")
    print(f"\nspot-check subset (seed {SUBSET_SEED}): {len(sub)} prompts")
    print(f"  coverage base-failed criteria in subset:  {cov}")
    print(f"  precision base-failed criteria in subset: {prec}")
    print(f"  ids: {sub}")
