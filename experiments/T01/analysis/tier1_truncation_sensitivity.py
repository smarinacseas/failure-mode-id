"""T1.4 truncation-confound sensitivity analysis (post-hoc, exploratory).

Motivation: analysis.json's §8 artifact tell shows SA/coverage hit the
max_new_tokens=2048 cap on 16.5% of decodes (99/600, 81/200 prompts), ~10x
every other cell. Because Rec(SA,coverage) enters the H1 interaction with a
negative sign, truncation-induced SA failures would push the interaction
*down*, i.e. the confound works in the direction of the observed -0.3391.
This script quantifies that before analysis.json is treated as final.

Q1  Are capped-decode failures concentrated on truncation-sensitive criterion
    types (end_phrase / required_sections / keyword_include; need late or
    complete content) vs insensitive types (start_phrase / casing / no_commas /
    keyword_exclude; prefix-checkable or only violated by present content)?

Q2  Recompute the Rec cells and H1 interaction under three treatments of
    capped decodes, applied symmetrically to every arm (same 10k cluster
    bootstrap, seed 20260715, as PREREG §6):
      V0 baseline      : reproduce analysis.json exactly (sanity)
      V1 drop-decodes  : capped decodes removed from the majority vote
      V2 drop-prompts  : prompts where any interaction arm (RA/SA on coverage,
                         RB/SB on precision) had a capped decode are removed
                         from that cause pool (cluster-level exclusion)
      V3 upper-bound   : every criterion on a capped decode counted as PASS
                         (maximally generous to SA; if the interaction is
                         still negative here, truncation cannot explain it)

Q3/Q4  Marker forensics: ===FINAL=== (and fragments) in raw completions,
    split by capped/non-capped, for SA and RA: truncation symptom vs
    scaffold-compliance issue.

Estimand code (majority_tables / rec_on_prompts / interaction_on / bootstrap)
is imported from tier1_analysis.py, not reimplemented.

Run:  python3 experiments/T01/analysis/tier1_truncation_sensitivity.py
Out:  results/eval_t1_4/truncation_sensitivity.json (+ stdout report)
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from tier1_analysis import (  # noqa: E402
    BOOTSTRAP_SEED, N_BOOT, CAUSES,
    majority_tables, rec_on_prompts, interaction_on, h2_did_on, bootstrap,
)

_REPO = _HERE.parents[3]
EVAL_DIR = _REPO / "results" / "eval_t1_4"
FINAL_MARKER = "===FINAL==="

# needs late/complete content vs prefix-checkable / only-violated-by-present-content
TRUNC_SENSITIVE = {"end_phrase", "required_sections", "keyword_include"}
INTERACTION_ARMS = {"coverage": ("RA", "SA"), "precision": ("RB", "SB")}


def load_capped() -> set[tuple[str, str, str, int]]:
    capped = set()
    with (EVAL_DIR / "decode_meta.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d["len_capped"]:
                    capped.add((d["arm"], d["cause_pool"], d["prompt_id"], d["decode_idx"]))
    return capped


def load_criteria_with_capped(capped):
    """-> {(arm, cause): {pid: {cid: [(pass, capped), ...]}}}"""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with (EVAL_DIR / "criteria.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                is_cap = (d["arm"], d["cause_pool"], d["prompt_id"], d["decode_idx"]) in capped
                data[(d["arm"], d["cause_pool"])][d["prompt_id"]][d["criterion_id"]].append(
                    (d["pass"], is_cap))
    return data


def to_votes(data, transform):
    """Apply transform(list[(pass, capped)]) -> list[pass] and drop empty cells."""
    out: dict = {}
    for key, prompts in data.items():
        out[key] = {}
        for pid, crits in prompts.items():
            cell = {cid: transform(votes) for cid, votes in crits.items()}
            cell = {cid: v for cid, v in cell.items() if v}
            if cell:
                out[key][pid] = cell
    return out


def crit_type(cid: str) -> str:
    return cid.rsplit(":", 1)[1]


def fail_rate_by_type(data, arm: str, cause: str):
    """Decode-level fail rate per criterion type, split capped/non-capped."""
    acc: dict = defaultdict(lambda: {"capped": [0, 0], "uncapped": [0, 0]})
    for pid, crits in data[(arm, cause)].items():
        for cid, votes in crits.items():
            t = crit_type(cid)
            for passed, is_cap in votes:
                slot = acc[t]["capped" if is_cap else "uncapped"]
                slot[0] += not passed
                slot[1] += 1
    return {
        t: {
            "capped_fail_rate": round(v["capped"][0] / v["capped"][1], 4) if v["capped"][1] else None,
            "capped_n": v["capped"][1],
            "uncapped_fail_rate": round(v["uncapped"][0] / v["uncapped"][1], 4),
            "uncapped_n": v["uncapped"][1],
        }
        for t, v in sorted(acc.items())
    }


def variant_stats(maj, pools):
    cells = {}
    for cause, arms in INTERACTION_ARMS.items():
        for arm in arms:
            r, n_failed = rec_on_prompts(maj, arm, cause, pools[cause])
            cells[f"{arm}/{cause}"] = {"rec": round(r, 4), "arm0_failed_criteria": n_failed}
    point = interaction_on(maj, pools)
    boot = bootstrap(maj, pools, interaction_on, N_BOOT, BOOTSTRAP_SEED)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    h2 = h2_did_on(maj, pools)
    return {
        "rec_cells": cells,
        "interaction": round(point, 4),
        "ci95": [round(ci[0], 4), round(ci[1], 4)],
        "h1_confirmed_under_prereg_rule": bool(point > 0 and ci[0] > 0),
        "h2_did_point": round(h2, 4),
        "n_prompts": {c: len(pools[c]) for c in CAUSES},
    }


def marker_forensics(arm: str, capped) -> dict:
    out = {"capped": defaultdict(int), "uncapped": defaultdict(int)}
    with (EVAL_DIR / "decodes" / f"{arm}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            is_cap = (d["arm"], d["cause"], d["prompt_id"], d["decode_idx"]) in capped
            slot = out["capped" if is_cap else "uncapped"]
            slot["n"] += 1
            text = d["completion"]
            slot["full_marker"] += FINAL_MARKER in text
            slot["contains_FINAL"] += "FINAL" in text
            # truncated mid-marker: completion ends inside a partial marker
            tail = text.rstrip()[-15:]
            slot["ends_in_marker_fragment"] += any(
                tail.endswith(FINAL_MARKER[:k]) for k in range(3, len(FINAL_MARKER)))
    return {side: dict(v) for side, v in out.items()}


def main() -> None:
    capped = load_capped()
    data = load_criteria_with_capped(capped)

    # ---- Q1: failure concentration by criterion type (SA/coverage) ----
    q1 = {
        "SA_coverage_fail_rate_by_type": fail_rate_by_type(data, "SA", "coverage"),
        "trunc_sensitive_types": sorted(TRUNC_SENSITIVE),
    }
    # share of capped-decode failures that sit on truncation-sensitive types
    sens_fail = insens_fail = 0
    for pid, crits in data[("SA", "coverage")].items():
        for cid, votes in crits.items():
            for passed, is_cap in votes:
                if is_cap and not passed:
                    if crit_type(cid) in TRUNC_SENSITIVE:
                        sens_fail += 1
                    else:
                        insens_fail += 1
    q1["SA_coverage_capped_failures"] = {
        "on_trunc_sensitive_types": sens_fail,
        "on_trunc_insensitive_types": insens_fail,
        "sensitive_share": round(sens_fail / (sens_fail + insens_fail), 4),
    }

    # ---- Q2: sensitivity variants ----
    full_pools_maj = majority_tables(
        {k: {p: {c: [v for v, _ in vs] for c, vs in cr.items()} for p, cr in d.items()}
         for k, d in data.items()})
    pools_full = {c: sorted(full_pools_maj[("0", c)].keys()) for c in CAUSES}

    variants = {}

    # V0 baseline: must reproduce analysis.json
    v0_votes = to_votes(data, lambda vs: [p for p, _ in vs])
    variants["V0_baseline"] = variant_stats(majority_tables(v0_votes), pools_full)

    # V1 drop capped decodes (all arms); prompts losing all k decodes for an arm
    # drop out of that arm's table -> counted as not-recovered (conservative)
    v1_votes = to_votes(data, lambda vs: [p for p, cap in vs if not cap])
    v1_maj = majority_tables(v1_votes)
    zero_vote = {f"{arm}/{cause}": sorted(set(pools_full[cause]) - set(v1_maj.get((arm, cause), {})))
                 for cause, arms in INTERACTION_ARMS.items() for arm in arms}
    variants["V1_drop_capped_decodes"] = variant_stats(v1_maj, pools_full)
    variants["V1_drop_capped_decodes"]["prompts_with_zero_valid_decodes"] = {
        k: v for k, v in zero_vote.items() if v}

    # V2 drop affected prompts (cluster-level): any interaction arm capped there
    dropped = {c: sorted({pid for (arm, cause, pid, _idx) in capped
                          if cause == c and arm in INTERACTION_ARMS[c]}) for c in CAUSES}
    pools_v2 = {c: [p for p in pools_full[c] if p not in set(dropped[c])] for c in CAUSES}
    variants["V2_drop_affected_prompts"] = variant_stats(majority_tables(v0_votes), pools_v2)
    variants["V2_drop_affected_prompts"]["prompts_dropped"] = {c: len(dropped[c]) for c in CAUSES}

    # V3 upper bound: capped decode => criterion PASS, all arms
    v3_votes = to_votes(data, lambda vs: [(p or cap) for p, cap in vs])
    variants["V3_capped_counted_as_pass_upper_bound"] = variant_stats(
        majority_tables(v3_votes), pools_full)

    # ---- Q3/Q4: marker forensics ----
    forensics = {arm: marker_forensics(arm, capped) for arm in ("SA", "RA")}

    result = {
        "capped_decode_counts": {
            f"{a}/{c}": sum(1 for (arm, cause, _p, _i) in capped if arm == a and cause == c)
            for a in ("0", "P", "SA", "SB", "RA", "RB") for c in CAUSES
            if any(arm == a and cause == c for (arm, cause, _p, _i) in capped)},
        "Q1_failure_concentration": q1,
        "Q2_sensitivity_variants": variants,
        "Q3_Q4_marker_forensics": forensics,
    }
    out = EVAL_DIR / "truncation_sensitivity.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[done] -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
