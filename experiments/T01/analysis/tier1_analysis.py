"""T1.4 Tier-1 analysis (PREREG §§3, 6).

Inputs: results/eval_t1_4/criteria.jsonl (per-criterion records, grade_tier1)
        results/eval_t1_4/decode_meta.jsonl (lengths/malformed — §8 artifact tell)

Estimand chain (all frozen in PREREG before any T1.4 data existed):

  majority(arm, prompt, criterion)  — pass in a strict majority of the arm's
      k decodes (k=3 → ≥2). Arm 0's majority defines its failed-criterion set.
  Rec(arm, cause) — fraction of Arm-0's failed criteria on that cause pool the
      arm now passes (majority over the arm's decodes).           (§3)
  Interaction = [Rec(RB,B) − Rec(SB,B)] − [Rec(RA,A) − Rec(SA,A)] (§3)

Uncertainty (§6): prompt-level cluster bootstrap — resample prompts with
replacement *within each cause pool*, 10,000 iterations, seed 20260715,
percentile 2.5/97.5. Criteria travel with their prompt (the cluster); Arm-0's
failed set is recomputed inside every replicate, so the denominator's sampling
noise is propagated, and a prompt drawn twice contributes twice.

  H1 confirmed  iff  CI excludes 0  and  Interaction > 0.        (§6)

Everything beyond H1 (full 6×2 Rec matrix, H2 SFT crossover DiD, artifact
table) is exploratory/descriptive (§6 multiple-comparisons stance).

Run:  python3 experiments/T01/analysis/tier1_analysis.py [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[3]
EVAL_DIR = _REPO / "results" / "eval_t1_4"

BOOTSTRAP_SEED = 20260715   # PREREG §7
N_BOOT = 10_000             # PREREG §6
CAUSES = ("coverage", "precision")
ARMS = ("0", "P", "SA", "SB", "RA", "RB")


def load_criteria(path: pathlib.Path):
    """-> {(arm, cause): {prompt_id: {criterion_id: [pass, ...]}}}"""
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                data[(d["arm"], d["cause_pool"])][d["prompt_id"]][d["criterion_id"]].append(d["pass"])
    return data


def majority_tables(data):
    """-> {(arm, cause): {prompt_id: {criterion_id: bool}}} strict-majority pass."""
    out: dict = {}
    for key, prompts in data.items():
        out[key] = {
            pid: {cid: sum(votes) * 2 > len(votes) for cid, votes in crits.items()}
            for pid, crits in prompts.items()
        }
    return out


def rec_on_prompts(maj, arm: str, cause: str, prompt_ids) -> tuple[float, int]:
    """Rec(arm, cause) over a prompt multiset (bootstrap draws repeat prompts).

    Arm-0's failed set is recomputed on the same multiset — numerator and
    denominator always see identical clusters."""
    base = maj.get(("0", cause), {})
    armt = maj.get((arm, cause), {})
    n_failed = 0
    n_recovered = 0
    for pid in prompt_ids:
        base_p = base.get(pid, {})
        arm_p = armt.get(pid, {})
        for cid, base_pass in base_p.items():
            if not base_pass:
                n_failed += 1
                if arm_p.get(cid, False):
                    n_recovered += 1
    return (n_recovered / n_failed if n_failed else float("nan")), n_failed


def interaction_on(maj, prompts_by_cause) -> float:
    ra, _ = rec_on_prompts(maj, "RA", "coverage", prompts_by_cause["coverage"])
    sa, _ = rec_on_prompts(maj, "SA", "coverage", prompts_by_cause["coverage"])
    rb, _ = rec_on_prompts(maj, "RB", "precision", prompts_by_cause["precision"])
    sb, _ = rec_on_prompts(maj, "SB", "precision", prompts_by_cause["precision"])
    return (rb - sb) - (ra - sa)


def h2_did_on(maj, prompts_by_cause) -> float:
    """H2 (exploratory): SFT data-targeting crossover DiD —
    [Rec(SA,A) − Rec(SB,A)] − [Rec(SA,B) − Rec(SB,B)]: own-data advantage on
    coverage minus own-data (dis)advantage on precision. > 0 → cause-matched
    SFT data helps the matched cause more."""
    sa_a, _ = rec_on_prompts(maj, "SA", "coverage", prompts_by_cause["coverage"])
    sb_a, _ = rec_on_prompts(maj, "SB", "coverage", prompts_by_cause["coverage"])
    sa_b, _ = rec_on_prompts(maj, "SA", "precision", prompts_by_cause["precision"])
    sb_b, _ = rec_on_prompts(maj, "SB", "precision", prompts_by_cause["precision"])
    return (sa_a - sb_a) - (sa_b - sb_b)


def bootstrap(maj, prompts_by_cause, stat_fn, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    ids = {c: list(prompts_by_cause[c]) for c in CAUSES}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        draw = {c: [ids[c][i] for i in rng.integers(0, len(ids[c]), len(ids[c]))]
                for c in CAUSES}
        stats[b] = stat_fn(maj, draw)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--criteria", default=str(EVAL_DIR / "criteria.jsonl"))
    ap.add_argument("--decode-meta", default=str(EVAL_DIR / "decode_meta.jsonl"))
    ap.add_argument("--out", default=str(EVAL_DIR / "analysis.json"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    data = load_criteria(pathlib.Path(args.criteria))
    maj = majority_tables(data)

    # full holdout prompt sets per cause (from Arm 0, which sees every prompt)
    prompts_by_cause = {c: sorted(maj[("0", c)].keys()) for c in CAUSES}
    n_prompts = {c: len(prompts_by_cause[c]) for c in CAUSES}

    # ---- descriptive 6x2 Rec matrix + base failure denominators ----
    rec_matrix: dict[str, dict[str, dict]] = {}
    for arm in ARMS:
        rec_matrix[arm] = {}
        for cause in CAUSES:
            if (arm, cause) not in maj:
                continue
            r, n_failed = rec_on_prompts(maj, arm, cause, prompts_by_cause[cause])
            rec_matrix[arm][cause] = {"rec": None if np.isnan(r) else round(r, 4),
                                      "arm0_failed_criteria": n_failed}

    # base criterion-pass rate (sanity vs the 30-70 difficulty band)
    base_pass = {}
    for cause in CAUSES:
        tbl = maj[("0", cause)]
        allc = [p for crits in tbl.values() for p in crits.values()]
        base_pass[cause] = round(sum(allc) / len(allc), 4)

    # ---- H1 point estimate + pre-registered cluster bootstrap ----
    point = interaction_on(maj, prompts_by_cause)
    boot = bootstrap(maj, prompts_by_cause, interaction_on, args.n_boot, BOOTSTRAP_SEED)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    # CI excludes 0 AND Interaction > 0  ⇔  entire CI above 0 with a positive point
    h1_confirmed = bool(point > 0 and ci[0] > 0)

    # ---- H2 (exploratory) ----
    h2_point = h2_did_on(maj, prompts_by_cause)
    h2_boot = bootstrap(maj, prompts_by_cause, h2_did_on, args.n_boot, BOOTSTRAP_SEED)
    h2_ci = (float(np.percentile(h2_boot, 2.5)), float(np.percentile(h2_boot, 97.5)))

    # ---- §8 artifact tell: answer length / malformed / marker per arm-pool ----
    tell: dict = defaultdict(lambda: {"n": 0, "chars": 0, "marker": 0, "malformed": 0, "capped": 0})
    meta_path = pathlib.Path(args.decode_meta)
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    t = tell[f"{d['arm']}/{d['cause_pool']}"]
                    t["n"] += 1
                    t["chars"] += d["answer_chars"]
                    t["marker"] += d["has_final_marker"]
                    t["malformed"] += d["malformed_empty"]
                    t["capped"] += bool(d.get("len_capped"))
    artifact = {k: {"n": v["n"],
                    "mean_answer_chars": round(v["chars"] / v["n"], 1),
                    "final_marker_rate": round(v["marker"] / v["n"], 4),
                    "malformed_rate": round(v["malformed"] / v["n"], 4),
                    "len_capped_rate": round(v["capped"] / v["n"], 4)}
                for k, v in sorted(tell.items()) if v["n"]}

    result = {
        "n_prompts": n_prompts,
        "base_criterion_pass_rate": base_pass,
        "rec_matrix": rec_matrix,
        "H1": {
            "interaction": round(point, 4),
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "n_boot": args.n_boot,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confirmed": h1_confirmed,
            "rule": "confirmed iff 95% CI excludes 0 and Interaction > 0 (PREREG §6)",
        },
        "H2_exploratory": {
            "did": round(h2_point, 4),
            "ci95": [round(h2_ci[0], 4), round(h2_ci[1], 4)],
            "definition": "[Rec(SA,A)-Rec(SB,A)] - [Rec(SA,B)-Rec(SB,B)]",
        },
        "artifact_tell": artifact,
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
