"""E07-vs-E05 failure-Pareto stability analysis (spec 2026-07-08, section 6).

Read-only over run artifacts. Compares by_root_cause rankings across four
cuts (E05's 20; E07 full-75; E07 restricted to E05's 20 = repeat-draw A/A;
E07's disjoint 55) with cluster-bootstrap CIs (clusters = prompts, because
label counts within one response are heavily correlated — one E05 response
owned 15 of 219 labels), Wilson intervals (supplementary; criteria are not
independent), rank-stability metrics, and the pre-registered decision rule.
Also computes the judge-reliability proxies (concurrence-restricted and
auto-verifiable-restricted Paretos, Fable refusal census, judge_suspect
concentration). Writes runs/<e07>/stability_analysis.json and prints a
markdown summary.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from pipeline._io import read_jsonl
from pipeline._select import select_prompts

Z95 = 1.959963984540054
DEFAULT_N_BOOT = 5000
DEFAULT_SEED = 20260708


def wilson_interval(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def rank_causes(rows: list[dict]) -> list[str]:
    counts = Counter(r["root_cause"] for r in rows)
    return [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def top_k_set(rows: list[dict], k: int = 3) -> set[str]:
    return set(rank_causes(rows)[:k])


def cause_shares(rows: list[dict]) -> dict[str, float]:
    n = len(rows)
    if not n:
        return {}
    counts = Counter(r["root_cause"] for r in rows)
    return {c: counts[c] / n for c in counts}


def kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    """Tau-a over the causes present in BOTH orderings."""
    common = [c for c in order_a if c in order_b]
    if len(common) < 2:
        return None
    pos_b = {c: i for i, c in enumerate(order_b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            if pos_b[common[i]] < pos_b[common[j]]:
                conc += 1
            else:
                disc += 1
    return (conc - disc) / (conc + disc)


def _share_samples(rows: list[dict], n_boot: int,
                   rng: random.Random) -> dict[str, list[float]]:
    """Bootstrap share samples per cause; resampling unit = prompt id."""
    by_prompt: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_prompt[r["id"]].append(r["root_cause"])
    ids = sorted(by_prompt)
    causes = sorted({r["root_cause"] for r in rows})
    samples: dict[str, list[float]] = {c: [] for c in causes}
    for _ in range(n_boot):
        labels: list[str] = []
        for _ in ids:
            labels.extend(by_prompt[rng.choice(ids)])
        counts = Counter(labels)
        n = len(labels)
        for c in causes:
            samples[c].append(counts[c] / n)
    return samples


def _pctile_ci(samples: list[float]) -> tuple[float, float]:
    s = sorted(samples)
    n = len(s)
    return (s[int(0.025 * n)], s[min(n - 1, int(0.975 * n))])


def cluster_bootstrap_shares(rows: list[dict], n_boot: int = DEFAULT_N_BOOT,
                             seed: int = DEFAULT_SEED
                             ) -> dict[str, tuple[float, float]]:
    samples = _share_samples(rows, n_boot, random.Random(seed))
    return {c: _pctile_ci(s) for c, s in samples.items()}


def bootstrap_delta_ci(rows_a: list[dict], rows_b: list[dict],
                       n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED
                       ) -> dict[str, tuple[float, float]]:
    """CI on share_a - share_b; the two runs are resampled independently."""
    sa = _share_samples(rows_a, n_boot, random.Random(seed))
    sb = _share_samples(rows_b, n_boot, random.Random(seed + 1))
    zeros = [0.0] * n_boot
    out = {}
    for c in sorted(set(sa) | set(sb)):
        da, db = sa.get(c, zeros), sb.get(c, zeros)
        out[c] = _pctile_ci([a - b for a, b in zip(da, db)])
    return out


def decision_rule(e07_rows: list[dict], e05_rows: list[dict],
                  e07_ci: dict[str, tuple[float, float]],
                  e05_ci: dict[str, tuple[float, float]]) -> dict:
    """Pre-registered (spec section 6): Pareto 'confirmed' iff E05's top-3 set
    is preserved on the E07 full cut AND each top-3 cause's bootstrap CI
    overlaps its E05 counterpart."""
    e05_top3 = top_k_set(e05_rows, 3)
    e07_top3 = top_k_set(e07_rows, 3)
    set_holds = e07_top3 == e05_top3
    overlaps = {}
    for c in sorted(e05_top3):
        lo7, hi7 = e07_ci.get(c, (0.0, 0.0))
        lo5, hi5 = e05_ci.get(c, (0.0, 0.0))
        overlaps[c] = lo7 <= hi5 and lo5 <= hi7
    return {
        "e05_top3": sorted(e05_top3),
        "e07_top3": sorted(e07_top3),
        "top3_set_holds": set_holds,
        "ci_overlap": overlaps,
        "pareto_confirmed": set_holds and all(overlaps.values()),
    }


def load_rows(slug: str) -> list[dict]:
    path = config.EXPERIMENTS_DIR / f"{slug}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("failure_analysis") or {}
    rows = block.get("rows") or []
    if not rows:
        raise SystemExit(
            f"{path} has no failure_analysis rows — run diagnose + aggregate "
            f"for {slug} first.")
    return rows


def load_verifiability(slug: str) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    for cell in read_jsonl(config.RUNS_DIR / slug / "criteria_tags.jsonl"):
        for t in cell["tags"]:
            out[(cell["id"], t["index"])] = t["verifiability"]
    return out


def load_candidate_keys(slug: str) -> list[str]:
    frozen = json.loads((config.RUNS_DIR / slug / "experiment.json")
                        .read_text(encoding="utf-8"))
    return list(frozen["params"]["candidates"])


def e05_sample_ids(limit: int = 20, seed: int = 20260706) -> set[str]:
    """E05's exact prompt subset, re-derived — never hardcoded."""
    return {r["id"] for r in
            select_prompts(read_jsonl(config.DATA_JSONL), limit, seed)}


def fable_refusal_census(slug: str, candidate_keys: list[str]) -> dict:
    cells = []
    for key in candidate_keys:
        path = (config.RUNS_DIR / slug / "grades" / "claude-fable-5"
                / f"{key}.jsonl")
        for cell in read_jsonl(path):
            if any(str(v.get("reason", "")).startswith("judge_refusal")
                   for v in cell["verdicts"]):
                cells.append({"id": cell["id"], "model": key})
    cells.sort(key=lambda c: (c["id"], c["model"]))
    return {"n_cells": len(cells), "cells": cells}


def _cut_block(rows: list[dict], ci: dict[str, tuple[float, float]]) -> dict:
    counts = Counter(r["root_cause"] for r in rows)
    n = len(rows)
    return {
        "n_rows": n,
        "n_prompts": len({r["id"] for r in rows}),
        "ranking": rank_causes(rows),
        "causes": {c: {"count": counts[c], "share": counts[c] / n,
                       "wilson_95": wilson_interval(counts[c], n),
                       "bootstrap_95": ci[c]}
                   for c in sorted(counts)},
    }


def analyze(e05_slug: str, e07_slug: str,
            n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> dict:
    e05_rows = load_rows(e05_slug)
    e07_rows = load_rows(e07_slug)
    shared_ids = e05_sample_ids()
    e07_shared = [r for r in e07_rows if r["id"] in shared_ids]
    e07_disjoint = [r for r in e07_rows if r["id"] not in shared_ids]

    cuts = {"e05_full": e05_rows, "e07_full": e07_rows,
            "e07_shared": e07_shared, "e07_disjoint": e07_disjoint}
    cis = {name: cluster_bootstrap_shares(rows, n_boot, seed)
           for name, rows in cuts.items()}

    result = {
        "config": {"e05": e05_slug, "e07": e07_slug,
                   "n_boot": n_boot, "seed": seed,
                   "n_shared_prompt_ids": len(shared_ids)},
        "cuts": {name: _cut_block(rows, cis[name])
                 for name, rows in cuts.items()},
        "rank_stability": {
            "kendall_tau_e07full_vs_e05":
                kendall_tau(rank_causes(e07_rows), rank_causes(e05_rows)),
            "kendall_tau_shared_vs_e05":
                kendall_tau(rank_causes(e07_shared), rank_causes(e05_rows)),
            "kendall_tau_disjoint_vs_e05":
                kendall_tau(rank_causes(e07_disjoint), rank_causes(e05_rows)),
        },
        "delta_e07full_minus_e05":
            bootstrap_delta_ci(e07_rows, e05_rows, n_boot, seed),
        "decision": decision_rule(e07_rows, e05_rows,
                                  cis["e07_full"], cis["e05_full"]),
    }

    # Judge-reliability proxies (spec section 6.3), E07 only.
    verif = load_verifiability(e07_slug)
    both = [r for r in e07_rows if r["judge_concurrence"] == "both_fail"]
    auto = [r for r in e07_rows
            if verif.get((r["id"], r["criterion_index"])) == "auto"]
    full_top3 = top_k_set(e07_rows)
    suspect = Counter((r["id"], r["criterion_index"]) for r in e07_rows
                      if r["root_cause"] == "judge_suspect")
    result["proxies"] = {
        "concurrence_both_fail": {
            "n_rows": len(both), "ranking": rank_causes(both),
            "top3_matches_full": top_k_set(both) == full_top3},
        "auto_verifiable": {
            "n_rows": len(auto), "ranking": rank_causes(auto),
            "top3_matches_full": top_k_set(auto) == full_top3},
        "fable_refusals":
            fable_refusal_census(e07_slug, load_candidate_keys(e07_slug)),
        "judge_suspect_concentration": [
            {"id": pid, "criterion_index": ci_, "n_models": n}
            for (pid, ci_), n in suspect.most_common()],
    }
    return result


def print_markdown(result: dict) -> None:
    for name, cut in result["cuts"].items():
        print(f"\n### {name} — {cut['n_rows']} rows / {cut['n_prompts']} prompts")
        print("| root cause | count | share | bootstrap 95% |")
        print("|---|---|---|---|")
        for c in cut["ranking"][:6]:
            info = cut["causes"][c]
            lo, hi = info["bootstrap_95"]
            print(f"| {c} | {info['count']} | {info['share']:.1%} "
                  f"| [{lo:.1%}, {hi:.1%}] |")
    d = result["decision"]
    print(f"\ntop-3 E05: {d['e05_top3']}  |  top-3 E07: {d['e07_top3']}")
    print(f"top3_set_holds={d['top3_set_holds']}  "
          f"ci_overlap={d['ci_overlap']}")
    print(f"**PARETO {'CONFIRMED' if d['pareto_confirmed'] else 'NOT STABLE'}**")
    print(f"rank_stability: {result['rank_stability']}")
    p = result["proxies"]
    print(f"proxies: both_fail top3 match={p['concurrence_both_fail']['top3_matches_full']} "
          f"(n={p['concurrence_both_fail']['n_rows']}); "
          f"auto-verifiable top3 match={p['auto_verifiable']['top3_matches_full']} "
          f"(n={p['auto_verifiable']['n_rows']}); "
          f"fable refusal cells={p['fable_refusals']['n_cells']}; "
          f"judge_suspect hotspots={p['judge_suspect_concentration'][:3]}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="E07-vs-E05 failure-Pareto stability analysis (spec section 6)")
    ap.add_argument("--e05", default="E05-reasoning-rand20p")
    ap.add_argument("--e07", default="E07-reasoning-full75")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()
    result = analyze(args.e05, args.e07, args.n_boot, args.seed)
    out = config.RUNS_DIR / args.e07 / "stability_analysis.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"stability: wrote {out}")
    print_markdown(result)


if __name__ == "__main__":
    main()
