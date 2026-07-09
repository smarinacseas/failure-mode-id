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
from collections import Counter, defaultdict
from math import sqrt

import config
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
