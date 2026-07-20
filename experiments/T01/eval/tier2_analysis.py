"""Tier-2 CC-75 recovery analysis — the 4×2 recovery matrix on the
stack-parity-corrected denominator, + breakage, + prompt-clustered bootstrap CIs.

Denominator (per cause) = {(pid, idx0) : E08 labels it that cause} ∩
{local Arm-0 majority-FAILS it under the identical opus/k=3 protocol}. Measuring
recovery against the LOCAL Arm-0 (same grader/stack/k as the trained arms) cancels
the ~20% stack+grader+k false-recovery floor the Step-2 spot-check measured
(RUN_TIER2.md §2). E08's panel labels are used only to assign the *cause* of each
criterion, never as the pass/fail reference.

    Rec_CC75(arm, cause) = fraction of the cause denominator the arm now passes.
    Brk_CC75(arm)        = fraction of {local Arm-0 majority-PASSES} the arm now fails.

Exploratory/descriptive (PREREG §6): CIs are prompt-clustered bootstrap, reported
without any confirmatory pass/fail rule.

Run (after tier2_decode + tier2_grade for arms 0,SA,SB,RA,RB, full 75):
    python3 experiments/T01/eval/tier2_analysis.py
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parent))
from tier2_labels import load_e08          # noqa: E402

GRADES = _REPO / "results" / "eval_t2" / "grades"
OUT = _REPO / "results" / "eval_t2" / "analysis.json"
ARMS = ("SA", "SB", "RA", "RB")
CAUSES = ("coverage", "precision")
BOOT_SEED = 20260715
BOOT_N = 10000


def majority(tag: str) -> dict:
    """(pid, idx0) -> majority pass (over k decodes) from grades/{tag}.jsonl."""
    path = GRADES / f"{tag}.jsonl"
    if not path.exists():
        raise SystemExit(f"missing grades for {tag}: {path}")
    votes: dict[tuple[str, int], list[bool]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                votes.setdefault((d["prompt_id"], d["idx0"]), []).append(bool(d["pass"]))
    return {k: (sum(v) > len(v) / 2) for k, v in votes.items()}


def main() -> None:
    _base, labeled = load_e08()
    arm0 = majority("0")
    arm_pass = {a: majority(a) for a in ARMS}

    # Denominator per cause: E08 cause-labeled ∩ local Arm-0 majority-fails.
    denom = {c: [] for c in CAUSES}         # list of (pid, idx0)
    dropped_no_arm0 = 0
    for (pid, idx0), cause in labeled.items():
        key = (pid, idx0)
        if key not in arm0:
            dropped_no_arm0 += 1            # local Arm-0 not graded (should be 0 on a full run)
            continue
        if arm0[key] is False:             # local base also fails it
            denom[cause].append(key)
    # base-PASSED set (for breakage) = local Arm-0 majority-passes
    base_passed = [k for k, p in arm0.items() if p is True]

    def rec(arm: str, cause: str, keys: list) -> tuple[int, int]:
        ap = arm_pass[arm]
        passed = sum(1 for k in keys if ap.get(k) is True)
        return passed, len(keys)

    # point estimates
    matrix = {}
    for a in ARMS:
        matrix[a] = {}
        for c in CAUSES:
            p, n = rec(a, c, denom[c])
            matrix[a][c] = {"recovered": p, "denom": n, "rec": round(p / n, 4) if n else None}
        bp = sum(1 for k in base_passed if arm_pass[a].get(k) is False)
        matrix[a]["breakage"] = {"broken": bp, "base_passed": len(base_passed),
                                 "brk": round(bp / len(base_passed), 4) if base_passed else None}

    # prompt-clustered bootstrap CIs for Rec (resample the 75 prompts w/ replacement)
    prompts = sorted({pid for (pid, _idx0) in labeled})
    by_prompt = {p: {c: [] for c in CAUSES} for p in prompts}
    for c in CAUSES:
        for (pid, idx0) in denom[c]:
            by_prompt[pid][c].append((pid, idx0))
    rng = random.Random(BOOT_SEED)
    ci = {a: {c: None for c in CAUSES} for a in ARMS}
    for a in ARMS:
        ap = arm_pass[a]
        for c in CAUSES:
            dist = []
            for _ in range(BOOT_N):
                num = den = 0
                for _ in range(len(prompts)):
                    pid = prompts[rng.randrange(len(prompts))]
                    keys = by_prompt[pid][c]
                    den += len(keys)
                    num += sum(1 for k in keys if ap.get(k) is True)
                dist.append(num / den if den else 0.0)
            dist.sort()
            ci[a][c] = [round(dist[int(0.025 * BOOT_N)], 4), round(dist[int(0.975 * BOOT_N)], 4)]

    out = {
        "denominator": {c: len(denom[c]) for c in CAUSES},
        "denominator_note": "E08 cause-labeled ∩ local-Arm-0 majority-fails (stack+grader+k bias canceled)",
        "e08_labeled_totals": {c: sum(v == c for v in labeled.values()) for c in CAUSES},
        "dropped_no_local_arm0_grade": dropped_no_arm0,
        "matrix": matrix, "rec_ci95_prompt_bootstrap": ci,
        "boot_n": BOOT_N, "boot_seed": BOOT_SEED,
    }
    OUT.write_text(json.dumps(out, indent=2))

    print("=== Tier-2 CC-75 recovery (local-Arm-0-intersected denominator) ===")
    print(f"denominator: coverage {out['denominator']['coverage']} "
          f"(of {out['e08_labeled_totals']['coverage']} E08-labeled), "
          f"precision {out['denominator']['precision']} (of {out['e08_labeled_totals']['precision']})")
    print(f"{'arm':<4}{'Rec(cov)':>20}{'Rec(prec)':>20}{'Brk':>10}")
    for a in ARMS:
        rc = matrix[a]["coverage"]; rp = matrix[a]["precision"]; bk = matrix[a]["breakage"]
        cc = f"{rc['rec']} {ci[a]['coverage']}"
        pc = f"{rp['rec']} {ci[a]['precision']}"
        print(f"{a:<4}{cc:>20}{pc:>20}{bk['brk']:>10}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
