#!/usr/bin/env python3
"""Post-hoc DESCRIPTIVE tables for the T1.4 Tier-1 eval (NOT pre-registered).

Reads results/eval_t1_4/criteria.jsonl and prints:
  1. decode-level raw criterion pass rate per arm x cause,
  2. per-type decode-level pass rate per arm on the coverage pool,
  3. breakage Brk(arm, cause): the complement of the pre-registered recovery
     metric: the fraction of Arm-0 majority-PASS criteria that the arm now
     majority-FAILS. Rec counts base failures fixed; Brk counts base passes
     broken. Same majority-vote convention as tier1_analysis.py (pass iff
     >= 2 of 3 decodes pass).

Used for the descriptive sections of T1_3_FINAL_REPORT.md. The confirmatory
estimand lives in tier1_analysis.py; nothing here feeds H1.
"""
import collections
import json
import sys
from pathlib import Path

EVAL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
                "/workspace/failure-mode-id/results/eval_t1_4")
ARMS = ["0", "P", "SA", "SB", "RA", "RB"]
CAUSES = ["coverage", "precision"]


def load_rows():
    with open(EVAL_DIR / "criteria.jsonl") as f:
        for line in f:
            yield json.loads(line)


def main():
    tot = collections.Counter()
    ok = collections.Counter()
    votes = collections.defaultdict(list)  # (arm, cause, criterion_id) -> [pass]
    type_tot = collections.Counter()
    type_ok = collections.Counter()

    for r in load_rows():
        key = (r["arm"], r["cause_pool"])
        tot[key] += 1
        ok[key] += bool(r["pass"])
        votes[(r["arm"], r["cause_pool"], r["criterion_id"])].append(bool(r["pass"]))
        if r["cause_pool"] == "coverage":
            ctype = r["criterion_id"].split(":")[-1]
            type_tot[(r["arm"], ctype)] += 1
            type_ok[(r["arm"], ctype)] += bool(r["pass"])

    print("== 1. Raw decode-level criterion pass rate ==")
    print(f"{'arm':<4}" + "".join(f"{c:>11}" for c in CAUSES))
    for arm in ARMS:
        row = [ok[(arm, c)] / tot[(arm, c)] for c in CAUSES]
        print(f"{arm:<4}" + "".join(f"{v:>11.4f}" for v in row)
              + f"   (n={tot[(arm, CAUSES[0])]}/{tot[(arm, CAUSES[1])]})")

    print("\n== 2. Per-type decode-level pass rate (coverage pool) ==")
    types = sorted({t for (_, t) in type_tot})
    print(f"{'type':<20}" + "".join(f"{a:>8}" for a in ARMS))
    for t in types:
        row = [type_ok[(a, t)] / type_tot[(a, t)] for a in ARMS]
        print(f"{t:<20}" + "".join(f"{v:>8.3f}" for v in row))

    # majority vote per criterion
    maj = {k: sum(v) * 2 > len(v) for k, v in votes.items()}
    base_pass = {c: {cid for (a, cc, cid) in maj if a == "0" and cc == c and maj[(a, cc, cid)]}
                 for c in CAUSES}
    base_fail = {c: {cid for (a, cc, cid) in maj if a == "0" and cc == c and not maj[(a, cc, cid)]}
                 for c in CAUSES}

    print("\n== 3. Recovery vs breakage (majority-vote, criterion-level) ==")
    print(f"{'arm':<4}{'cause':<11}{'Rec':>8}{'Brk':>8}{'net':>8}"
          f"   (base-fail n / base-pass n)")
    for arm in ARMS[1:]:
        for c in CAUSES:
            rec = sum(maj[(arm, c, cid)] for cid in base_fail[c]) / len(base_fail[c])
            brk = sum(not maj[(arm, c, cid)] for cid in base_pass[c]) / len(base_pass[c])
            net = rec * len(base_fail[c]) - brk * len(base_pass[c])
            print(f"{arm:<4}{c:<11}{rec:>8.4f}{brk:>8.4f}{net:>+8.0f}"
                  f"   ({len(base_fail[c])} / {len(base_pass[c])})")


if __name__ == "__main__":
    main()
