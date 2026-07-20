"""T3 aggregate — build the H3 verdict from the per-arm MMLU summaries.

Reads results/eval_t3/{arm}_summary.json for all 6 arms, computes each arm's
accuracy and its delta (in percentage points) vs Arm 0, and applies the
pre-registered guard (PREREG §4, H3):

    FAIL iff any *trained* arm (SA, SB, RA, RB) scores > 3.0 points below Arm 0.

Arm P is reported as a reference but is NOT gated — it is a $0 system prompt on
the base weights, not a trained arm (PREREG §4 names only "trained arm").

Writes results/eval_t3/verdict.json and prints the table.
"""
from __future__ import annotations

import json
import pathlib

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
OUT_DIR = _REPO / "results" / "eval_t3"
ARMS = ("0", "P", "SA", "SB", "RA", "RB")
TRAINED_ARMS = ("SA", "SB", "RA", "RB")
THRESHOLD_PP = 3.0   # PREREG §4, H3: "more than 3 points below Arm 0"


def main() -> None:
    summaries = {}
    for arm in ARMS:
        p = OUT_DIR / f"{arm}_summary.json"
        if not p.exists():
            raise SystemExit(f"missing summary for arm {arm}: {p}")
        summaries[arm] = json.loads(p.read_text())

    base_acc = summaries["0"]["accuracy"]
    table = {}
    for arm in ARMS:
        acc = summaries[arm]["accuracy"]
        delta_pp = round((acc - base_acc) * 100, 2)
        table[arm] = {
            "n": summaries[arm]["n"],
            "n_correct": summaries[arm]["n_correct"],
            "accuracy": acc,
            "acc_pct": round(acc * 100, 2),
            "delta_pp_vs_arm0": delta_pp,
            "trained": arm in TRAINED_ARMS,
            "gated": arm in TRAINED_ARMS,
        }

    violations = [
        {"arm": arm, "delta_pp": table[arm]["delta_pp_vs_arm0"]}
        for arm in TRAINED_ARMS
        if table[arm]["delta_pp_vs_arm0"] < -THRESHOLD_PP
    ]
    verdict = "FAIL" if violations else "PASS"

    out = {
        "battery": "mmlu-0shot-letterloglik",
        "subset_n": summaries["0"]["subset_n"],
        "subset_seed": summaries["0"]["subset_seed"],
        "threshold_pp": THRESHOLD_PP,
        "rule": "FAIL iff any trained arm (SA,SB,RA,RB) > 3.0 pp below Arm 0 (PREREG §4, H3)",
        "arm0_accuracy": base_acc,
        "table": table,
        "violations": violations,
        "verdict": verdict,
    }
    (OUT_DIR / "verdict.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== H3 Tier-3 regression guard — {verdict} ===")
    print(f"battery: MMLU 0-shot letter-loglik, N={out['subset_n']}, seed {out['subset_seed']}")
    print(f"Arm 0 accuracy: {base_acc * 100:.2f}%\n")
    print(f"{'arm':<4} {'acc%':>7} {'Δpp vs 0':>9}  {'role':<10} {'gated':<6}")
    for arm in ARMS:
        t = table[arm]
        role = "base" if arm == "0" else ("prompt-ref" if arm == "P" else "trained")
        flag = "  <-- VIOLATION" if any(v["arm"] == arm for v in violations) else ""
        print(f"{arm:<4} {t['acc_pct']:>7.2f} {t['delta_pp_vs_arm0']:>+9.2f}  "
              f"{role:<10} {'yes' if t['gated'] else 'no':<6}{flag}")
    print(f"\nverdict: {verdict}"
          + ("" if verdict == "PASS" else f"  ({len(violations)} violation(s))"))


if __name__ == "__main__":
    main()
