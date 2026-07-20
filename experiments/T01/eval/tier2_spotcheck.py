"""Tier-2 STEP-2 stack-parity spot-check — compares a LOCAL Arm-0 re-decode
(single-judge opus-4.8, k=3 majority vote) against E08's ORIGINAL OpenRouter
Arm-0 panel labels, on the same subset prompts/criteria.

The disagreement it measures is the COMBINED "before-state reproduction error":
stack (local HF vs OpenRouter) + grader (single opus vs 3-judge panel) + k
(3-vote vs E08's k=1). That combination is exactly the bias that will sit on the
reused-baseline recovery metric (trained arms are graded single-opus/k=3 against a
panel/k=1-labeled base), so this is the right quantity to caveat with.

Two numbers reported:
  1. Agreement rate  — over all subset criteria with both an E08 base verdict and a
     local grade: fraction where local pass/fail == E08 base pass/fail (+ Cohen's κ).
  2. Arm-0 FALSE-RECOVERY — of the coverage/precision base-FAILED criteria (E08 says
     FAIL), the fraction the LOCAL untrained Arm 0 now PASSES. This is the recovery
     floor: a trained arm's recovery is only credible above this stack+grader floor.

Run (after tier2_decode --arm 0 --subset and tier2_grade --arm 0 --subset):
    python3 experiments/T01/eval/tier2_spotcheck.py
"""
from __future__ import annotations

import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parent))
from tier2_labels import load_e08, select_subset   # noqa: E402

GRADES = _REPO / "results" / "eval_t2" / "grades" / "0_subset.jsonl"
OUT = _REPO / "results" / "eval_t2" / "spotcheck.json"


def majority_local() -> dict:
    """(pid, idx0) -> local pass (majority over k decodes)."""
    votes: dict[tuple[str, int], list[bool]] = {}
    with GRADES.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                votes.setdefault((d["prompt_id"], d["idx0"]), []).append(bool(d["pass"]))
    return {k: (sum(v) > len(v) / 2) for k, v in votes.items()}


def cohens_kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def main() -> None:
    base, labeled = load_e08()
    subset = set(select_subset())
    local = majority_local()

    # (1) overall agreement on all subset criteria with both verdicts
    pairs = []
    for (pid, idx0), b in base.items():
        if pid not in subset or b["base_pass"] is None:
            continue
        if (pid, idx0) in local:
            pairs.append((bool(local[(pid, idx0)]), bool(b["base_pass"])))
    agree = sum(a == c for a, c in pairs) / len(pairs) if pairs else float("nan")

    # (2) Arm-0 false-recovery on coverage/precision base-FAILED criteria in subset
    fr = {"coverage": [0, 0], "precision": [0, 0]}   # [recovered_by_local_arm0, total]
    for (pid, idx0), cause in labeled.items():
        if pid not in subset or (pid, idx0) not in local:
            continue
        fr[cause][1] += 1
        if local[(pid, idx0)]:                       # local Arm 0 now PASSES an E08 base-FAIL
            fr[cause][0] += 1

    out = {
        "subset_n_prompts": len(subset),
        "graded_criteria": len(pairs),
        "agreement_rate": round(agree, 4),
        "cohens_kappa": round(cohens_kappa(pairs), 4),
        "local_arm0_pass_rate": round(sum(a for a, _ in pairs) / len(pairs), 4) if pairs else None,
        "e08_base_pass_rate": round(sum(c for _, c in pairs) / len(pairs), 4) if pairs else None,
        "false_recovery": {
            c: {"recovered": fr[c][0], "of_base_failed": fr[c][1],
                "rate": round(fr[c][0] / fr[c][1], 4) if fr[c][1] else None}
            for c in ("coverage", "precision")},
        "judge": "opus-4.8 (single) vs E08 3-judge panel; local HF k=3 vs OpenRouter k=1",
    }
    OUT.write_text(json.dumps(out, indent=2))

    print("=== Tier-2 STEP-2 stack-parity spot-check ===")
    print(f"subset: {out['subset_n_prompts']} prompts, {out['graded_criteria']} graded criteria")
    print(f"agreement (local-opus vs E08-panel base): {agree*100:.1f}%  (Cohen κ {out['cohens_kappa']})")
    print(f"  local Arm-0 pass rate {out['local_arm0_pass_rate']*100:.1f}%  vs  "
          f"E08 base pass rate {out['e08_base_pass_rate']*100:.1f}%")
    print("Arm-0 FALSE-RECOVERY on base-FAILED criteria (the recovery floor):")
    for c in ("coverage", "precision"):
        f = out["false_recovery"][c]
        print(f"  {c:9s}: {f['recovered']}/{f['of_base_failed']} = "
              f"{(f['rate'] or 0)*100:.1f}% of E08 base-fails now pass locally")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
