"""T1.4 Tier-1 grader (PREREG §5).

Joins decode outputs (results/eval_t1_4/decodes/*.jsonl) with the holdout specs
and emits one per-criterion record per (arm, prompt, decode, criterion):

    {arm, prompt_id, decode_idx, cause_pool, criterion_id, pass}

exactly the record schema PREREG §5 pre-registers. criterion_id is
"<prompt_id>#c<idx>:<type>" — the spec's position in the prompt's frozen spec
list (stable across arms, so criterion-level pairing vs Arm 0 is exact).

Grading follows the training-time convention (reward_adapter): the gradeable
answer is extract_final(completion) — text after the last ===FINAL=== marker,
or the whole completion when the marker is absent (Arm 0 has no scaffold; its
whole response is the answer, same as the T1.2 calibration regime). Each spec
is checked by the registered verifier (verifiers.base.check). No length cap or
malformed penalty at eval — those are training-reward anti-gaming terms, not
part of the §5 criterion-pass estimand; answer length and malformed flags are
emitted to a side table instead (the §8 verbosity-artifact tell).

Run:  python3 experiments/T01/eval/grade_tier1.py [--decode-dir ...] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve()
_T01 = _HERE.parents[1]
_REPO = _T01.parents[1]
sys.path.insert(0, str(_T01 / "training"))    # common.extract_final
sys.path.insert(0, str(_T01 / "verifiers"))   # base.check + pool registration

from common import extract_final, FINAL_MARKER  # noqa: E402
from reward import constraint_reward  # noqa: E402,F401 — import registers verifier pools
from base import check  # noqa: E402

EVAL_DIR = _REPO / "results" / "eval_t1_4"


def load_specs() -> dict[str, dict]:
    """prompt_id -> holdout record (specs list frozen at T1.2)."""
    recs: dict[str, dict] = {}
    for cause in ("coverage", "precision"):
        path = _T01 / "data" / "holdout" / f"{cause}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[r["id"]] = r
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-dir", default=str(EVAL_DIR / "decodes"))
    ap.add_argument("--out", default=str(EVAL_DIR / "criteria.jsonl"))
    ap.add_argument("--decode-meta-out", default=str(EVAL_DIR / "decode_meta.jsonl"))
    args = ap.parse_args()

    specs_by_id = load_specs()
    decode_dir = pathlib.Path(args.decode_dir)
    files = sorted(decode_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"no decode files in {decode_dir}")

    n_crit = 0
    n_dec = 0
    with open(args.out, "w", encoding="utf-8") as crit_f, \
         open(args.decode_meta_out, "w", encoding="utf-8") as meta_f:
        for path in files:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    rec = specs_by_id[d["prompt_id"]]
                    answer = extract_final(d["completion"])
                    meta_f.write(json.dumps({
                        "arm": d["arm"], "prompt_id": d["prompt_id"],
                        "decode_idx": d["decode_idx"], "cause_pool": d["cause"],
                        "answer_chars": len(answer),
                        "completion_chars": len(d["completion"]),
                        "has_final_marker": FINAL_MARKER in d["completion"],
                        "malformed_empty": not answer.strip(),
                        "len_capped": d.get("len_capped", False),
                        "gen_tokens": d.get("gen_tokens"),
                    }) + "\n")
                    n_dec += 1
                    for idx, spec in enumerate(rec["specs"]):
                        crit_f.write(json.dumps({
                            "arm": d["arm"],
                            "prompt_id": d["prompt_id"],
                            "decode_idx": d["decode_idx"],
                            "cause_pool": d["cause"],
                            "criterion_id": f"{d['prompt_id']}#c{idx:02d}:{spec['type']}",
                            "pass": bool(check(answer, spec).passed),
                        }) + "\n")
                        n_crit += 1
            print(f"[graded] {path.name}", flush=True)

    print(f"[done] {n_dec} decodes -> {n_crit} criterion records -> {args.out}")


if __name__ == "__main__":
    main()
