"""Tier-2 CC-75 grading — grades an arm's CC-75 responses against the E08 criteria
with a single judge (opus-4.8), REUSING E08's exact judge protocol: the same
`prompts/judge.txt` system prompt, the same numbered user message, and the same
parser (`pipeline.grade._grade_one` → `_judge_llm.call_json`). Only the panel is
reduced to one anchor judge (opus48 = E08's tie-breaker) — the descriptive Tier-2
grading regime (real API cost, so single-judge; see RUN_TIER2.md).

Writes one verdict row per (prompt_id, idx0, decode_idx) to
results/eval_t2/grades/{tag}.jsonl. Resumable at (prompt_id, decode_idx) grain.

Run (after tier2_decode.py for the same tag):
    python3 experiments/T01/eval/tier2_grade.py --arm 0 --subset
    python3 experiments/T01/eval/tier2_grade.py --arm SA
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = pathlib.Path(__file__).resolve()
_T01 = _HERE.parents[1]
_REPO = _T01.parents[1]
sys.path.insert(0, str(_REPO))                 # pipeline / config
sys.path.insert(0, str(_HERE.parent))          # tier2_labels

from pipeline.grade import _grade_one           # noqa: E402  (E08's exact grading call)
from pipeline.run_config import JudgeSpec       # noqa: E402
from tier2_labels import load_corpus            # noqa: E402

JUDGE = JudgeSpec(key="opus48", client="openrouter", model="anthropic/claude-opus-4.8")
RESP_DIR = _REPO / "results" / "eval_t2" / "responses"
OUT_DIR = _REPO / "results" / "eval_t2" / "grades"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    corpus = load_corpus()
    tag = args.arm + ("_subset" if args.subset else "")
    resp_path = RESP_DIR / f"{tag}.jsonl"
    if not resp_path.exists():
        raise SystemExit(f"no responses at {resp_path} — run tier2_decode.py --arm {args.arm} first")
    responses = [json.loads(l) for l in resp_path.open(encoding="utf-8") if l.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tag}.jsonl"
    done: set[tuple[str, int]] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done.add((d["prompt_id"], d["decode_idx"]))
        # a (prompt,decode) is "done" only if all its criteria rows are present;
        # track by presence of any row and re-grade nothing already written.
        print(f"[resume] {len(done)} (prompt,decode) cells already graded in {out_path.name}")

    todo = [r for r in responses if (r["prompt_id"], r["decode_idx"]) not in done]
    print(f"[grade] {tag}: {len(todo)} (prompt,decode) responses to grade "
          f"(judge {JUDGE.model}, {args.workers} workers)")

    def grade_cell(r: dict) -> list[dict]:
        criteria = corpus[r["prompt_id"]]["criteria"]
        verdicts = _grade_one(JUDGE, corpus[r["prompt_id"]]["prompt"], r["response"], criteria)
        rows = []
        for v in verdicts:
            idx0 = int(v["index"]) - 1                       # judge emits 1-based
            rows.append({
                "arm": args.arm, "prompt_id": r["prompt_id"], "idx0": idx0,
                "decode_idx": r["decode_idx"],
                "pass": v["verdict"] == "PASS", "reason": v.get("reason", ""),
            })
        return rows

    t0 = time.time()
    n_cells = 0
    with out_path.open("a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(grade_cell, r): r for r in todo}
            for fut in as_completed(futs):
                for row in fut.result():
                    out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                n_cells += 1
                if n_cells % 5 == 0 or n_cells == len(todo):
                    print(f"[grade {tag}] {n_cells}/{len(todo)} cells, {time.time()-t0:.0f}s", flush=True)

    print(f"[done] graded {n_cells} new (prompt,decode) cells for {tag} -> {out_path.name}")


if __name__ == "__main__":
    main()
