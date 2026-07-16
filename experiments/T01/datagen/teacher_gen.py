"""SFT teacher generation (T1.2 step 4). For each train prompt, up to 4 attempts
with a strong teacher, prompted in the archetype's reasoning format:
  * coverage  — inventory every requirement -> draft -> checklist-verify
  * precision — work out exact counts/arithmetic -> draft -> reconcile
The teacher marks its finished answer after a line `===FINAL===`; the verifier
grades that answer (not the scaffold). Keep responses whose answer 100%-passes.
Record per-cell token counts. Gate GT2: yield >= 30%.

The full completion (scaffold + ===FINAL=== + answer) is the SFT target — the
student learns the enumerate/compute-then-verify BEHAVIOUR, which is exactly the
trainable signal each archetype targets. Tier-1 eval + the GRPO reward extract
the post-===FINAL=== answer the same way (see extract_final).

Run:  uv run python experiments/T01/datagen/teacher_gen.py [--limit N] [--cause C]
Writes data/sft/{coverage,precision}.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "verifiers"))

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

import archetypes  # noqa: E402,F401 — registers every verifier
from base import check  # noqa: E402

REPO_ROOT = _HERE.parents[3]
T01_DIR = _HERE.parents[1]
TEACHER = "anthropic/claude-fable-5"
MAX_ATTEMPTS = 4
FINAL_MARKER = "===FINAL==="

SYSTEM = {
    "coverage": (
        "You are completing a writing task that has many separate requirements. "
        "Work in this exact order:\n"
        "1. INVENTORY: list every requirement the task states, as a checklist.\n"
        "2. DRAFT: write a response that tries to satisfy all of them.\n"
        "3. VERIFY: go through your checklist item by item against the draft and "
        "fix every miss.\n"
        f"4. Output ONLY the finished response on the lines after a line "
        f"containing exactly {FINAL_MARKER} (no commentary after it)."),
    "precision": (
        "You are completing a writing task with exact, checkable requirements. "
        "Work in this exact order:\n"
        "1. COMPUTE: work out every exact count, arithmetic result, or ordering "
        "the task demands.\n"
        "2. DRAFT: write a response built to those exact numbers.\n"
        "3. RECONCILE: recount/recheck the draft against each requirement and fix "
        "any mismatch.\n"
        f"4. Output ONLY the finished response on the lines after a line "
        f"containing exactly {FINAL_MARKER} (no commentary after it)."),
}


def extract_final(text: str) -> str:
    """The gradeable answer: everything after the last ===FINAL=== marker, or the
    whole text if the marker is absent (a malformed attempt graded as-is)."""
    return text.rsplit(FINAL_MARKER, 1)[-1].strip() if FINAL_MARKER in text else text.strip()


def _all_pass(rec: dict, answer: str) -> bool:
    return all(check(answer, s).passed for s in rec["specs"])


def teach_one(client: OpenAI, rec: dict) -> tuple[dict | None, int, dict]:
    """Return (sft_record_or_None, attempts_used, token_totals)."""
    tokens = {"prompt": 0, "completion": 0}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(
                model=TEACHER, temperature=0.7, max_tokens=4096,
                messages=[{"role": "system", "content": SYSTEM[rec["cause"]]},
                          {"role": "user", "content": rec["prompt"]}],
                timeout=180)
        except Exception:  # noqa: BLE001 — transient/rate-limit; back off and retry
            time.sleep(2 ** attempt)
            continue
        u = resp.usage
        tokens["prompt"] += u.prompt_tokens
        tokens["completion"] += u.completion_tokens
        completion = resp.choices[0].message.content or ""
        answer = extract_final(completion)
        if _all_pass(rec, answer):
            return ({"id": rec["id"], "cause": rec["cause"], "prompt": rec["prompt"],
                     "specs": rec["specs"], "completion": completion, "answer": answer,
                     "attempts": attempt}, attempt, tokens)
    return None, MAX_ATTEMPTS, tokens


def run_cause(client: OpenAI, cause: str, limit: int | None) -> dict:
    recs = [json.loads(l) for l in
            open(T01_DIR / "data" / "train" / f"{cause}.jsonl", encoding="utf-8")]
    if limit:
        recs = recs[:limit]
    kept, tok_p, tok_c, attempts = [], 0, 0, 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(teach_one, client, r): r for r in recs}
        for f in as_completed(futures):
            sft, used, tokens = f.result()
            tok_p += tokens["prompt"]
            tok_c += tokens["completion"]
            attempts += used
            if sft:
                kept.append(sft)
    yield_pct = 100 * len(kept) / len(recs) if recs else 0.0
    if not limit:
        out = T01_DIR / "data" / "sft" / f"{cause}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        kept.sort(key=lambda r: r["id"])
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                       encoding="utf-8")
    print(f"{cause:10s} kept {len(kept)}/{len(recs)}  yield {yield_pct:5.1f}%  "
          f"(>=30% GT2)  attempts~{attempts / len(recs):.1f}/prompt  "
          f"tokens: {tok_p}p + {tok_c}c")
    return {"cause": cause, "kept": len(kept), "n": len(recs), "yield": yield_pct,
            "prompt_tokens": tok_p, "completion_tokens": tok_c}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke: only the first N prompts per cause (no file written).")
    ap.add_argument("--cause", choices=("coverage", "precision"), default=None)
    args = ap.parse_args(argv)

    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set (expected in .env)")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    causes = [args.cause] if args.cause else ["coverage", "precision"]
    mode = f"SMOKE limit={args.limit}" if args.limit else "FULL"
    print(f"teacher-gen [{mode}] — teacher {TEACHER}\n")
    for cause in causes:
        run_cause(client, cause, args.limit)


if __name__ == "__main__":
    main()
