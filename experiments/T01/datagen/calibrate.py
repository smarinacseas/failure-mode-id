"""T1.2 difficulty calibration (plan step 1). Probe base Llama-3.2-3B on a small
sample of composed prompts via OpenRouter, grade constraints with the verifier
library, and report the criterion-pass distribution vs the 30-70% target band.

The band is what leaves headroom to measure recovery: too easy (>70%) => the arms
have little to fix; too hard (<30%) => the reward is too sparse. If we're out of
band, tune the constraint counts in compose.py and regenerate before spending on
SFT teacher data.

~20 calls, ~$1. OpenRouter is a proxy for the local base model the T1.3 arms
train from — a small serving-stack caveat to note when reading the number.

Run:  uv run python experiments/T01/datagen/calibrate.py
"""
from __future__ import annotations

import json
import os
import random
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
MODEL = "meta-llama/llama-3.2-3b-instruct"
N_PER_CAUSE = 10
SAMPLE_SEED = 20260715
# No provider pin here: the bf16/fp16 route (Parasail) is rate-limited on the free
# tier, so for this rough difficulty probe we let OpenRouter route to any provider.
# (T1.3 training runs the base model locally anyway; this is only a proxy.)
EXTRA_BODY = {"reasoning": {"enabled": False}}


def load_sample() -> list[dict]:
    rng = random.Random(SAMPLE_SEED)
    sample = []
    for cause in ("coverage", "precision"):
        recs = [json.loads(l) for l in
                open(T01_DIR / "data" / "train" / f"{cause}.jsonl", encoding="utf-8")]
        sample += rng.sample(recs, N_PER_CAUSE)
    return sample


def grade(rec: dict, response: str) -> tuple[int, int]:
    passed = sum(check(response, s).passed for s in rec["specs"])
    return passed, len(rec["specs"])


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set (expected in .env)")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    sample = load_sample()
    print(f"calibration: {len(sample)} prompts, base {MODEL} via OpenRouter\n")

    def _one(rec):
        last = None
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, temperature=0.6, max_tokens=2048,
                    messages=[{"role": "user", "content": rec["prompt"]}],
                    extra_body=EXTRA_BODY, timeout=180)
                passed, total = grade(rec, resp.choices[0].message.content or "")
                return rec, passed, total, None
            except Exception as e:  # noqa: BLE001 — retry transient/rate-limit errors
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt + random.random())  # ~1,2,4,8,16s backoff
        return rec, 0, len(rec["specs"]), last

    by_cause: dict[str, list[float]] = {"coverage": [], "precision": []}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one, r) for r in sample]
        for f in as_completed(futures):
            rec, passed, total, err = f.result()
            if err:
                print(f"  ERR  {rec['id']:9s} {err}")
                continue
            rate = passed / total
            by_cause[rec["cause"]].append(rate)
            print(f"  {rec['id']:9s} {rec['cause']:9s} {passed}/{total} = {rate * 100:5.1f}%")

    print()
    allrates: list[float] = []
    for cause, rates in by_cause.items():
        if not rates:
            continue
        allrates += rates
        m = 100 * sum(rates) / len(rates)
        verdict = "IN BAND" if 30 <= m <= 70 else ("TOO EASY (>70)" if m > 70 else "TOO HARD (<30)")
        print(f"{cause:10s} mean criterion-pass {m:5.1f}%  [{verdict}]  (n={len(rates)})")
    if allrates:
        print(f"{'OVERALL':10s} {100 * sum(allrates) / len(allrates):5.1f}%   target band 30-70%")


if __name__ == "__main__":
    main()
