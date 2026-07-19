"""SFT parity down-sampling (per PREREG amendment DRAFT 2026-07-16 — sft-parity;
see experiments/T01/config/DRAFT_amendment_2026-07-16_sft-parity.md).

Down-sample each SFT accepted-pair set (SA=coverage, SB=precision) to
target_n = min accepted-count across causes, so both SFT cells train on equal n
and the accepted-count asymmetry (coverage 204 vs precision 270) is not a
confound. GRPO prompt pools (300/cause), all holdout data, and the 2-epoch
schedule are UNTOUCHED. Deterministic: sort ids, then
random.Random(20260715).sample (repo convention, cf. calibrate.py). Identity for
the min-count cell (coverage: sample 204 of 204 = the whole set).

Materializes ids-only manifests under data/sft_manifests/ (tracked; never any
response text). The SFT runner filters data/sft/{cause}.jsonl to the manifest ids.

Run:  python experiments/T01/datagen/downsample_sft.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20260715
T01 = Path(__file__).resolve().parents[1]
SFT_DIR = T01 / "data" / "sft"
MANIFEST_DIR = T01 / "data" / "sft_manifests"
ARM_CAUSE = {"SA": "coverage", "SB": "precision"}
TRAIN_PROMPTS_PER_CAUSE = 300   # datagen holds 300 prompts fixed per cause


def accepted_ids(cause: str) -> list[str]:
    return [json.loads(l)["id"] for l in (SFT_DIR / f"{cause}.jsonl").open(encoding="utf-8")]


def _select(ids: list[str], n: int, seed: int = SEED) -> list[str]:
    """Sort BEFORE sampling so the pick is invariant to input order, then return
    the selection sorted for a stable manifest."""
    return sorted(random.Random(seed).sample(sorted(ids), n))


def select_ids(cause: str, n: int, seed: int = SEED) -> list[str]:
    return _select(accepted_ids(cause), n, seed)


def target_n() -> int:
    return min(len(accepted_ids(c)) for c in ARM_CAUSE.values())


def source_stats(cause: str) -> dict:
    acc = len(accepted_ids(cause))
    return {"accepted": acc, "train_prompts": TRAIN_PROMPTS_PER_CAUSE,
            "yield_pct": round(100 * acc / TRAIN_PROMPTS_PER_CAUSE, 1)}


def build_manifest(arm: str, n: int, seed: int = SEED) -> dict:
    cause = ARM_CAUSE[arm]
    return {"arm": arm, "cause": cause, "target_n": n, "seed": seed,
            "source": source_stats(cause), "n_selected": n,
            "selected_ids": select_ids(cause, n, seed)}


def write_manifests(seed: int = SEED) -> dict[str, Path]:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    n = target_n()
    out = {}
    for arm in ARM_CAUSE:
        p = MANIFEST_DIR / f"{arm}.json"
        p.write_text(json.dumps(build_manifest(arm, n, seed), indent=2) + "\n", encoding="utf-8")
        out[arm] = p
    return out


def load_manifest_ids(arm: str) -> list[str]:
    return json.loads((MANIFEST_DIR / f"{arm}.json").read_text(encoding="utf-8"))["selected_ids"]


if __name__ == "__main__":
    n = target_n()
    print(f"target_n = min accepted across causes = {n}")
    for arm, p in write_manifests().items():
        m = json.loads(p.read_text())
        print(f"  {arm} ({m['cause']}): {m['n_selected']} of {m['source']['accepted']} accepted "
              f"(yield {m['source']['yield_pct']}%) -> {p.relative_to(T01.parents[1])}")
