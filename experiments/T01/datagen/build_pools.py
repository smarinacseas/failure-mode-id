"""Build the T01 train/holdout prompt pools (T1.2, plan step 2-3). Per cause:
300 train + 200 holdout unique prompts, screened for external contamination and
split disjointly.

Contamination policy (see the module test + PREREG §5 holdout rule):
  * 13-gram screen vs EXTERNAL eval sets — CC-75 (Tier-2) and IFEval originals
    (Tier-3) — so training prompts never leak into what measures transfer /
    regression. Composed prompts are templated differently, so real hits are
    rare; the screen is logged either way.
  * train/holdout leakage is prevented by EXACT-PROMPT uniqueness, NOT a 13-gram
    screen between our own prompts: templated instruction boilerplate (e.g. the
    fixed no-placeholders sentence) makes an intra-set 13-gram screen over-reject
    on shared constraint phrasings, which is a false positive for leakage — the
    task differs even when one constraint sentence repeats.

Run:  uv run python experiments/T01/datagen/build_pools.py
Writes data/{train,holdout}/{coverage,precision}.jsonl.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
T01_DIR = _HERE.parents[1]
REPO_ROOT = _HERE.parents[3]
# Runnable as a script: put datagen/ + verifiers/ on sys.path (pytest uses conftest).
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(T01_DIR / "verifiers"))

from compose import compose_prompt  # noqa: E402
from contamination import load_reference_ngrams, shares_ngram  # noqa: E402

SPLIT_SEED = 20260715          # PREREG §7
N_TRAIN, N_HOLD = 300, 200
_CAUSE_SEED = {"coverage": SPLIT_SEED, "precision": SPLIT_SEED + 1}


def build_pool(cause: str, n_train: int, n_hold: int, seed: int,
               external_ngrams=frozenset()) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    need = n_train + n_hold
    seen_prompts: set[str] = set()
    accepted: list[dict] = []
    n_dup = n_contam = guard = 0
    while len(accepted) < need:
        guard += 1
        if guard > need * 300:
            raise RuntimeError(
                f"{cause}: only {len(accepted)}/{need} unique prompts after "
                f"{guard} tries — expand the content banks in constraints.py")
        rec = compose_prompt(cause, rng, None)
        if rec["prompt"] in seen_prompts:
            n_dup += 1
            continue
        if external_ngrams and shares_ngram(rec["prompt"], external_ngrams):
            n_contam += 1
            continue
        seen_prompts.add(rec["prompt"])
        accepted.append(rec)

    train, hold = accepted[:n_train], accepted[n_train:]
    for split, recs in (("train", train), ("holdout", hold)):
        tag = f"{cause[:3].upper()}-{split[0].upper()}"
        for j, r in enumerate(recs):
            r["split"] = split
            r["id"] = f"{tag}-{j + 1:03d}"
    print(f"  {cause}: {len(train)}+{len(hold)} accepted "
          f"({n_dup} exact-dups, {n_contam} external-contam skipped, {guard} draws)")
    return train, hold


def _write_jsonl(recs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    cc75 = REPO_ROOT / "data" / "complexconstraints.jsonl"
    ifeval_dir = REPO_ROOT / "data" / "verih" / "Eval" / "evals" / "ifeval" / "inputs"
    ifeval = sorted(ifeval_dir.glob("*.jsonl")) if ifeval_dir.exists() else []
    external, report = load_reference_ngrams([cc75, *ifeval])
    print("external contamination screen:")
    for line in report:
        print(f"  {line}")

    for cause, seed in _CAUSE_SEED.items():
        train, hold = build_pool(cause, N_TRAIN, N_HOLD, seed, external)
        _write_jsonl(train, T01_DIR / "data" / "train" / f"{cause}.jsonl")
        _write_jsonl(hold, T01_DIR / "data" / "holdout" / f"{cause}.jsonl")
    print(f"wrote pools under {T01_DIR / 'data'}/{{train,holdout}}/")


if __name__ == "__main__":
    main()
