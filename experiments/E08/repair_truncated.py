"""E08 truncation repair: re-draw the finish_reason=length responses at a
larger answer budget, then re-grade / re-diagnose / re-aggregate via the normal
resumable pipeline.

Why: the E08 census froze max_tokens=1536 (§0.2 reasoning-off answer budget),
but 12/75 responses hit that cap (finish_reason=length) and were cut off
mid-answer. The deterministic decode-health detector shows this is truncation,
not repetition (1 mechanical loop in the whole run), and it inflated the
degenerate_output cause (99 of 107 such labels sit inside these 12 prompts).
The pipeline's own default is max_tokens=8000 with a comment that 4000 truncates
complex answers, so 1536 was far under the house budget.

Scope of the fix: re-draw ONLY the truncated 12 at max_tokens=8192 (all other
decode params frozen: reasoning off, temperature 0.6, seed 20260715, provider
pin bf16/fp16). Same seed + a larger budget lets the truncated draw finish
rather than being a fresh sample (best-effort; OpenRouter seeds are not
guaranteed deterministic). The other 63 responses finished naturally within
1536 and are byte-identical at any higher budget, so they are left untouched;
the repaired census is therefore "all 75 at an adequate budget", with the 12
repaired records flagged (`repair`, `max_tokens`) and a manifest written.

Downstream (grade/diagnose/aggregate) is redone ONLY for the 12 via the normal
resume path: this script drops the 12 ids from the grade/diagnosis artifacts,
then `main.py grade|diagnose|aggregate` refills exactly those (done-ids are
re-derived from the output files, verified in grade.py / diagnose.py).

Usage (from repo root):
    uv run python experiments/E08/repair_truncated.py --probe        # smoke 1, no writes
    uv run python experiments/E08/repair_truncated.py --apply        # regen 12 + surgery
    # then:
    uv run python main.py grade    --experiment E08-llama3-2-3b-cc75
    uv run python main.py diagnose --experiment E08-llama3-2-3b-cc75
    uv run python main.py aggregate --experiment E08-llama3-2-3b-cc75 \
        --run-report meta/2026-07-15-e08-llama3-2-3b-cc75.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from pipeline._io import read_jsonl  # noqa: E402
from pipeline.generate import _generate_one  # noqa: E402
from pipeline.run_config import RunConfig  # noqa: E402

SLUG = "E08-llama3-2-3b-cc75"
KEY = "llama-3b"
NEW_MAX_TOKENS = 8192
DEADLINE_S = 600.0
RUN_DIR = config.RUNS_DIR / SLUG
BACKUP_DIR = RUN_DIR / "repair-truncation"


def load_cfg() -> RunConfig:
    frozen = json.loads((RUN_DIR / "experiment.json").read_text(encoding="utf-8"))["params"]
    return RunConfig.from_json_dict(SLUG, frozen)


def truncated_ids(cfg: RunConfig) -> list[str]:
    return [r["id"] for r in read_jsonl(cfg.responses_path(KEY))
            if r.get("finish_reason") == "length"]


def prompts_for(ids: list[str]) -> dict[str, str]:
    byid = {r["id"]: r for r in read_jsonl(config.DATA_JSONL)}
    return {i: byid[i]["prompt"] for i in ids}


def regenerate(cfg: RunConfig, ids: list[str]) -> dict[str, dict]:
    """Re-draw each id at NEW_MAX_TOKENS (all else frozen). Returns id -> fields."""
    cfg2 = dataclasses.replace(cfg, max_tokens=NEW_MAX_TOKENS)
    model_id = cfg2.candidates[KEY]
    prompts = prompts_for(ids)
    out: dict[str, dict] = {}

    def _one(i: str) -> tuple[str, dict]:
        return i, _generate_one(cfg2, model_id, prompts[i], deadline_s=DEADLINE_S)

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="repair") as pool:
        futs = [pool.submit(_one, i) for i in ids]
        try:
            for f in as_completed(futs):
                i, fields = f.result()
                out[i] = fields
                print(f"  {i}: finish={fields.get('finish_reason')} "
                      f"provider={fields.get('provider')} "
                      f"chars={len(fields.get('response', ''))}")
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return out


def _rewrite_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def apply(cfg: RunConfig) -> int:
    ids = truncated_ids(cfg)
    print(f"truncated responses (finish_reason=length): {len(ids)} -> {ids}")
    if not ids:
        print("nothing to repair.")
        return 0

    new = regenerate(cfg, ids)
    still = [i for i, f in new.items() if f.get("finish_reason") == "length"]
    if still:
        print(f"WARNING: still truncated at {NEW_MAX_TOKENS}: {still} "
              "(raise NEW_MAX_TOKENS and rerun)", file=sys.stderr)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # --- responses: back up, then replace the 12 records in place ---
    rp = cfg.responses_path(KEY)
    resp = read_jsonl(rp)
    shutil.copy(rp, BACKUP_DIR / "responses.pre-repair.jsonl")
    _rewrite_jsonl(BACKUP_DIR / "responses_truncated.before.jsonl",
                   [r for r in resp if r["id"] in new])
    merged = []
    for r in resp:
        if r["id"] in new:
            merged.append({"id": r["id"], **new[r["id"]],
                           "max_tokens": NEW_MAX_TOKENS,
                           "repair": "truncation-2026-07-15"})
        else:
            merged.append(r)
    _rewrite_jsonl(rp, merged)

    # --- drop the 12 from every grade file + the diagnosis file (resume refills) ---
    grades_dir = RUN_DIR / "grades"
    for jd in sorted(grades_dir.iterdir()) if grades_dir.exists() else []:
        gf = jd / f"{KEY}.jsonl"
        if gf.exists():
            shutil.copy(gf, BACKUP_DIR / f"grades_{jd.name}.pre-repair.jsonl")
            _rewrite_jsonl(gf, [r for r in read_jsonl(gf) if r["id"] not in new])
    dg = cfg.diagnosis_path(KEY)
    if dg.exists():
        shutil.copy(dg, BACKUP_DIR / "diagnosis.pre-repair.jsonl")
        _rewrite_jsonl(dg, [r for r in read_jsonl(dg) if r["id"] not in new])

    manifest = {
        "repair": "E08 truncation repair",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "reason": "12/75 responses hit finish_reason=length at frozen max_tokens=1536",
        "ids": ids,
        "new_max_tokens": NEW_MAX_TOKENS,
        "frozen_max_tokens": cfg.max_tokens,
        "held_frozen": {"reasoning": cfg.reasoning, "temperature": cfg.temperature,
                        "seed": cfg.seed, "provider_quantizations": list(cfg.provider_quantizations or ())},
        "finish_reason_after": {i: new[i].get("finish_reason") for i in ids},
        "provider_after": {i: new[i].get("provider") for i in ids},
        "still_truncated": still,
        "backups": str(BACKUP_DIR.relative_to(ROOT)),
        "next": ["main.py grade", "main.py diagnose", "main.py aggregate --run-report ..."],
    }
    (BACKUP_DIR / "repair_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\napplied: {len(new)} responses re-drawn at {NEW_MAX_TOKENS}; "
          f"grades+diagnosis rows dropped for resume; backups in {BACKUP_DIR.relative_to(ROOT)}")
    print("next: main.py grade | diagnose | aggregate --run-report meta/2026-07-15-e08-llama3-2-3b-cc75.md")
    return 1 if still else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", nargs="?", const="__first__", default=None,
                    help="Re-draw ONE truncated id (default: first) and report; no writes.")
    ap.add_argument("--apply", action="store_true",
                    help="Re-draw all truncated ids + drop downstream rows for resume.")
    args = ap.parse_args(argv)
    cfg = load_cfg()

    if args.apply:
        return apply(cfg)

    # default / --probe: smoke one, no writes
    ids = truncated_ids(cfg)
    target = ids[0] if args.probe in (None, "__first__") else args.probe
    print(f"probe (no writes): {target} at max_tokens={NEW_MAX_TOKENS}")
    regenerate(cfg, [target])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
