"""Judge-validation: sample 60 rows for the human, then score agreement.

Two subcommands:
  - sample : write runs/<slug>/judge_validation.json with 60 fixed-seed random
            (model, id, criterion) rows; human fills the `human` field.
  - score  : read the filled file, compute agreement %, list disagreements,
            and merge `judge_agreement` into run_manifest.json.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone

from config import (
    DATA_JSONL,
    VALIDATE_RESPONSE_EXCERPT_CHARS,
    VALIDATE_SAMPLE_TARGET,
    VALIDATE_SEED,
)
from pipeline._io import read_jsonl
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import RunConfig


def _build_pool(cfg: RunConfig) -> list[dict]:
    records = read_jsonl(DATA_JSONL)
    by_id = {r["id"]: r for r in records}
    pool: list[dict] = []
    for key in cfg.candidates:
        grades = read_jsonl(cfg.grades_path(key))
        responses = {r["id"]: r["response"] for r in read_jsonl(cfg.responses_path(key))}
        for g in grades:
            rec = by_id.get(g["id"])
            if not rec:
                continue
            criteria = rec["criteria"]
            for v in g["verdicts"]:
                idx = int(v["index"])
                if not 1 <= idx <= len(criteria):
                    continue
                # Skip judge errors — they aren't a real verdict to grade against.
                if v.get("reason", "").startswith("judge_parse_error"):
                    continue
                pool.append({
                    "model": key,
                    "id": g["id"],
                    "criterion_index": idx,
                    "criterion_text": criteria[idx - 1],
                    "prompt_text": rec["prompt"],
                    "response_excerpt": (responses.get(g["id"], "") or "")[:VALIDATE_RESPONSE_EXCERPT_CHARS],
                    "judge_verdict": v["verdict"],
                    "judge_reason": v["reason"],
                    "human": "",
                })
    return pool


def sample(cfg: RunConfig, mon: "RunMonitor | None" = None) -> None:
    pool = _build_pool(cfg)
    if not pool:
        raise RuntimeError("No graded rows found. Run `grade` first.")
    n = min(VALIDATE_SAMPLE_TARGET, len(pool))
    rng = random.Random(VALIDATE_SEED)
    rows = rng.sample(pool, n)
    cfg.judge_validation_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.judge_validation_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    msg = (f"validate sample: wrote {n} rows → {cfg.judge_validation_path}; "
           f"fill the `human` field (PASS/FAIL), then run `validate --mode score`.")
    mon.note(msg) if mon is not None else print(msg)


def _merge_manifest(cfg: RunConfig, update: dict) -> None:
    existing: dict = {}
    if cfg.run_manifest_path.exists():
        try:
            existing = json.loads(cfg.run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(update)
    cfg.run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.run_manifest_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def score(cfg: RunConfig) -> None:
    if not cfg.judge_validation_path.exists():
        raise RuntimeError(
            f"{cfg.judge_validation_path} missing. Run `validate --mode sample` first."
        )
    rows = json.loads(cfg.judge_validation_path.read_text(encoding="utf-8"))
    filled = [r for r in rows if str(r.get("human", "")).strip().upper() in {"PASS", "FAIL"}]
    if not filled:
        raise RuntimeError(
            "No rows have a human verdict filled in. Edit the file and try again."
        )

    agree = 0
    disagreements: list[dict] = []
    for r in filled:
        human = str(r["human"]).strip().upper()
        judge = str(r["judge_verdict"]).strip().upper()
        if human == judge:
            agree += 1
        else:
            disagreements.append({
                "model": r["model"],
                "id": r["id"],
                "criterion_index": r["criterion_index"],
                "criterion_text": r["criterion_text"],
                "judge_verdict": judge,
                "judge_reason": r["judge_reason"],
                "human_verdict": human,
            })

    n = len(filled)
    pct = 100.0 * agree / n
    print(f"validate score: {agree}/{n} agree ({pct:.1f}%)")
    if disagreements:
        print(f"  {len(disagreements)} disagreement(s):")
        for d in disagreements:
            print(
                f"  · {d['model']} {d['id']} c{d['criterion_index']}: "
                f"judge={d['judge_verdict']} vs human={d['human_verdict']} "
                f"— {d['criterion_text'][:80]}"
            )

    _merge_manifest(cfg, {
        "judge_agreement": {
            "n_filled": n,
            "n_agree": agree,
            "agreement_pct": round(pct, 2),
            "disagreements": disagreements,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
    })
    print(f"  merged judge_agreement → {cfg.run_manifest_path}")


def run(cfg: RunConfig, mode: str = "sample", monitor: RunMonitor | None = None) -> None:
    if mode not in {"sample", "score"}:
        raise ValueError(f"validate mode must be 'sample' or 'score', got {mode!r}")
    with stage_ctx(monitor, "validate", 1) as mon:
        mon.start_stage("validate", total=1)
        mon.item_start()
        if mode == "sample":
            sample(cfg, mon)
        else:
            score(cfg)
        mon.item_done()
        mon.end_stage()
