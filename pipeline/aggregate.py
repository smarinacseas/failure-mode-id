"""Join everything into outputs/results.json (FIXED CONTRACT 4).

A prompt is included only when EVERY model in cfg.candidates has a response
and grades for it, and the prompt has tags. This keeps the cross-model
comparison apples-to-apples; partial prompts are skipped with a note.

`run_manifest.json` is merged (not overwritten) so a prior
`validate score` run's `judge_agreement` survives.

For tagged runs, the ConstraintLens dashboard's static data folder
(`dashboard/public/data/`) is refreshed at the end so the deliverable
is immediately visible in `vite dev` (and shipped by the next GH Pages
build). Sync failures are non-fatal — aggregate's canonical artifacts
still land regardless.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_JSONL, JUDGE_MAX_TOKENS, RESULTS_PATH, ROOT
from pipeline._experiment import (
    SCHEMA_VERSION,
    build_meta,
    judge_details_for,
    update_index,
    validation_block,
    write_experiment_copy,
)
from pipeline._io import read_jsonl
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import RunConfig

DASHBOARD_SYNC_SCRIPT: Path = ROOT / "scripts" / "dashboard_sync.py"


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _load_all(cfg: RunConfig):
    """Return (records, responses, grades_by_judge, tags).

    responses/tags are judge-independent (shared). grades_by_judge is
    {judge: {candidate: {id: verdicts}}} — one grade set per judge over the
    same responses.
    """
    records = read_jsonl(DATA_JSONL)
    responses: dict[str, dict[str, str]] = {}
    for key in cfg.candidates:
        responses[key] = {r["id"]: r["response"] for r in read_jsonl(cfg.responses_path(key))}
    grades_by_judge: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for judge in cfg.judges:
        grades_by_judge[judge] = {
            key: {r["id"]: r["verdicts"] for r in read_jsonl(cfg.grades_path(judge, key))}
            for key in cfg.candidates
        }
    tags = {r["id"]: r["tags"] for r in read_jsonl(cfg.criteria_tags_path)}
    return records, responses, grades_by_judge, tags


def _build_prompt_entry(rec: dict, responses: dict, grades: dict, tags: dict, models: list[str]) -> dict:
    """Compose one entry of results.prompts[] for a single prompt."""
    criteria_texts = rec["criteria"]
    n = len(criteria_texts)
    tag_by_idx = {t["index"]: t for t in tags[rec["id"]]}

    criteria_list: list[dict] = []
    for i, ctext in enumerate(criteria_texts, start=1):
        tag = tag_by_idx.get(i, {"verifiability": "judge", "gameable": False, "reward_hack": ""})
        per_model: dict[str, dict] = {}
        for key in models:
            verdicts = {v["index"]: v for v in grades[key][rec["id"]]}
            v = verdicts.get(i, {"verdict": "FAIL", "reason": "missing_verdict"})
            per_model[key] = {
                "pass": v["verdict"] == "PASS",
                "reason": v.get("reason", ""),
            }
        criteria_list.append({
            "text": ctext,
            "verifiability": tag["verifiability"],
            "gameable": tag["gameable"],
            "reward_hack": tag.get("reward_hack", ""),
            "results": per_model,
        })

    responses_per_model = {key: responses[key][rec["id"]] for key in models}
    criteria_passed: dict[str, str] = {}
    full_pass: dict[str, bool] = {}
    for key in models:
        passed = sum(1 for c in criteria_list if c["results"][key]["pass"])
        criteria_passed[key] = f"{passed}/{n}"
        full_pass[key] = passed == n

    return {
        "id": rec["id"],
        "use_case": rec["use_case"],
        "instruction_type": rec["instruction_type"],
        "prompt_style": rec["prompt_style"],
        "prompt_text": rec["prompt"],
        "responses": responses_per_model,
        "criteria_passed": criteria_passed,
        "full_pass": full_pass,
        "criteria": criteria_list,
    }


def _summary(prompts: list[dict], models: list[str]) -> dict:
    crit_pass: dict[str, int] = {m: 0 for m in models}
    crit_total: dict[str, int] = {m: 0 for m in models}
    full_pass_n: dict[str, int] = {m: 0 for m in models}

    by_it: dict[str, dict[str, list[int]]] = {}     # type -> model -> [passed, total]
    by_ps: dict[str, dict[str, list[int]]] = {}
    by_uc: dict[str, dict[str, list[int]]] = {}
    by_ver: dict[str, dict[str, list[int]]] = {"auto": {m: [0, 0] for m in models},
                                                "judge": {m: [0, 0] for m in models}}

    def _bump(d: dict, key_outer: str, model: str, passed: bool):
        d.setdefault(key_outer, {m: [0, 0] for m in models})
        d[key_outer][model][1] += 1
        if passed:
            d[key_outer][model][0] += 1

    for p in prompts:
        for m in models:
            n_pass = sum(1 for c in p["criteria"] if c["results"][m]["pass"])
            n_tot = len(p["criteria"])
            crit_pass[m] += n_pass
            crit_total[m] += n_tot
            if p["full_pass"][m]:
                full_pass_n[m] += 1
            for c in p["criteria"]:
                _bump(by_it, p["instruction_type"], m, c["results"][m]["pass"])
                _bump(by_ps, p["prompt_style"], m, c["results"][m]["pass"])
                _bump(by_uc, p["use_case"], m, c["results"][m]["pass"])
                ver = c["verifiability"]
                by_ver[ver][m][1] += 1
                if c["results"][m]["pass"]:
                    by_ver[ver][m][0] += 1

    def _flatten(d: dict) -> dict:
        return {
            group: {m: _pct(vals[0], vals[1]) for m, vals in per_model.items()}
            for group, per_model in d.items()
        }

    n_prompts = len(prompts)
    return {
        "criterion_pass_rate": {m: _pct(crit_pass[m], crit_total[m]) for m in models},
        "full_prompt_pass_rate": {m: _pct(full_pass_n[m], n_prompts) for m in models},
        "by_instruction_type": _flatten(by_it),
        "by_prompt_style": _flatten(by_ps),
        "by_use_case": _flatten(by_uc),
        "by_verifiability": _flatten(by_ver),
    }


def _sync_dashboard(mon) -> None:
    """Best-effort copy of every experiment deliverable into the dashboard's
    static data folder. Failure here never blocks aggregation — the sync is
    a convenience, not a correctness requirement."""
    if not DASHBOARD_SYNC_SCRIPT.exists():
        return
    try:
        subprocess.run(
            [sys.executable, str(DASHBOARD_SYNC_SCRIPT)],
            check=True,
            cwd=str(ROOT),
            timeout=20,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        mon.note(f"aggregate: dashboard sync skipped ({type(e).__name__}: {e})")


def _build_run_notes(cfg: RunConfig, responses: dict, by_judge: dict, skipped: list[str]) -> list[str]:
    """`meta.run_notes` — high-level issues/errors/factors for the dashboard's
    expandable Run summary. Hand-written context first (runs/<slug>/NOTES.md,
    one note per line, '#' lines skipped), then auto-derived anomalies, then
    config factors worth knowing that aren't already dashboard rows."""
    notes: list[str] = []

    notes_md = cfg.run_dir / "NOTES.md"
    if notes_md.exists():
        for line in notes_md.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                notes.append(line)

    # Auto-derived issues.
    empty = [f"{key}/{rid}" for key, by_id in responses.items()
             for rid, text in by_id.items() if not (text or "").strip()]
    if empty:
        notes.append(f"⚠ Empty candidate response(s) stored for {', '.join(sorted(empty))} — "
                     "their prompts are excluded from comparison; regenerate.")
    if skipped:
        notes.append(f"⚠ {len(skipped)} prompt(s) skipped as incomplete: "
                     + "; ".join(skipped[:5]) + ("…" if len(skipped) > 5 else ""))
    for judge, view in by_judge.items():
        counts: dict[str, int] = {}
        for p in view["prompts"]:
            for c in p["criteria"]:
                for r in c["results"].values():
                    reason = str(r.get("reason", ""))
                    for tag in ("judge_parse_error", "judge_truncated", "missing_in_judge_output"):
                        if reason.startswith(tag):
                            counts[tag] = counts.get(tag, 0) + 1
        if counts:
            detail = ", ".join(f"{n} × {tag}" for tag, n in sorted(counts.items()))
            notes.append(f"⚠ Judge {judge}: {detail} — affected criteria count as FAIL.")

    # Factors worth noting beyond the metadata rows already shown.
    if len(cfg.judges) > 1:
        notes.append("All judges graded the same candidate responses (single generation pass), "
                     "so judge columns are directly comparable.")
    notes.append("Judges reason with adaptive thinking over a streamed "
                 f"{JUDGE_MAX_TOKENS}-token budget; candidate reasoning "
                 f"{'enabled' if cfg.reasoning else 'disabled'} at temperature {cfg.temperature}.")
    return notes


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


def run(cfg: RunConfig, run_report: str | None = None, monitor: RunMonitor | None = None) -> None:
    with stage_ctx(monitor, "aggregate", 1) as mon:
        mon.start_stage("aggregate", total=1)
        mon.item_start()
        _run(cfg, run_report, mon)
        mon.item_done()
        mon.end_stage()


def _run(cfg: RunConfig, run_report: str | None, mon) -> None:
    """Join everything into results.json (canonical) and, since `cfg.slug`
    is always a valid slug like `E01-smoke-3p`, also into
    `outputs/experiments/<slug>.json` plus the dashboard `index.json`.
    """
    # A missing --run-report path silently misses the whole "each experiment
    # produces a standardized MD summary" discipline — every meta/*.md is
    # meant to derive from meta/TEMPLATE.md. Warn early rather than embed a
    # dead path into the JSON meta.
    if run_report:
        report_path = ROOT / run_report if not Path(run_report).is_absolute() else Path(run_report)
        if not report_path.exists():
            mon.note(
                f"aggregate: WARNING — --run-report path does not exist: {run_report}\n"
                f"           Copy meta/TEMPLATE.md → {run_report} and fill it in.",
            )

    records, responses, grades_by_judge, tags = _load_all(cfg)
    if cfg.limit is not None:
        records = records[: cfg.limit]

    models = list(cfg.candidates.keys())
    judges = list(cfg.judges)

    def _grade_present(rid: str, key: str) -> bool:
        return all(rid in grades_by_judge[j][key] for j in judges)

    eligible: list[dict] = []
    skipped: list[str] = []
    for rec in records:
        rid = rec["id"]
        if rid not in tags:
            skipped.append(f"{rid} (no tags)")
            continue
        missing = [k for k in models if rid not in responses[k] or not _grade_present(rid, k)]
        if missing:
            skipped.append(f"{rid} (missing {','.join(missing)})")
            continue
        eligible.append(rec)

    if skipped:
        mon.note(f"aggregate: skipping {len(skipped)} incomplete prompt(s):")
        for s in skipped[:10]:
            mon.note(f"  · {s}")
        if len(skipped) > 10:
            mon.note(f"  · …and {len(skipped) - 10} more")

    # Build a self-contained view per judge over the SAME eligible prompts.
    validation = validation_block(cfg)
    by_judge: dict[str, dict] = {}
    for judge in judges:
        j_prompts = [_build_prompt_entry(rec, responses, grades_by_judge[judge], tags, models)
                     for rec in eligible]
        by_judge[judge] = {
            "judge": judge,
            "judge_details": judge_details_for(judge),
            "validation": validation,
            "summary": _summary(j_prompts, models),
            "prompts": j_prompts,
        }

    run_date = datetime.now(timezone.utc).isoformat()

    # Default (top-level) view = first judge — back-compat for single-judge readers.
    default = by_judge[cfg.judge]
    prompts = default["prompts"]
    summary = default["summary"]

    meta = build_meta(cfg, run_report, run_date, prompts)
    meta["run_notes"] = _build_run_notes(cfg, responses, by_judge, skipped)
    results = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "summary": summary,
        "prompts": prompts,
        "by_judge": by_judge,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    mon.note(
        f"aggregate: wrote {meta['counts']['n_prompts']} prompts × "
        f"{meta['counts']['n_criteria']} criteria → {RESULTS_PATH}"
    )

    exp_path = write_experiment_copy(cfg.slug, results)
    idx_path = update_index(cfg.slug, meta)
    mon.note(f"aggregate: tagged as {cfg.slug} → {exp_path}")
    mon.note(f"aggregate: updated dashboard index → {idx_path}")
    _sync_dashboard(mon)

    _merge_manifest(cfg, {
        "schema_version": SCHEMA_VERSION,
        "experiment": meta["experiment"],
        "models": list(meta["models"]),
        "counts": meta["counts"],
        "judges": meta["judges"],
        "judge": meta["judge"],
        "run_date": run_date,
        "git": meta["git"],
        "config": meta["config"],
        "validation": meta["validation"],
    })
    mon.note(f"aggregate: merged run summary → {cfg.run_manifest_path}")
