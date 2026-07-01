"""Join everything into outputs/results.json (FIXED CONTRACT 4).

A prompt is included only when EVERY model in CANDIDATES has a response
and grades for it, and the prompt has tags. This keeps the cross-model
comparison apples-to-apples; partial prompts are skipped with a note.

`run_manifest.json` is merged (not overwritten) so a prior
`validate score` run's `judge_agreement` survives.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from config import (
    CANDIDATES,
    CRITERIA_TAGS_PATH,
    DATA_JSONL,
    GRADES_DIR,
    RESPONSES_DIR,
    RESULTS_PATH,
    RUN_MANIFEST_PATH,
)
from pipeline._experiment import (
    SCHEMA_VERSION,
    build_meta,
    update_index,
    write_experiment_copy,
)
from pipeline._io import read_jsonl


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


def _load_all() -> tuple[list[dict], dict[str, dict[str, str]], dict[str, dict[str, list[dict]]], dict[str, list[dict]]]:
    records = read_jsonl(DATA_JSONL)
    responses: dict[str, dict[str, str]] = {}
    grades: dict[str, dict[str, list[dict]]] = {}
    for key in CANDIDATES:
        responses[key] = {r["id"]: r["response"] for r in read_jsonl(RESPONSES_DIR / f"{key}.jsonl")}
        grades[key] = {r["id"]: r["verdicts"] for r in read_jsonl(GRADES_DIR / f"{key}.jsonl")}
    tags = {r["id"]: r["tags"] for r in read_jsonl(CRITERIA_TAGS_PATH)}
    return records, responses, grades, tags


def _build_prompt_entry(rec: dict, responses: dict, grades: dict, tags: dict) -> dict:
    """Compose one entry of results.prompts[] for a single prompt."""
    criteria_texts = rec["criteria"]
    n = len(criteria_texts)
    tag_by_idx = {t["index"]: t for t in tags[rec["id"]]}

    criteria_list: list[dict] = []
    for i, ctext in enumerate(criteria_texts, start=1):
        tag = tag_by_idx.get(i, {"verifiability": "judge", "gameable": False, "reward_hack": ""})
        per_model: dict[str, dict] = {}
        for key in CANDIDATES:
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

    responses_per_model = {key: responses[key][rec["id"]] for key in CANDIDATES}
    criteria_passed: dict[str, str] = {}
    full_pass: dict[str, bool] = {}
    for key in CANDIDATES:
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


def _summary(prompts: list[dict]) -> dict:
    models = list(CANDIDATES.keys())

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


def _merge_manifest(update: dict) -> None:
    existing: dict = {}
    if RUN_MANIFEST_PATH.exists():
        try:
            existing = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(update)
    RUN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_MANIFEST_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run(
    limit: int | None = None,
    experiment: str | None = None,
    description: str | None = None,
    run_report: str | None = None,
) -> None:
    """Join everything into results.json (canonical) and, if `experiment`
    is a valid slug like `E01-smoke-3p`, also into
    `outputs/experiments/<slug>.json` plus the dashboard `index.json`.
    """
    records, responses, grades, tags = _load_all()
    if limit is not None:
        records = records[:limit]

    eligible: list[dict] = []
    skipped: list[str] = []
    for rec in records:
        rid = rec["id"]
        if rid not in tags:
            skipped.append(f"{rid} (no tags)")
            continue
        missing = [k for k in CANDIDATES if rid not in responses[k] or rid not in grades[k]]
        if missing:
            skipped.append(f"{rid} (missing {','.join(missing)})")
            continue
        eligible.append(rec)

    if skipped:
        print(f"aggregate: skipping {len(skipped)} incomplete prompt(s):")
        for s in skipped[:10]:
            print(f"  · {s}")
        if len(skipped) > 10:
            print(f"  · …and {len(skipped) - 10} more")

    prompts = [_build_prompt_entry(rec, responses, grades, tags) for rec in eligible]
    summary = _summary(prompts)
    run_date = datetime.now(timezone.utc).isoformat()

    meta = build_meta(
        slug=experiment,
        description=description,
        run_report=run_report,
        run_date_iso=run_date,
        prompts=prompts,
    )
    results = {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "summary": summary,
        "prompts": prompts,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"aggregate: wrote {meta['counts']['n_prompts']} prompts × "
        f"{meta['counts']['n_criteria']} criteria → {RESULTS_PATH}"
    )

    if experiment:
        exp_path = write_experiment_copy(experiment, results)
        idx_path = update_index(experiment, meta)
        print(f"aggregate: tagged as {experiment} → {exp_path}")
        print(f"aggregate: updated dashboard index → {idx_path}")
    else:
        print("aggregate: no --experiment slug provided; only canonical results.json written.")

    _merge_manifest({
        "schema_version": SCHEMA_VERSION,
        "experiment": meta["experiment"],
        "models": [m["key"] for m in meta["models"]],
        "counts": meta["counts"],
        "judge": meta["judge"]["id"],
        "run_date": run_date,
        "git": meta["git"],
        "config": meta["config"],
        "validation": meta["validation"],
    })
    print(f"aggregate: merged run summary → {RUN_MANIFEST_PATH}")


if __name__ == "__main__":
    run()
