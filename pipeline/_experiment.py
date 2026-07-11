"""Experiment tagging + deliverable-schema helpers.

Every aggregate run produces the standardized deliverable defined in
`meta/RESULTS_SCHEMA.md`. The shape is fixed across experiments so the
dashboard can bind once and render any run; changes to the shape bump
`SCHEMA_VERSION` and get documented in that file.

Two files per tagged run land in `outputs/`:
1. `outputs/experiments/<slug>.json` — the full deliverable.
2. `outputs/experiments/index.json` — the dashboard dropdown, one
   compact entry per experiment, ordered by experiment number.

Slug convention: `E<NN>-<kebab-case-slug>` (e.g. `E01-smoke-3p`,
`E02-v1-75p`, `E03-reasoning-on`). Two-digit number gives ordering;
label hints at the axis under investigation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from config import (
    EXPERIMENT_INDEX_PATH,
    EXPERIMENTS_DIR,
    GENERATION_DEADLINE_BASIS,
    GENERATION_DEADLINE_S,
    GENERATION_WORKERS,
    JUDGE_DEADLINE_S,
    JUDGE_MAX_TOKENS,
    JUDGE_WORKERS,
    PROMPTS_DIR,
    ROOT,
    VALIDATE_RESPONSE_EXCERPT_CHARS,
    VALIDATE_SAMPLE_TARGET,
    VALIDATE_SEED,
)
from pipeline.run_config import RunConfig
from pipeline.run_config import InvalidSlugError, SLUG_RE, parse_slug  # noqa: F401 — re-export

SCHEMA_VERSION = "3.2"

DATASET = {
    "name": "Complex Constraints Benchmark Set",
    "source": "https://huggingface.co/datasets/surgeai/ComplexConstraints",
    "license": "CC-BY-4.0",
    "publisher": "Surge AI",
}

JUDGE_ROLE = "grader"
FAMILY_NOTE_CLEAN = ("Non-candidate family -> self-preference bias structurally absent "
                     "for this judge.")
FAMILY_NOTE_OVERLAP = ("Judge family matches a candidate family -> self-preference "
                       "bias possible; interpret this judge's column with care.")


def _sha256_prefix(path: Path, n: int = 12) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:n]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def experiment_block(cfg: RunConfig, run_report: str | None, run_date_iso: str) -> dict:
    number, label = parse_slug(cfg.slug)
    return {
        "slug": cfg.slug,
        "number": number,
        "label": label,
        "description": cfg.description,
        "run_report": run_report or "",
        "run_date": run_date_iso,
    }


def dataset_block() -> dict:
    """`meta.dataset` — benchmark identity + license."""
    return dict(DATASET)


def models_block(cfg: RunConfig) -> list[str]:
    """`meta.models` — ordered candidate KEYS.

    Dashboards use these keys directly to index `summary.criterion_pass_rate`,
    `prompt.responses`, `criterion.results`, etc. — i.e. the same short
    strings the pipeline uses as its model handles.
    """
    return list(cfg.candidates.keys())


def model_details_block(cfg: RunConfig) -> list[dict]:
    """`meta.model_details` — the {key, id, role} rich variant.

    Optional companion to `meta.models`. Kept for future dashboards / analyses
    that want provider-side IDs or role tags; the ConstraintLens dashboard
    doesn't read this.
    """
    return [{"key": k, "id": v, "role": "candidate"} for k, v in cfg.candidates.items()]


def judges_block(cfg: RunConfig) -> list[str]:
    """`meta.judges` — every grader KEY, in order. The dashboard builds its
    judge toggle from this; the first entry is the default view."""
    return list(cfg.judge_keys)


def judge_block(cfg: RunConfig) -> str:
    """`meta.judge` — the DEFAULT grader KEY (first judge, short string).

    Kept for back-compat / single-judge dashboards; multi-judge views read
    `meta.judges` and the per-judge blocks under `by_judge`.
    """
    return cfg.judge.key


def judge_details_for(spec, cfg: RunConfig) -> dict:
    """Per-judge provenance for by_judge[*].judge_details (schema 3.3)."""
    from pipeline.run_config import candidate_family, family_of
    fam = family_of(spec.client, spec.model)
    overlap = fam in {candidate_family(m) for m in cfg.candidates.values()}
    if spec.client == "anthropic":
        transport = cfg.judge_mode          # batch | sequential
        reasoning = {"type": "adaptive"}
    else:
        transport = "pooled_stream"
        reasoning = {"enabled": True}
    return {
        "id": spec.key,
        "model_id": spec.model,
        "provider": spec.client,
        "role": JUDGE_ROLE,
        "transport": transport,
        "reasoning": reasoning,
        "family": fam,
        "family_overlap": overlap,
        "family_stake_note": FAMILY_NOTE_OVERLAP if overlap else FAMILY_NOTE_CLEAN,
    }


def judge_details_block(cfg: RunConfig) -> dict:
    return judge_details_for(cfg.judge, cfg)


def counts_block(prompts: list[dict], cfg: RunConfig) -> dict:
    """`meta.counts` — cross-referencing sums the dashboard prints as headers."""
    n_prompts = len(prompts)
    n_criteria = sum(len(p["criteria"]) for p in prompts)
    n_models = len(cfg.candidates)
    # n_grade_cells is per-judge (criteria × models) — matches the dashboard's
    # "criteria graded" tile, which shows one judge's view at a time.
    return {
        "n_prompts": n_prompts,
        "n_criteria": n_criteria,
        "n_models": n_models,
        "n_judges": len(cfg.judges),
        "n_grade_cells": n_criteria * n_models,
    }


def categories_block(prompts: list[dict]) -> dict:
    """`meta.categories` — distinct values + counts, for dashboard filter UI."""
    it = Counter(p["instruction_type"] for p in prompts)
    ps = Counter(p["prompt_style"] for p in prompts)
    uc = Counter(p["use_case"] for p in prompts)
    return {
        "instruction_type": dict(sorted(it.items())),
        "prompt_style": dict(sorted(ps.items())),
        "use_case": dict(sorted(uc.items())),
    }


def config_block(cfg: RunConfig) -> dict:
    """`meta.config` — every knob that could differ across experiments.

    `candidates` and `judge id` live in `meta.models` / `meta.judge`
    respectively; this block is the tunables the dashboard cares about
    when explaining "what was different about this run."
    """
    return {
        "candidate_temperature": cfg.temperature,
        "candidate_max_tokens": cfg.max_tokens,
        "candidate_extra_body": cfg.extra_body,
        "candidate_timeout_s": cfg.timeout_s,
        "limit": cfg.limit,
        "sample_seed": cfg.sample_seed,
        # Concurrency knobs (see CONCURRENCY.md): how this run's calls were
        # executed. judge_mode is frozen per-experiment; the worker count and
        # generation deadline are config.py constants recorded for the record.
        "judge_mode": cfg.judge_mode,
        "classifier_chain": [s.key for s in cfg.classifier_chain],
        "generation_workers": GENERATION_WORKERS,
        "generation_deadline_s": GENERATION_DEADLINE_S,
        "generation_deadline_basis": GENERATION_DEADLINE_BASIS,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "judge_workers": JUDGE_WORKERS,
        "judge_deadline_s": JUDGE_DEADLINE_S,
        "judge_prompt_sha256_12": _sha256_prefix(PROMPTS_DIR / "judge.txt"),
        "classifier_prompt_sha256_12": _sha256_prefix(PROMPTS_DIR / "classifier.txt"),
        "validate_seed": VALIDATE_SEED,
        "validate_sample_target": VALIDATE_SAMPLE_TARGET,
        "validate_response_excerpt_chars": VALIDATE_RESPONSE_EXCERPT_CHARS,
    }


def git_block() -> dict:
    commit = _git("rev-parse", "--short", "HEAD")
    dirty_out = _git("status", "--porcelain")
    return {
        "commit": commit or "",
        "dirty": bool(dirty_out) if dirty_out is not None else None,
    }


def validation_block(cfg: RunConfig) -> dict:
    """`meta.validation` — judge-validation status for the dashboard limitations panel.

    Read-only view of two files:
      - runs/<slug>/judge_validation.json (written by `validate --mode sample`)
      - runs/<slug>/run_manifest.json's judge_agreement block (merged in by
        `validate --mode score`)

    Status transitions: not_run → sampled → scored. Never reverses.
    """
    status = "not_run"
    n_sampled = 0
    n_scored = 0
    agreement_pct: float | None = None
    scored_at: str | None = None

    if cfg.judge_validation_path.exists():
        try:
            rows = json.loads(cfg.judge_validation_path.read_text(encoding="utf-8"))
            n_sampled = len(rows) if isinstance(rows, list) else 0
            status = "sampled"
        except json.JSONDecodeError:
            pass

    if cfg.run_manifest_path.exists():
        try:
            manifest = json.loads(cfg.run_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        ja = manifest.get("judge_agreement")
        if isinstance(ja, dict) and ja.get("n_filled"):
            status = "scored"
            n_scored = int(ja.get("n_filled") or 0)
            pct = ja.get("agreement_pct")
            agreement_pct = float(pct) if pct is not None else None
            scored_at = ja.get("scored_at") or None

    return {
        "status": status,
        "n_sampled": n_sampled,
        "n_scored": n_scored,
        "agreement_pct": agreement_pct,
        "scored_at": scored_at,
    }


def _promoted_config_fields(cfg: RunConfig) -> dict:
    """Top-level meta fields lifted out of `meta.config` for the dashboard.

    The ConstraintLens dashboard's Logic reads a fixed set of `meta.*` keys
    when rendering the run-details panel. Fields it wants to surface (token
    budget, reasoning-mode) live nested in `meta.config` in our schema, so
    we duplicate them at the top level here. Nested versions stay authoritative;
    these are display-only aliases — never edit them in-place downstream.

    Additive since schema 2.1 — dashboards that pre-date these fields still
    render fine (they fall back to "Not recorded" / omitted rows).
    """
    return {
        "run_date": None,  # filled by build_meta from run_date_iso
        "max_tokens": cfg.max_tokens,
        "reasoning_enabled": cfg.reasoning,
    }


def build_meta(cfg: RunConfig, run_report: str | None, run_date_iso: str, prompts: list[dict]) -> dict:
    """Assemble the entire `meta` block from the standardized sub-blocks.

    Dashboard-facing top-level fields (`run_date`, `max_tokens`,
    `reasoning_enabled`) are promoted from their canonical homes in
    `meta.experiment` / `meta.config` so the design's Logic can read them
    without knowing our nesting.
    """
    promoted = _promoted_config_fields(cfg)
    promoted["run_date"] = run_date_iso
    return {
        "experiment": experiment_block(cfg, run_report, run_date_iso),
        "dataset": dataset_block(),
        "models": models_block(cfg),
        "model_details": model_details_block(cfg),
        "judges": judges_block(cfg),
        "judge": judge_block(cfg),
        "judge_details": judge_details_block(cfg),
        "counts": counts_block(prompts, cfg),
        "categories": categories_block(prompts),
        "config": config_block(cfg),
        "git": git_block(),
        "validation": validation_block(cfg),
        # --- Promoted display-only aliases (dashboard reads these) ---
        **promoted,
    }


def write_experiment_copy(slug: str, results: dict) -> Path:
    """Write the full deliverable under `outputs/experiments/<slug>.json`."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / f"{slug}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def update_index(slug: str, meta: dict) -> Path:
    """Insert or replace this experiment's entry in `experiments/index.json`.

    Index entries are the dashboard dropdown's data source. They are
    compact by design — everything else lives in the per-experiment
    results file. Ordered by experiment number.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict = {"schema_version": SCHEMA_VERSION, "experiments": []}
    if EXPERIMENT_INDEX_PATH.exists():
        try:
            existing = json.loads(EXPERIMENT_INDEX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {"schema_version": SCHEMA_VERSION, "experiments": []}
    entries: list[dict] = list(existing.get("experiments", []))

    exp = meta["experiment"]
    v = meta["validation"]
    entry = {
        "slug": exp["slug"],
        "number": exp["number"],
        "label": exp["label"],
        "description": exp["description"],
        "run_date": exp["run_date"],
        "run_report": exp["run_report"],
        "n_prompts": meta["counts"]["n_prompts"],
        "n_criteria": meta["counts"]["n_criteria"],
        "n_models": meta["counts"]["n_models"],
        "models": list(meta["models"]),
        # Restored pre-3.0 deliverables carry a scalar judge only.
        "judges": list(meta.get("judges") or ([meta["judge"]] if meta.get("judge") else [])),
        "judge": meta.get("judge", ""),
        "validation_status": v["status"],
        "agreement_pct": v["agreement_pct"],
        "git_commit": meta["git"]["commit"],
        "results_path": f"experiments/{slug}.json",
    }
    entries = [e for e in entries if e.get("slug") != slug]
    entries.append(entry)
    entries.sort(key=lambda e: (e.get("number") is None, e.get("number") or 0, e.get("slug", "")))

    EXPERIMENT_INDEX_PATH.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "experiments": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return EXPERIMENT_INDEX_PATH
