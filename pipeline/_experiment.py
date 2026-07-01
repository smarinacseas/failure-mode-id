"""Experiment tagging + config snapshot helpers.

Every aggregate run produces two things a dashboard needs:
1. A `meta` block on the results.json that fully specifies the run's
   configuration (models, knobs, prompt-file fingerprints, git commit).
2. If the run was tagged with an `--experiment` slug, a copy of the
   results file under `outputs/experiments/<slug>.json` plus an entry
   in `outputs/experiments/index.json` (the dashboard dropdown source).

Slug convention: `E<NN>-<kebab-case-slug>`, e.g. `E01-smoke-3p`,
`E02-v1-75p`, `E03-reasoning-on`. The number gives dropdown ordering;
the kebab-case portion hints at the axis under investigation.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from config import (
    CANDIDATE_EXTRA_BODY,
    CANDIDATE_MAX_TOKENS,
    CANDIDATE_TEMPERATURE,
    CANDIDATE_TIMEOUT_S,
    CANDIDATES,
    EXPERIMENT_INDEX_PATH,
    EXPERIMENTS_DIR,
    JUDGE,
    JUDGE_MAX_TOKENS,
    PROMPTS_DIR,
    ROOT,
    VALIDATE_RESPONSE_EXCERPT_CHARS,
    VALIDATE_SAMPLE_TARGET,
    VALIDATE_SEED,
)

SLUG_RE = re.compile(r"^E(\d{2,})-[a-z0-9]+(-[a-z0-9]+)*$")


class InvalidSlugError(ValueError):
    """Raised when an experiment slug does not match the E<NN>-<name> convention."""


def parse_slug(slug: str) -> tuple[int, str]:
    """Return (number, label) for a valid slug or raise InvalidSlugError.

    Example: `E01-smoke-3p` → (1, "smoke-3p").
    """
    m = SLUG_RE.match(slug)
    if not m:
        raise InvalidSlugError(
            f"experiment slug {slug!r} must match E<NN>-<kebab-case-label> "
            "(e.g. E01-smoke-3p). Digits ≥ 2; label is lowercase-alphanumeric "
            "words separated by hyphens."
        )
    number = int(m.group(1))
    label = slug.split("-", 1)[1]
    return number, label


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


def config_snapshot() -> dict:
    """The full run-time configuration used for THIS aggregate.

    Every knob that could change between experiments is captured here.
    Prompt files are fingerprinted by SHA256 prefix so a dashboard can
    flag runs where the judge or classifier prompt was edited.
    """
    return {
        "candidates": dict(CANDIDATES),
        "candidate_temperature": CANDIDATE_TEMPERATURE,
        "candidate_max_tokens": CANDIDATE_MAX_TOKENS,
        "candidate_extra_body": CANDIDATE_EXTRA_BODY,
        "candidate_timeout_s": CANDIDATE_TIMEOUT_S,
        "judge": JUDGE,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "judge_prompt_sha256_12": _sha256_prefix(PROMPTS_DIR / "judge.txt"),
        "classifier_prompt_sha256_12": _sha256_prefix(PROMPTS_DIR / "classifier.txt"),
        "validate_seed": VALIDATE_SEED,
        "validate_sample_target": VALIDATE_SAMPLE_TARGET,
        "validate_response_excerpt_chars": VALIDATE_RESPONSE_EXCERPT_CHARS,
    }


def git_state() -> dict:
    commit = _git("rev-parse", "--short", "HEAD")
    dirty_out = _git("status", "--porcelain")
    return {
        "commit": commit or "",
        "dirty": bool(dirty_out) if dirty_out is not None else None,
    }


def experiment_block(
    slug: str | None,
    description: str | None,
    run_report: str | None,
) -> dict:
    """The `experiment` sub-block of the meta section.

    When `slug` is None the run is "untagged" — the meta still carries an
    `experiment` block with slug=None so downstream schemas stay stable.
    """
    if slug is None:
        return {
            "slug": None,
            "number": None,
            "label": None,
            "description": description or "",
            "run_report": run_report or "",
        }
    number, label = parse_slug(slug)
    return {
        "slug": slug,
        "number": number,
        "label": label,
        "description": description or "",
        "run_report": run_report or "",
    }


def write_experiment_copy(slug: str, results: dict) -> Path:
    """Write results.json under `outputs/experiments/<slug>.json`."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENTS_DIR / f"{slug}.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def update_index(slug: str, meta: dict) -> Path:
    """Insert or replace this experiment's entry in `experiments/index.json`.

    Entries are ordered by experiment number.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict = {"experiments": []}
    if EXPERIMENT_INDEX_PATH.exists():
        try:
            existing = json.loads(EXPERIMENT_INDEX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {"experiments": []}
    entries: list[dict] = list(existing.get("experiments", []))

    exp = meta["experiment"]
    entry = {
        "slug": exp["slug"],
        "number": exp["number"],
        "label": exp["label"],
        "description": exp["description"],
        "n_prompts": meta["n_prompts"],
        "n_criteria": meta["n_criteria"],
        "n_models": len(meta["models"]),
        "models": meta["models"],
        "judge": meta["judge"],
        "run_date": meta["run_date"],
        "git_commit": meta["git"]["commit"],
        "results_path": f"experiments/{slug}.json",
        "run_report": exp["run_report"],
    }
    entries = [e for e in entries if e.get("slug") != slug]
    entries.append(entry)
    entries.sort(key=lambda e: (e.get("number") is None, e.get("number") or 0, e.get("slug", "")))

    EXPERIMENT_INDEX_PATH.write_text(
        json.dumps({"experiments": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return EXPERIMENT_INDEX_PATH
