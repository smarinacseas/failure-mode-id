"""Per-experiment run configuration.

One frozen RunConfig per experiment slug. Built by `resolve()` in Task 2:
first invocation freezes params to runs/<slug>/experiment.json; later
invocations reload them. Stages receive the cfg explicitly — no module-level
knob mutation anywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config

SLUG_RE = re.compile(r"^E(\d{2,})-[a-z0-9]+(-[a-z0-9]+)*$")

_PARAM_FIELDS = (
    "candidates", "judge", "max_tokens", "temperature",
    "reasoning", "timeout_s", "limit", "description",
)


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


@dataclass(frozen=True)
class RunConfig:
    """Everything that can differ between experiments, plus derived paths."""

    slug: str
    candidates: dict[str, str]      # key -> provider model id
    judge: str                      # Anthropic model id (grader + classifier)
    max_tokens: int
    temperature: float
    reasoning: bool
    timeout_s: float
    limit: int | None               # prompt count; None = all
    description: str

    # --- derived paths (config.RUNS_DIR read at access time for testability) ---
    @property
    def run_dir(self) -> Path:
        return config.RUNS_DIR / self.slug

    @property
    def experiment_json_path(self) -> Path:
        return self.run_dir / "experiment.json"

    def responses_path(self, key: str) -> Path:
        return self.run_dir / "responses" / f"{key}.jsonl"

    def grades_path(self, key: str) -> Path:
        return self.run_dir / "grades" / f"{key}.jsonl"

    @property
    def criteria_tags_path(self) -> Path:
        return self.run_dir / "criteria_tags.jsonl"

    @property
    def judge_validation_path(self) -> Path:
        return self.run_dir / "judge_validation.json"

    @property
    def run_manifest_path(self) -> Path:
        return self.run_dir / "run_manifest.json"

    @property
    def extra_body(self) -> dict:
        """OpenRouter extra_body for candidate calls (reasoning toggle)."""
        return {"reasoning": {"enabled": self.reasoning}}

    # --- serialization (slug is the filename's job, not the payload's) ---
    def to_json_dict(self) -> dict:
        return {f: getattr(self, f) for f in _PARAM_FIELDS}

    @classmethod
    def from_json_dict(cls, slug: str, d: dict) -> "RunConfig":
        return cls(slug=slug, **{f: d[f] for f in _PARAM_FIELDS})


FREEZE_SCHEMA = 1


class ConfigConflictError(ValueError):
    """An explicitly-passed flag conflicts with a slug's frozen parameters."""


def parse_candidates(spec: str) -> dict[str, str]:
    """Parse `--candidates`: comma list of registry keys or key=provider/id pairs."""
    out: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "=" in entry:
            key, model_id = (s.strip() for s in entry.split("=", 1))
            if not key or not model_id:
                raise ValueError(f"malformed candidate entry {entry!r}; use key=provider/model-id")
            out[key] = model_id
        elif entry in config.CANDIDATES:
            out[entry] = config.CANDIDATES[entry]
        else:
            raise ValueError(
                f"unknown candidate key {entry!r}; registry keys: "
                f"{', '.join(config.CANDIDATES)} — or add a new model with key=provider/model-id"
            )
    if not out:
        raise ValueError("--candidates parsed to an empty set")
    return out


def validate_judge(judge: str) -> str:
    if not judge.startswith("claude-"):
        raise ValueError(
            f"--judge must be an Anthropic model id (claude-*), got {judge!r}. "
            "The judge/classifier client is Anthropic-only; non-Anthropic judges "
            "are the E04-judge-swap roadmap item."
        )
    return judge


def _defaults() -> dict:
    return {
        "candidates": dict(config.CANDIDATES),
        "judge": config.JUDGE,
        "max_tokens": config.CANDIDATE_MAX_TOKENS,
        "temperature": config.CANDIDATE_TEMPERATURE,
        "reasoning": False,
        "timeout_s": config.CANDIDATE_TIMEOUT_S,
        "limit": None,
        "description": "",
    }


def resolve(slug: str, overrides: dict) -> RunConfig:
    """Freeze-on-first-run: build a RunConfig for `slug`.

    `overrides` must contain ONLY the params the user explicitly passed.
    First invocation: defaults ⊕ overrides, frozen to experiment.json.
    Later invocations: frozen params win; a conflicting override is an error.
    """
    parse_slug(slug)  # validates format
    path = config.RUNS_DIR / slug / "experiment.json"

    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))["params"]
        diffs = {
            k: (frozen[k], v) for k, v in overrides.items()
            if k in frozen and frozen[k] != v
        }
        if diffs:
            lines = "\n".join(
                f"  {k}: frozen={f!r}  passed={p!r}" for k, (f, p) in sorted(diffs.items())
            )
            raise ConfigConflictError(
                f"experiment {slug!r} has frozen parameters; conflicting flags:\n"
                f"{lines}\n"
                "Parameters freeze on an experiment's first run — start a new "
                "slug to run with different parameters."
            )
        return RunConfig.from_json_dict(slug, frozen)

    params = _defaults()
    params.update(overrides)
    cfg = RunConfig(slug=slug, **params)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": FREEZE_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "params": cfg.to_json_dict(),
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return cfg
