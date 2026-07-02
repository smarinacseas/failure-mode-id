"""Per-experiment run configuration.

One frozen RunConfig per experiment slug. Built by `resolve()` in Task 2:
first invocation freezes params to runs/<slug>/experiment.json; later
invocations reload them. Stages receive the cfg explicitly — no module-level
knob mutation anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
