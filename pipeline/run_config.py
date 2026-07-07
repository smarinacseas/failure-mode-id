"""Per-experiment run configuration.

One frozen RunConfig per experiment slug. Built by `resolve()` in Task 2:
first invocation freezes params to runs/<slug>/experiment.json; later
invocations reload them. Stages receive the cfg explicitly — no module-level
knob mutation anywhere.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config

SLUG_RE = re.compile(r"^E(\d{2,})-[a-z0-9]+(-[a-z0-9]+)*$")

_PARAM_FIELDS = (
    "candidates", "judges", "max_tokens", "temperature",
    "reasoning", "timeout_s", "limit", "description", "provider_sort",
    "sample_seed", "judge_mode",
)

PROVIDER_SORTS = ("throughput", "latency", "price")
JUDGE_MODES = ("batch", "sequential")


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
    judges: tuple[str, ...]         # Anthropic model ids; each grades the same responses
    max_tokens: int
    temperature: float
    reasoning: bool
    timeout_s: float
    limit: int | None               # prompt count; None = all
    description: str
    # OpenRouter provider routing preference (None = default routing).
    # Default routing is a provider lottery whose throughput spans ~10x
    # (20 vs 164 tok/s observed 2026-07-02); "throughput" pins the fast end,
    # which reasoning runs need — a 32k+ thinking budget at 20 tok/s is a
    # ~25-minute call. Caveat: fast providers may serve quantized weights,
    # so routing preference is itself a (frozen, documented) treatment.
    provider_sort: str | None = None
    # Seeded stratified sampling: None = first-`limit` rows (E01–E04
    # behavior); an int selects `limit` prompts spread across use cases /
    # instruction types / prompt styles (see pipeline/_select.py). Frozen —
    # a different seed is a different prompt subset, hence a different
    # experiment.
    sample_seed: int | None = None
    # Judge transport: "batch" (Anthropic Message Batches: submit → poll →
    # collect) or "sequential" (one streamed call per grade cell, the
    # pre-concurrency path). Grading params and judge-blindness are identical
    # in both modes — this is a transport choice, not a treatment change —
    # but it is frozen so the manifest records how a run's grades were made.
    judge_mode: str = "batch"

    def __post_init__(self) -> None:
        if self.judge_mode not in JUDGE_MODES:
            raise ValueError(
                f"judge_mode must be one of {JUDGE_MODES}, got {self.judge_mode!r}"
            )

    @property
    def judge(self) -> str:
        """The canonical judge (first) — used for criterion classification and
        as the default view. Grades exist for every judge in `self.judges`."""
        return self.judges[0]

    # --- derived paths (config.RUNS_DIR read at access time for testability) ---
    @property
    def run_dir(self) -> Path:
        return config.RUNS_DIR / self.slug

    @property
    def experiment_json_path(self) -> Path:
        return self.run_dir / "experiment.json"

    def responses_path(self, key: str) -> Path:
        return self.run_dir / "responses" / f"{key}.jsonl"

    def grades_path(self, judge: str, key: str) -> Path:
        """Grades are keyed by (judge, candidate): grades/<judge>/<candidate>.jsonl.
        Each judge scores the same responses, so its verdicts live in its own dir."""
        return self.run_dir / "grades" / judge / f"{key}.jsonl"

    def diagnosis_path(self, key: str) -> Path:
        """Root-cause diagnoses per candidate: diagnosis/<candidate>.jsonl.
        Single analyst model (config.DIAGNOSE_JUDGE), so no judge subdir."""
        return self.run_dir / "diagnosis" / f"{key}.jsonl"

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
        """OpenRouter extra_body for candidate calls (reasoning toggle +
        optional provider routing preference)."""
        body: dict = {"reasoning": {"enabled": self.reasoning}}
        if self.provider_sort:
            body["provider"] = {"sort": self.provider_sort}
        return body

    # --- serialization (slug is the filename's job, not the payload's) ---
    def to_json_dict(self) -> dict:
        d = {f: getattr(self, f) for f in _PARAM_FIELDS}
        d["judges"] = list(self.judges)  # tuple → JSON array
        return d

    @classmethod
    def from_json_dict(cls, slug: str, d: dict) -> "RunConfig":
        d = dict(d)
        # Back-compat: pre-multi-judge freezes carry a scalar `judge`.
        if "judges" not in d and "judge" in d:
            d["judges"] = [d["judge"]]
        # Back-compat: freezes older than the provider_sort knob (pre-E04).
        d.setdefault("provider_sort", None)
        # Back-compat: freezes older than the sample_seed knob (pre-E05).
        d.setdefault("sample_seed", None)
        # Back-compat: freezes older than the judge_mode knob (pre-concurrency).
        d.setdefault("judge_mode", "batch")
        try:
            kwargs = {f: d[f] for f in _PARAM_FIELDS}
        except KeyError as e:
            raise ValueError(
                f"experiment.json for {slug!r} is missing field {e.args[0]!r} — "
                f"was it hand-edited? Delete runs/{slug}/ to start over."
            ) from e
        kwargs["judges"] = tuple(kwargs["judges"])
        return cls(slug=slug, **kwargs)


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


def parse_judges(spec: str) -> tuple[str, ...]:
    """Parse `--judges`: comma list of Anthropic model ids (dedup, order-preserving)."""
    out: list[str] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if entry and entry not in out:
            out.append(validate_judge(entry))
    if not out:
        raise ValueError("--judges parsed to an empty set")
    return tuple(out)


def _defaults() -> dict:
    return {
        "candidates": dict(config.CANDIDATES),
        "judges": tuple(config.JUDGES),
        "max_tokens": config.CANDIDATE_MAX_TOKENS,
        "temperature": config.CANDIDATE_TEMPERATURE,
        "reasoning": False,
        "timeout_s": config.CANDIDATE_TIMEOUT_S,
        "limit": None,
        "description": "",
        "provider_sort": None,
        "sample_seed": None,
        "judge_mode": "batch",
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

        def _norm(x):  # JSON round-trips tuples to lists; compare shape-agnostically
            return list(x) if isinstance(x, (list, tuple)) else x

        diffs = {
            k: (frozen[k], v) for k, v in overrides.items()
            if k in frozen and _norm(frozen[k]) != _norm(v)
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
    payload = json.dumps(
        {
            "schema": FREEZE_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "params": cfg.to_json_dict(),
        },
        ensure_ascii=False, indent=2,
    )
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return cfg
