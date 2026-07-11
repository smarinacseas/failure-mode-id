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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import config

SLUG_RE = re.compile(r"^E(\d{2,})-[a-z0-9]+(-[a-z0-9]+)*$")

_PARAM_FIELDS = (
    "candidates", "judges", "classifier_chain", "max_tokens", "temperature",
    "reasoning", "timeout_s", "limit", "description", "provider_sort",
    "sample_seed", "judge_mode",
)

PROVIDER_SORTS = ("throughput", "latency", "price")
JUDGE_MODES = ("batch", "sequential")

JUDGE_CLIENTS = ("anthropic", "openrouter")
# Path- and custom_id-safe; no "/" (paths) and no "__" (CUSTOM_ID_SEP).
JUDGE_KEY_RE = re.compile(r"^(?!.*__)[a-zA-Z0-9.-]+(_[a-zA-Z0-9.-]+)*$")


@dataclass(frozen=True)
class JudgeSpec:
    """One panel member. `key` is the judge's identity everywhere downstream
    (grades/<key>/, by_judge[<key>], dashboard labels, batch custom_ids);
    `model` is the provider-side id and never appears in a path."""
    key: str
    client: str
    model: str

    def __post_init__(self) -> None:
        if self.client not in JUDGE_CLIENTS:
            raise ValueError(f"judge client must be one of {JUDGE_CLIENTS}, got {self.client!r}")
        if not JUDGE_KEY_RE.match(self.key):
            raise ValueError(
                f"judge key {self.key!r} must be path-safe ([a-zA-Z0-9._-], no '__', no '/')")

    def to_dict(self) -> dict:
        return {"key": self.key, "client": self.client, "model": self.model}

    @classmethod
    def from_value(cls, v: "JudgeSpec | dict | str") -> "JudgeSpec":
        """Hydrate a freeze entry. Plain strings are pre-panel Anthropic ids
        (key == model) — byte-identical legacy behavior."""
        if isinstance(v, cls):
            return v
        if isinstance(v, dict):
            return cls(key=v["key"], client=v["client"], model=v["model"])
        return cls(key=v, client="anthropic", model=v)


def _infer_client(model: str, entry: str) -> str:
    if model.startswith("claude-"):
        return "anthropic"
    if "/" in model:
        return "openrouter"
    raise ValueError(
        f"cannot infer client for judge entry {entry!r}: model {model!r} is neither "
        "claude-* (anthropic) nor org/model-id (openrouter)")


def resolve_judge(entry: str) -> JudgeSpec:
    """CLI judge entry → JudgeSpec: registry key | key=provider/id | bare claude-*."""
    entry = entry.strip()
    if "=" in entry:
        key, model = (s.strip() for s in entry.split("=", 1))
        if not key or not model:
            raise ValueError(f"malformed judge entry {entry!r}; use key=provider/model-id")
        return JudgeSpec(key=key, client=_infer_client(model, entry), model=model)
    if entry in config.JUDGE_REGISTRY:
        reg = config.JUDGE_REGISTRY[entry]
        return JudgeSpec(key=entry, client=reg["client"], model=reg["model"])
    if entry.startswith("claude-"):
        return JudgeSpec(key=entry, client="anthropic", model=entry)
    raise ValueError(
        f"unknown judge {entry!r}; registry keys: {', '.join(config.JUDGE_REGISTRY)} — "
        "or add an unregistered judge with key=provider/model-id")


def family_of(client: str, model: str) -> str:
    """Model family for self-preference checks: 'anthropic' for the Anthropic
    client, else the OpenRouter id's org prefix (qwen/… → qwen)."""
    return "anthropic" if client == "anthropic" else model.split("/", 1)[0]


def candidate_family(model_id: str) -> str:
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


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
    # Panel graders; each grades the SAME responses. Constructor accepts
    # registry-key strings / {key,client,model} dicts / JudgeSpecs — all
    # hydrated to JudgeSpec in __post_init__.
    judges: tuple[JudgeSpec, ...]
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
    # Classify fallback chain (frozen). None -> (first judge,). Walked per prompt; refusal advances, one retry then advance (spec §4).
    classifier_chain: tuple[JudgeSpec, ...] | None = None

    def __post_init__(self) -> None:
        if self.judge_mode not in JUDGE_MODES:
            raise ValueError(
                f"judge_mode must be one of {JUDGE_MODES}, got {self.judge_mode!r}"
            )
        # Frozen dataclass: hydrate the accepted string/dict/spec forms in place
        # (object.__setattr__ is the sanctioned escape hatch for frozen writes).
        object.__setattr__(self, "judges",
                           tuple(JudgeSpec.from_value(j) for j in self.judges))
        chain = self.classifier_chain if self.classifier_chain is not None else (self.judges[0],)
        object.__setattr__(self, "classifier_chain",
                           tuple(JudgeSpec.from_value(j) for j in chain))

    @property
    def judge(self) -> JudgeSpec:
        """The canonical judge (first spec) — default dashboard view;
        classification uses classifier_chain."""
        return self.judges[0]

    @property
    def judge_keys(self) -> tuple[str, ...]:
        return tuple(s.key for s in self.judges)

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
        Each judge scores the same responses, so its verdicts live in its own dir.
        `judge` is the judge KEY (path-safe), never a provider model id."""
        return self.run_dir / "grades" / judge / f"{key}.jsonl"

    def diagnosis_path(self, key: str) -> Path:
        """Root-cause diagnoses per candidate: diagnosis/<candidate>.jsonl.
        Single analyst model (config.DIAGNOSE_JUDGE), so no judge subdir."""
        return self.run_dir / "diagnosis" / f"{key}.jsonl"

    @property
    def synthesis_path(self) -> Path:
        """Iteration synthesis (diagnose's final step): synthesis.json."""
        return self.run_dir / "synthesis.json"

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
        # tuple[JudgeSpec] → JSON array of {key,client,model} dicts.
        d["judges"] = [s.to_dict() for s in self.judges]
        d["classifier_chain"] = [s.to_dict() for s in self.classifier_chain]
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
        # Back-compat: freezes older than the classifier_chain knob (pre-panel).
        d.setdefault("classifier_chain", None)
        try:
            kwargs = {f: d[f] for f in _PARAM_FIELDS}
        except KeyError as e:
            raise ValueError(
                f"experiment.json for {slug!r} is missing field {e.args[0]!r} — "
                f"was it hand-edited? Delete runs/{slug}/ to start over."
            ) from e
        # Lists → tuples; the string/dict entries hydrate to JudgeSpec in
        # __post_init__ (None classifier_chain stays None → defaults there).
        kwargs["judges"] = tuple(kwargs["judges"])
        if kwargs["classifier_chain"] is not None:
            kwargs["classifier_chain"] = tuple(kwargs["classifier_chain"])
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


def parse_judges(spec: str) -> tuple[JudgeSpec, ...]:
    """Parse `--judges`/`--classifier`: comma list of registry keys,
    key=provider/model-id pairs, or bare claude-* ids (dedup by key)."""
    out: list[JudgeSpec] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        js = resolve_judge(entry)
        dup = next((o for o in out if o.key == js.key), None)
        if dup is not None:
            if dup != js:
                raise ValueError(f"judge key {js.key!r} given twice with different models")
            continue
        out.append(js)
    if not out:
        raise ValueError("--judges parsed to an empty set")
    return tuple(out)


def _defaults() -> dict:
    return {
        "candidates": dict(config.CANDIDATES),
        # Resolve registry keys to specs so the default panel carries real
        # clients/models — a raw "gpt-5" key would otherwise be sent to the
        # Anthropic API by grade/connectivity.
        "judges": tuple(resolve_judge(k) for k in config.JUDGES),
        "classifier_chain": None,
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


def family_overlaps(cfg: RunConfig) -> list[tuple[str, str]]:
    """Judges whose model family matches a candidate family — the
    self-preference-bias configuration the v1 design structurally avoided.
    Recorded and warned about, never blocked (a deliberate same-family
    ablation is a legitimate experiment)."""
    cand = {candidate_family(m) for m in cfg.candidates.values()}
    return [(s.key, family_of(s.client, s.model)) for s in cfg.judges
            if family_of(s.client, s.model) in cand]


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

        def _norm(x):  # JSON round-trips tuples→lists and specs→dicts; compare shape-agnostically
            if isinstance(x, JudgeSpec):
                return x.to_dict()
            if isinstance(x, (list, tuple)):
                return [_norm(i) for i in x]
            return x

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
    for key, fam in family_overlaps(cfg):
        print(f"⚠ judge {key!r} shares the {fam!r} family with a candidate — "
              "self-preference bias is possible; this is recorded in the results JSON.",
              file=sys.stderr)
    return cfg
