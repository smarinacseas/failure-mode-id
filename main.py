"""CLI orchestrator for the ComplexConstraints eval pipeline.

Usage:
    uv run python main.py <step> --experiment E<NN>-<label> [param flags]
    uv run python main.py status

Steps: load · generate · grade · classify · validate · aggregate · all
       · connectivity · status

Data-producing steps require --experiment. The first invocation of a slug
freezes its parameters to runs/<slug>/experiment.json; later invocations
need only the slug. `all` runs connectivity → load → generate → grade →
classify → validate(sample) → aggregate under one live progress display
and requires --limit on first invocation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import config
from config import CANDIDATE_TIMEOUT_S, PROGRESS_PATH
from pipeline import (
    aggregate, classify, connectivity, diagnose, generate, grade, load, validate,
)
from pipeline.monitor import build_monitor, render_lines
from pipeline.run_config import (
    ConfigConflictError,
    InvalidSlugError,
    parse_candidates,
    parse_judges,
    resolve,
    resolve_judge,
)

DATA_STEPS = ("generate", "grade", "classify", "diagnose", "validate", "aggregate", "all")

# A heartbeat only refreshes between items; a single in-flight model call can
# legitimately freeze `updated_at` for up to the candidate timeout. Only warn
# well beyond that, so a healthy slow call is never mistaken for a dead process.
_STALE_AFTER_S = CANDIDATE_TIMEOUT_S + 60.0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__)
    p.add_argument("step", choices=("load", *DATA_STEPS, "connectivity", "status"),
                   help="Pipeline step to run.")
    p.add_argument("--experiment", default=None,
                   help="Experiment slug E<NN>-<label>. Required for data steps; "
                        "first use freezes this run's parameters.")
    p.add_argument("--limit", type=int, default=None,
                   help="Prompt count (frozen). Required for a new experiment via `all`.")
    p.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                   help="Candidate max output tokens (frozen; default 8000).")
    p.add_argument("--reasoning", choices=("on", "off"), default=None,
                   help="Candidate reasoning mode (frozen; default off).")
    p.add_argument("--temperature", type=float, default=None,
                   help="Candidate temperature (frozen; default 0.0).")
    p.add_argument("--timeout", type=float, default=None,
                   help="Per-call timeout seconds (frozen; default 300).")
    p.add_argument("--candidates", default=None,
                   help="Comma list: registry keys and/or key=provider/model-id pairs "
                        "(frozen; default: full registry).")
    p.add_argument("--judge", default=None,
                   help="Single judge (frozen): a registry key, a key=provider/model-id "
                        "pair, or a bare claude-* id. Shorthand for --judges with one entry.")
    p.add_argument("--judges", default=None,
                   help="Comma list of judges (frozen; default: full registry). Entries are "
                        "registry keys, key=provider/model-id pairs, or bare claude-* ids. "
                        "Each grades the same responses; the dashboard toggles between them.")
    p.add_argument("--classifier", default=None,
                   help="Criterion-classifier fallback chain (frozen): comma list in "
                        "--judges syntax, walked per prompt (refusal/failed members "
                        "fall through). Default: the first judge.")
    p.add_argument("--tiebreaker", default=None,
                   help="Panel tie-break anchor judge KEY (frozen; E08 policy). When "
                        "set, a 1-1 tie is broken by this judge's vote and "
                        "undecidable/under-quorum criteria become EXCLUDE (dropped, "
                        "never FAIL) per plan §0.3. Must be one of --judges. Default "
                        "unset → legacy consensus (ties/no-quorum → FAIL; E01-E07).")
    p.add_argument("--sample-seed", type=int, default=None, dest="sample_seed",
                   help="Seeded stratified sampling (frozen): pick --limit prompts "
                        "spread across use cases / instruction types / prompt styles "
                        "instead of the first N rows. Same seed → same subset in "
                        "every stage and on resume.")
    p.add_argument("--provider-sort", choices=("throughput", "latency", "price"),
                   default=None, dest="provider_sort",
                   help="OpenRouter provider routing preference for candidate calls "
                        "(frozen; default: OpenRouter's own routing). Reasoning runs "
                        "want 'throughput' — thinking budgets on a slow provider mean "
                        "20+ minute calls.")
    p.add_argument("--provider-quant", default=None, dest="provider_quant",
                   help="Comma list of allowed candidate quantizations (frozen; e.g. "
                        "'bf16,fp16'). Pins OpenRouter routing to matching providers and "
                        "sets require_parameters, so an int-quantized endpoint can't "
                        "silently serve the candidate (§0.2 provider-variance guard).")
    p.add_argument("--seed", type=int, default=None,
                   help="Candidate decode seed (frozen; §0.2 'seeds logged'). Best-effort "
                        "on OpenRouter, but the requested seed is always recorded.")
    p.add_argument("--judge-mode", choices=("batch", "sequential"),
                   default=None, dest="judge_mode",
                   help="Judge transport (frozen; default batch): 'batch' grades via "
                        "the Anthropic Message Batches API (submit → poll → collect); "
                        "'sequential' keeps the one-streamed-call-per-cell path. "
                        "Grading params are identical either way.")
    p.add_argument("--diagnose", choices=("on", "off"), default="on",
                   dest="diagnose",
                   help="For `all`: run the failure root-cause analysis stage "
                        "after classify (default on). NOT frozen — analysis is "
                        "post-hoc and re-runnable; skipping it costs nothing "
                        "later (`main.py diagnose --experiment <slug>` backfills). "
                        "'on' spends extra judge tokens (~1 batch call per "
                        "failed response cell).")
    p.add_argument("--description", default=None,
                   help="One-liner describing what makes this experiment distinct (frozen).")
    p.add_argument("--run-report", default=None, dest="run_report",
                   help="Path to the meta/ run report (aggregate-time; not frozen).")
    p.add_argument("--mode", choices=("sample", "score"), default="sample",
                   help="For `validate`: sample (default) or score.")
    return p


def _overrides(args: argparse.Namespace) -> dict:
    """Map explicitly-passed flags to RunConfig param names."""
    out: dict = {}
    if args.limit is not None:
        out["limit"] = args.limit
    if args.max_tokens is not None:
        out["max_tokens"] = args.max_tokens
    if args.reasoning is not None:
        out["reasoning"] = args.reasoning == "on"
    if args.temperature is not None:
        out["temperature"] = args.temperature
    if args.timeout is not None:
        out["timeout_s"] = args.timeout
    if args.candidates is not None:
        out["candidates"] = parse_candidates(args.candidates)
    if args.judges is not None:
        out["judges"] = parse_judges(args.judges)
    elif args.judge is not None:
        out["judges"] = (resolve_judge(args.judge),)
    if args.classifier is not None:
        out["classifier_chain"] = parse_judges(args.classifier)
    if args.tiebreaker is not None:
        out["tiebreaker_judge"] = args.tiebreaker
    if args.description is not None:
        out["description"] = args.description
    if args.provider_sort is not None:
        out["provider_sort"] = args.provider_sort
    if args.provider_quant is not None:
        out["provider_quantizations"] = tuple(
            q.strip() for q in args.provider_quant.split(",") if q.strip())
    if args.seed is not None:
        out["seed"] = args.seed
    if args.sample_seed is not None:
        out["sample_seed"] = args.sample_seed
    if args.judge_mode is not None:
        out["judge_mode"] = args.judge_mode
    return out


def _run_all(cfg, run_report, diagnose_enabled: bool = True) -> None:
    with build_monitor("all", cfg.limit or 0, cfg.slug,
                       n_candidates=len(cfg.candidates)) as mon:
        connectivity.run(cfg, monitor=mon)
        # data/complexconstraints.jsonl is parameter-independent (spec invariant);
        # cfg.limit does all limiting downstream, so load the full dataset here.
        load.run(limit=None, monitor=mon)
        generate.run(cfg, monitor=mon)
        grade.run(cfg, monitor=mon)
        classify.run(cfg, monitor=mon)
        if diagnose_enabled:
            diagnose.run(cfg, monitor=mon)
        validate.run(cfg, mode="sample", monitor=mon)
        aggregate.run(cfg, run_report=run_report, monitor=mon)


def _print_status() -> None:
    if not PROGRESS_PATH.exists():
        print("no run found (outputs/progress.json missing).")
        return
    try:
        snap = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print("no run found (outputs/progress.json is unreadable).")
        return
    print(render_lines(snap))
    if snap.get("state") == "running":
        try:
            updated = datetime.fromisoformat(snap["updated_at"])
            age = (datetime.now(timezone.utc) - updated).total_seconds()
        except (KeyError, ValueError, TypeError):
            age = None
        if age is not None and age > _STALE_AFTER_S:
            print(f"\n⚠ possibly stalled — last update {int(age)}s ago (pid {snap.get('pid')}).")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.step == "status":
        _print_status()
        return 0

    if args.step == "load":
        with build_monitor("load", args.limit or 0, args.experiment) as mon:
            load.run(limit=args.limit, monitor=mon)
        return 0

    # connectivity: cfg optional — with a slug it pings that experiment's models.
    cfg = None
    if args.experiment is not None or args.step in DATA_STEPS:
        if args.experiment is None:
            print(f"error: `{args.step}` requires --experiment E<NN>-<label> "
                  "(parameters freeze on the slug's first run).", file=sys.stderr)
            return 2
        if (args.step == "connectivity"
                and not (config.RUNS_DIR / args.experiment / "experiment.json").exists()):
            print(
                f"error: experiment '{args.experiment}' has no frozen parameters yet — "
                "run a data step first, or run connectivity without --experiment to ping "
                "the default models.",
                file=sys.stderr,
            )
            return 2
        if (args.step == "all" and args.limit is None
                and not (config.RUNS_DIR / args.experiment / "experiment.json").exists()):
            print("error: a new `all` run requires --limit (e.g. `--limit 3` for a "
                  "smoke test, `--limit 75` for the full set).", file=sys.stderr)
            return 2
        try:
            cfg = resolve(args.experiment, _overrides(args))
        except (ConfigConflictError, InvalidSlugError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    if args.step == "connectivity":
        n = len(cfg.candidates) if cfg else None
        with build_monitor("connectivity", 0, args.experiment, n_candidates=n) as mon:
            connectivity.run(cfg, monitor=mon)
        return 0

    if args.step == "all":
        if cfg.limit is None:
            print(f"error: experiment '{cfg.slug}' was frozen without a limit; 'all' needs "
                  "one. Run steps individually, or start a new slug with --limit.",
                  file=sys.stderr)
            return 2
        _run_all(cfg, args.run_report, diagnose_enabled=args.diagnose == "on")
        return 0

    with build_monitor(args.step, cfg.limit or 0, cfg.slug,
                       n_candidates=len(cfg.candidates)) as mon:
        if args.step == "generate":
            generate.run(cfg, monitor=mon)
        elif args.step == "grade":
            grade.run(cfg, monitor=mon)
        elif args.step == "classify":
            classify.run(cfg, monitor=mon)
        elif args.step == "diagnose":
            diagnose.run(cfg, monitor=mon)
        elif args.step == "validate":
            validate.run(cfg, mode=args.mode, monitor=mon)
        elif args.step == "aggregate":
            aggregate.run(cfg, run_report=args.run_report, monitor=mon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
