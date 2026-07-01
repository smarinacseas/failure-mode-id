"""CLI orchestrator for the ComplexConstraints v1 eval pipeline.

Usage:
    uv run python main.py <step> [--limit N] [--mode sample|score]
                                 [--experiment SLUG] [--description STR]
                                 [--run-report PATH]
    uv run python main.py status          # print the live progress heartbeat

Steps: load · generate · grade · classify · validate · aggregate · all
       · connectivity · status

`all` runs connectivity → load → generate → grade → classify →
validate(sample) → aggregate, all under one live progress display.
`--limit N` is honored by every step; `all` REQUIRES `--limit`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from config import CANDIDATE_TIMEOUT_S, PROGRESS_PATH
from pipeline import (
    aggregate, classify, connectivity, generate, grade, load, validate,
)
from pipeline.monitor import build_monitor, render_lines

STEPS = ("load", "generate", "grade", "classify", "validate", "aggregate")

# A heartbeat only refreshes between items; a single in-flight model call can
# legitimately freeze `updated_at` for up to CANDIDATE_TIMEOUT_S. Only warn well
# beyond that, so a healthy slow call is never mistaken for a dead process.
_STALE_AFTER_S = CANDIDATE_TIMEOUT_S + 60.0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__)
    p.add_argument("step", choices=(*STEPS, "all", "connectivity", "status"),
                   help="Pipeline step to run.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N prompts. Required for `all`.")
    p.add_argument("--mode", choices=("sample", "score"), default="sample",
                   help="For `validate`: sample (default) or score.")
    p.add_argument("--experiment", default=None,
                   help="Experiment slug E<NN>-<label> (tags the run + heartbeat).")
    p.add_argument("--description", default=None,
                   help="One-liner describing what makes this experiment distinct.")
    p.add_argument("--run-report", default=None, dest="run_report",
                   help="Path to the meta/ run report for this experiment.")
    return p


def _run_all(limit, experiment, description, run_report) -> None:
    with build_monitor("all", limit, experiment) as mon:
        connectivity.run(monitor=mon)
        load.run(limit=limit, monitor=mon)
        generate.run(limit=limit, monitor=mon)
        grade.run(limit=limit, monitor=mon)
        classify.run(limit=limit, monitor=mon)
        validate.run(mode="sample", monitor=mon)
        aggregate.run(limit=limit, experiment=experiment,
                      description=description, run_report=run_report, monitor=mon)


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

    if args.step == "connectivity":
        with build_monitor("connectivity", 0, args.experiment) as mon:
            connectivity.run(monitor=mon)
        return 0

    if args.step == "all":
        if args.limit is None:
            print("error: `all` requires --limit (e.g. `--limit 3` for a smoke test, "
                  "`--limit 75` for the full set).", file=sys.stderr)
            return 2
        _run_all(args.limit, args.experiment, args.description, args.run_report)
        return 0

    with build_monitor(args.step, args.limit or 0, args.experiment) as mon:
        if args.step == "load":
            load.run(limit=args.limit, monitor=mon)
        elif args.step == "generate":
            generate.run(limit=args.limit, monitor=mon)
        elif args.step == "grade":
            grade.run(limit=args.limit, monitor=mon)
        elif args.step == "classify":
            classify.run(limit=args.limit, monitor=mon)
        elif args.step == "validate":
            validate.run(mode=args.mode, monitor=mon)
        elif args.step == "aggregate":
            aggregate.run(limit=args.limit, experiment=args.experiment,
                          description=args.description, run_report=args.run_report,
                          monitor=mon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
