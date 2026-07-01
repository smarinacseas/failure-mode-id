"""CLI orchestrator for the ComplexConstraints v1 eval pipeline.

Usage:
    uv run python main.py <step> [--limit N] [--mode sample|score]
                                 [--experiment SLUG] [--description STR]
                                 [--run-report PATH]

Steps: load · generate · grade · classify · validate · aggregate · all · connectivity

`all` runs load → generate → grade → classify → validate(sample) → aggregate.
`--limit N` is honored by every step. `all` REQUIRES `--limit` so that the
full 75-prompt run is always an explicit opt-in (use `--limit 75`).

`--experiment SLUG` tags the run for the dashboard dropdown. Slug convention
is `E<NN>-<kebab-case-label>` (e.g. `E01-smoke-3p`, `E02-v1-75p`,
`E03-reasoning-on`). When set, aggregate also writes a per-experiment copy
under `outputs/experiments/<slug>.json` and updates
`outputs/experiments/index.json`. Untagged runs still write the canonical
`outputs/results.json` but do not appear in the dashboard dropdown.
"""

from __future__ import annotations

import argparse
import sys

from pipeline import (
    aggregate,
    classify,
    connectivity,
    generate,
    grade,
    load,
    validate,
)

STEPS = ("load", "generate", "grade", "classify", "validate", "aggregate")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="main.py", description=__doc__)
    p.add_argument(
        "step",
        choices=(*STEPS, "all", "connectivity"),
        help="Pipeline step to run.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N prompts. Required for `all`.",
    )
    p.add_argument(
        "--mode",
        choices=("sample", "score"),
        default="sample",
        help="For `validate`: sample (default) or score.",
    )
    p.add_argument(
        "--experiment",
        default=None,
        help="Experiment slug `E<NN>-<label>` (e.g. E01-smoke-3p). "
             "Tags this run for the dashboard dropdown.",
    )
    p.add_argument(
        "--description",
        default=None,
        help="Free-text one-liner describing what makes this experiment distinct.",
    )
    p.add_argument(
        "--run-report",
        default=None,
        dest="run_report",
        help="Path to the meta/ run report for this experiment (e.g. meta/2026-06-30-smoke-test.md).",
    )
    return p


def _run_all(
    limit: int,
    experiment: str | None,
    description: str | None,
    run_report: str | None,
) -> None:
    tag = f" · {experiment}" if experiment else ""
    print(f"=== all · limit={limit}{tag} ===\n")
    connectivity.run()
    print()
    load.run(limit=limit)
    print()
    generate.run(limit=limit)
    print()
    grade.run(limit=limit)
    print()
    classify.run(limit=limit)
    print()
    validate.run(mode="sample")
    print()
    aggregate.run(
        limit=limit,
        experiment=experiment,
        description=description,
        run_report=run_report,
    )
    print("\n=== all done ===")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.step == "connectivity":
        connectivity.run()
        return 0

    if args.step == "all":
        if args.limit is None:
            print(
                "error: `all` requires --limit (e.g. `--limit 3` for a smoke test, "
                "`--limit 75` for the full set).",
                file=sys.stderr,
            )
            return 2
        _run_all(args.limit, args.experiment, args.description, args.run_report)
        return 0

    if args.step == "load":
        load.run(limit=args.limit)
    elif args.step == "generate":
        generate.run(limit=args.limit)
    elif args.step == "grade":
        grade.run(limit=args.limit)
    elif args.step == "classify":
        classify.run(limit=args.limit)
    elif args.step == "validate":
        validate.run(mode=args.mode)
    elif args.step == "aggregate":
        aggregate.run(
            limit=args.limit,
            experiment=args.experiment,
            description=args.description,
            run_report=args.run_report,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
