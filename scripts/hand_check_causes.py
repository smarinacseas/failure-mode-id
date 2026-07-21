"""Blinded hand-check tool for validating E08's root-cause labels.

Two subcommands, mirroring the repo's `validate sample/score` pattern:

  build --rule "<text>" [--seed 20260715] [--out <dir>]
      Cross-checks dashboard export integrity against the post-repair
      responses.jsonl, draws a stratified n=40 sample from the three biggest
      root-cause pools, and writes a BLINDED worksheet.md (prompt + criterion
      + full response + a blank `human_label:` line, nothing that reveals the
      classifier's call) plus an answer_key.json for later scoring.

  score --dir <dir>
      Parses the filled worksheet, rejects blanks / out-of-vocabulary labels,
      and reports raw agreement, per-cause agreement, the full human×classifier
      confusion matrix (with the constraint_unaddressed↔execution_slip cell
      called out), and a disagreement list.

Read-only over the dashboard and responses.jsonl. The ONLY thing it writes is
the worksheet / answer_key / score_summary inside the chosen output dir.

The decision rule is stamped into the worksheet header BEFORE any label
exists, so the labeller cannot tune the rule to the answers.

Data source of record: dashboard/E08-llama3-2-3b-cc75.json (tracked).
Integrity input: runs/E08-llama3-2-3b-cc75/responses/llama-3b.jsonl.

Criterion indexing convention: failure_analysis rows carry a 1-based
`criterion_index` (a string), so the join to the prompt's criterion list is
`criteria[int(criterion_index) - 1]`. This is verified, not assumed: the
1-based join lands on a consensus-FAIL criterion for all 882 real rows,
whereas the 0-based reading lands on a fail for only 585. `build` re-asserts
that invariant on every sampled row (see `assert_consensus_fail`) so the base
convention is self-enforcing against future schema drift rather than a
one-off investigation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Fixed contract constants --------------------------------------------
MODEL = "llama-3b"

# Pool sizes verified against the real 882-row failure_analysis. Asserted at
# build time as a schema-drift guard: if the diagnosis pools have shifted, the
# dashboard is not the run we contracted against and the sample frame is wrong.
EXPECTED_POOL_SIZES = {
    "execution_slip": 321,
    "constraint_unaddressed": 254,
    "constraint_misread": 127,
}

# Stratified sample: how many rows per stratum. Order is fixed so the single
# seeded RNG draws deterministically.
STRATA = (
    ("constraint_unaddressed", 15),
    ("execution_slip", 15),
    ("constraint_misread", 10),
)
STRATA_CAUSES = [cause for cause, _ in STRATA]
SAMPLE_N = sum(k for _, k in STRATA)

# The A<->B confusion cell called out explicitly in scoring.
A_CAUSE = "constraint_unaddressed"
B_CAUSE = "execution_slip"

SAMPLE_SEED_DEFAULT = 20260715
N_INTEGRITY_IDS = 3

BODY_DELIMITER = "===== WORKSHEET ROWS: do not edit anything above this line ====="

DEFAULT_DASHBOARD = REPO_ROOT / "dashboard" / "E08-llama3-2-3b-cc75.json"
DEFAULT_RESPONSES = REPO_ROOT / "runs" / "E08-llama3-2-3b-cc75" / "responses" / "llama-3b.jsonl"
DEFAULT_OUT = REPO_ROOT / "runs" / "E08-llama3-2-3b-cc75" / "hand_check"


class HandCheckError(Exception):
    """Any abort condition, surfaced as a clear CLI message, exit code 1."""


# --- Loading / accessors -------------------------------------------------

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def get_rows(dashboard: dict) -> list[dict]:
    return dashboard["failure_analysis"]["rows"]


def get_vocab(dashboard: dict) -> list[str]:
    """Permitted label set = every taxonomy key (includes other, judge_suspect)."""
    return [t["key"] for t in dashboard["failure_analysis"]["taxonomy"]]


def prompts_index(dashboard: dict) -> dict[str, dict]:
    return {p["id"]: p for p in dashboard["prompts"]}


def criterion_text_for(row: dict, pidx: dict[str, dict]) -> str:
    """Join a failure-analysis row to its criterion text.

    criterion_index is 1-based and may arrive as a string; coerce and shift.
    """
    prompt = pidx[row["id"]]
    idx = int(row["criterion_index"]) - 1
    return prompt["criteria"][idx]["text"]


def response_for(row: dict, pidx: dict[str, dict]) -> str:
    return pidx[row["id"]]["responses"][MODEL]


def criterion_pass_for(row: dict, pidx: dict[str, dict]):
    """The consensus pass/fail for the joined (1-based) criterion, or None if
    the results structure is absent."""
    prompt = pidx[row["id"]]
    idx = int(row["criterion_index"]) - 1
    crit = prompt["criteria"][idx]
    return crit.get("results", {}).get(MODEL, {}).get("pass")


# --- Guards --------------------------------------------------------------

def assert_pool_sizes(rows: list[dict], expected: dict[str, int] = EXPECTED_POOL_SIZES) -> None:
    counts = Counter(r["root_cause"] for r in rows)
    mismatched = {
        cause: (counts.get(cause, 0), want)
        for cause, want in expected.items()
        if counts.get(cause, 0) != want
    }
    if mismatched:
        detail = ", ".join(
            f"{cause}: found {found}, expected {want}"
            for cause, (found, want) in mismatched.items()
        )
        raise HandCheckError(
            "Pool-size guard failed (schema drift): the dashboard's diagnosis "
            f"pools are not the run we contracted against. {detail}. "
            "Refusing to build a sample against a drifted frame."
        )


# --- Sample frame + integrity -------------------------------------------

def assert_consensus_fail(sampled: list[dict], pidx: dict[str, dict]) -> None:
    """Every sampled row's 1-based join must land on a consensus-FAIL criterion.

    A row diagnoses a *failure*, so `criteria[int(criterion_index)-1]` must have
    `results.llama-3b.pass is False`. Anything else (a pass, or a missing
    results block) means the 1-based convention no longer holds for this export
    Abort rather than build a worksheet against a mis-joined frame.
    """
    violations = []
    for n, row in enumerate(sampled, start=1):
        pass_val = criterion_pass_for(row, pidx)
        if pass_val is not False:
            violations.append(
                f"row {n} (id={row['id']} criterion_index={row['criterion_index']} "
                f"root_cause={row['root_cause']}): joined criterion pass={pass_val!r}"
            )
    if violations:
        raise HandCheckError(
            "Consensus-FAIL invariant broken: the 1-based criterion join no "
            "longer lands on a failed criterion, so the index convention or the "
            "dashboard export has drifted. Offending rows:\n  "
            + "\n  ".join(violations)
        )


def build_frame(rows: list[dict]) -> dict[str, list[dict]]:
    """The three strata pools, each sorted by (id, criterion_index).

    Sorting on the row's identity before any RNG draw makes the fixed-seed
    sample a pure function of pool CONTENT, never of dashboard row order:
    the order-stability convention from pipeline/validate.py.
    """
    frame: dict[str, list[dict]] = {cause: [] for cause in STRATA_CAUSES}
    for r in rows:
        if r["root_cause"] in frame:
            frame[r["root_cause"]].append(r)
    for cause in frame:
        frame[cause].sort(key=lambda r: (r["id"], int(r["criterion_index"])))
    return frame


def frame_rows(rows: list[dict]) -> list[dict]:
    """Flat, sorted union of the three strata pools (for the integrity draw)."""
    frame = build_frame(rows)
    return [r for cause in STRATA_CAUSES for r in frame[cause]]


def integrity_check_ids(frame: list[dict], seed: int, k: int = N_INTEGRITY_IDS) -> list[str]:
    """Deterministically pick k distinct prompt ids from the sample frame.

    Uses its own RNG so it does not perturb the sampling RNG.
    """
    ids = sorted({r["id"] for r in frame})
    rng = random.Random(seed)
    return rng.sample(ids, min(k, len(ids)))


def run_integrity_check(pidx: dict[str, dict], responses: dict[str, str], ids: list[str]) -> None:
    """Assert dashboard response bytes match responses.jsonl for each id."""
    for pid in ids:
        dash = pidx[pid]["responses"][MODEL]
        jl = responses.get(pid)
        if jl is None:
            raise HandCheckError(
                f"Integrity check: id {pid!r} is in the dashboard but missing from "
                f"{DEFAULT_RESPONSES.name}. The dashboard export is out of sync; "
                "re-run `main.py aggregate` before any hand-check."
            )
        if dash != jl:
            raise HandCheckError(
                f"Integrity check: response for id {pid!r} differs between the "
                "dashboard and responses.jsonl. The dashboard export is STALE "
                "relative to post-truncation-repair state, re-run "
                "`main.py aggregate` before any hand-check."
            )


# --- Sampling ------------------------------------------------------------

def stratified_sample(rows: list[dict], seed: int) -> list[dict]:
    """Draw 15/15/10 from the three pools, then shuffle: one seeded RNG for
    both the per-stratum draw and the final worksheet shuffle."""
    frame = build_frame(rows)
    rng = random.Random(seed)
    sampled: list[dict] = []
    for cause, k in STRATA:
        pool = frame[cause]
        if len(pool) < k:
            raise HandCheckError(
                f"Pool {cause!r} has only {len(pool)} rows, need {k} to sample."
            )
        sampled.extend(rng.sample(pool, k))
    rng.shuffle(sampled)
    return sampled


# --- Worksheet / answer key rendering ------------------------------------

def render_worksheet(
    sampled: list[dict], pidx: dict[str, dict], rule: str, seed: int,
    vocab: list[str], timestamp: str,
) -> str:
    per_stratum = ", ".join(f"{cause}={k}" for cause, k in STRATA)
    header = [
        "# Blinded root-cause hand-check worksheet",
        "",
        f"- timestamp: {timestamp}",
        f"- seed: {seed}",
        f"- n: {SAMPLE_N} ({per_stratum})",
        "",
        "## Decision rule (stamped before any label exists)",
        "",
        rule,
        "",
        "## Permitted labels",
        "",
        "Write exactly one of these values after `human_label:` on each row:",
        "",
    ]
    header += [f"- {key}" for key in vocab]
    header += ["", BODY_DELIMITER, ""]

    rows_md: list[str] = []
    for n, row in enumerate(sampled, start=1):
        rows_md += [
            f"## Row {n}",
            "",
            "### Prompt",
            "",
            pidx[row["id"]]["prompt_text"],
            "",
            "### Criterion",
            "",
            criterion_text_for(row, pidx),
            "",
            "### Response",
            "",
            response_for(row, pidx),
            "",
            "human_label: ",
            "",
        ]
    return "\n".join(header + rows_md) + "\n"


def build_answer_key(
    sampled: list[dict], rule: str, seed: int, timestamp: str, vocab: list[str],
) -> dict:
    return {
        "rule": rule,
        "seed": seed,
        "timestamp": timestamp,
        "vocab": vocab,
        "rows": {
            str(n): {
                "id": row["id"],
                "criterion_index": row["criterion_index"],
                "root_cause": row["root_cause"],
            }
            for n, row in enumerate(sampled, start=1)
        },
    }


def build_worksheet(
    dashboard: dict, responses: dict[str, str], rule: str, seed: int,
    out_dir: Path, expected_pool_sizes: dict[str, int] = EXPECTED_POOL_SIZES,
    timestamp: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    worksheet_path = out_dir / "worksheet.md"
    if worksheet_path.exists():
        raise HandCheckError(
            f"{worksheet_path} already exists, refusing to overwrite a labeling "
            "session in progress. Delete the directory manually to start fresh."
        )

    rows = get_rows(dashboard)
    pidx = prompts_index(dashboard)
    vocab = get_vocab(dashboard)

    # 1. schema-drift guard
    assert_pool_sizes(rows, expected=expected_pool_sizes)

    # 2. integrity cross-check (before sampling)
    frame = frame_rows(rows)
    ids = integrity_check_ids(frame, seed=seed)
    run_integrity_check(pidx, responses, ids)

    # 3. stratified sample
    sampled = stratified_sample(rows, seed=seed)

    # 4. self-enforcing 1-based join check: every sampled row must land on a
    #    consensus-FAIL criterion.
    assert_consensus_fail(sampled, pidx)

    ts = timestamp or datetime.now(timezone.utc).isoformat()
    worksheet = render_worksheet(sampled, pidx, rule, seed, vocab, ts)
    answer_key = build_answer_key(sampled, rule, seed, ts, vocab)

    out_dir.mkdir(parents=True, exist_ok=True)
    worksheet_path.write_text(worksheet, encoding="utf-8")
    (out_dir / "answer_key.json").write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


# --- Scoring -------------------------------------------------------------

_ROW_RE = re.compile(r"(?m)^## Row (\d+)\s*$")
_LABEL_RE = re.compile(r"(?m)^human_label:[ \t]*(.*?)[ \t]*$")


def parse_human_labels(worksheet_text: str, vocab: list[str]) -> dict[int, str]:
    """Map row number -> filled human_label; reject blanks and out-of-vocab.

    Only the body below BODY_DELIMITER is parsed, so the delimiter never sees
    the vocab list in the header as a label.
    """
    body = worksheet_text.split(BODY_DELIMITER, 1)[-1]
    matches = list(_ROW_RE.finditer(body))
    labels: dict[int, str] = {}
    blanks: list[int] = []
    invalid: list[tuple[int, str]] = []
    allowed = set(vocab)
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block = body[start:end]
        found = _LABEL_RE.findall(block)
        value = found[-1].strip() if found else ""
        if not value:
            blanks.append(n)
        elif value not in allowed:
            invalid.append((n, value))
        else:
            labels[n] = value
    problems = []
    if blanks:
        problems.append(f"{len(blanks)} unfilled row(s): {blanks}")
    if invalid:
        shown = ", ".join(f"row {n}={v!r}" for n, v in invalid)
        problems.append(f"{len(invalid)} out-of-vocabulary label(s): {shown}")
    if problems:
        raise HandCheckError("Cannot score: " + "; ".join(problems) + ".")
    return labels


def compute_scores(answer_key: dict, human_labels: dict[int, str]) -> dict:
    rows = answer_key["rows"]
    records = []
    for rn_str, info in sorted(rows.items(), key=lambda kv: int(kv[0])):
        n = int(rn_str)
        records.append({
            "row": n,
            "id": info["id"],
            "criterion_index": info["criterion_index"],
            "classifier": info["root_cause"],
            "human": human_labels[n],
        })

    total = len(records)
    agree = sum(1 for r in records if r["human"] == r["classifier"])

    per_cause: dict[str, dict] = {}
    for cause in STRATA_CAUSES:
        sub = [r for r in records if r["classifier"] == cause]
        a = sum(1 for r in sub if r["human"] == cause)
        per_cause[cause] = {
            "n": len(sub),
            "agree": a,
            "agreement": (a / len(sub)) if sub else None,
        }

    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        confusion[(r["human"], r["classifier"])] += 1

    a_to_b = confusion.get((B_CAUSE, A_CAUSE), 0)  # classifier=A, human=B
    b_to_a = confusion.get((A_CAUSE, B_CAUSE), 0)  # classifier=B, human=A
    ab_cell = {
        f"classifier_{A_CAUSE}__human_{B_CAUSE}": a_to_b,
        f"classifier_{B_CAUSE}__human_{A_CAUSE}": b_to_a,
        "total": a_to_b + b_to_a,
    }

    disagreements = [
        {"row": r["row"], "id": r["id"], "criterion_index": r["criterion_index"],
         "classifier": r["classifier"], "human": r["human"]}
        for r in records if r["human"] != r["classifier"]
    ]

    return {
        "rule": answer_key.get("rule"),
        "seed": answer_key.get("seed"),
        "timestamp": answer_key.get("timestamp"),
        "n": total,
        "agree": agree,
        "overall_agreement": (agree / total) if total else 0.0,
        "per_cause": per_cause,
        "confusion": dict(confusion),
        "ab_cell": ab_cell,
        "disagreements": disagreements,
    }


def _confusion_json(confusion: dict[tuple[str, str], int]) -> list[dict]:
    """Tuple-keyed confusion -> JSON-serialisable list of cells."""
    return [
        {"human": h, "classifier": c, "count": n}
        for (h, c), n in sorted(confusion.items())
    ]


def score_dir(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    ak_path = out_dir / "answer_key.json"
    ws_path = out_dir / "worksheet.md"
    if not ak_path.exists() or not ws_path.exists():
        raise HandCheckError(
            f"Expected worksheet.md and answer_key.json in {out_dir}. "
            "Run `build` first."
        )
    answer_key = load_json(ak_path)
    vocab = answer_key.get("vocab") or []
    human_labels = parse_human_labels(ws_path.read_text(encoding="utf-8"), vocab)

    expected_rows = set(int(k) for k in answer_key["rows"])
    if set(human_labels) != expected_rows:
        missing = sorted(expected_rows - set(human_labels))
        extra = sorted(set(human_labels) - expected_rows)
        raise HandCheckError(
            f"Worksheet rows do not match answer key (missing={missing}, extra={extra})."
        )

    summary = compute_scores(answer_key, human_labels)
    summary["scored_at"] = datetime.now(timezone.utc).isoformat()

    out = dict(summary)
    out["confusion"] = _confusion_json(summary["confusion"])
    (out_dir / "score_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _print_score(summary: dict, out_dir: Path) -> None:
    print("Decision rule (stamped at build):")
    print(f"  {summary['rule']}")
    print()
    print(f"hand-check score: {summary['agree']}/{summary['n']} raw agreement "
          f"({100.0 * summary['overall_agreement']:.1f}%)")
    print("per-cause agreement:")
    for cause in STRATA_CAUSES:
        pc = summary["per_cause"][cause]
        pct = f"{100.0 * pc['agreement']:.1f}%" if pc["agreement"] is not None else "n/a"
        print(f"  {cause}: {pc['agree']}/{pc['n']} ({pct})")
    print("confusion matrix (human × classifier):")
    for (h, c), n in sorted(summary["confusion"].items()):
        print(f"  human={h:<24} classifier={c:<24} {n}")
    ab = summary["ab_cell"]
    print(f"A↔B cell ({A_CAUSE} ↔ {B_CAUSE}), total {ab['total']}:")
    print(f"  classifier={A_CAUSE}, human={B_CAUSE}: {ab[f'classifier_{A_CAUSE}__human_{B_CAUSE}']}")
    print(f"  classifier={B_CAUSE}, human={A_CAUSE}: {ab[f'classifier_{B_CAUSE}__human_{A_CAUSE}']}")
    if summary["disagreements"]:
        print(f"disagreements ({len(summary['disagreements'])}):")
        for d in summary["disagreements"]:
            print(f"  row {d['row']} ({d['id']} c{d['criterion_index']}): "
                  f"classifier={d['classifier']} vs human={d['human']}")
    print(f"wrote {out_dir / 'score_summary.json'}")


# --- CLI -----------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:
    dashboard = load_json(args.dashboard)
    responses = {r["id"]: r["response"] for r in read_jsonl(args.responses)}
    out_dir = build_worksheet(
        dashboard, responses, rule=args.rule, seed=args.seed, out_dir=args.out,
    )
    print(f"integrity check passed for {N_INTEGRITY_IDS} seeded prompt ids.")
    print(f"pool-size guard passed ({EXPECTED_POOL_SIZES}).")
    print(f"wrote {out_dir / 'worksheet.md'} and {out_dir / 'answer_key.json'} "
          f"({SAMPLE_N} rows).")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    summary = score_dir(args.dir)
    _print_score(summary, Path(args.dir))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="draw a blinded stratified sample worksheet")
    b.add_argument("--rule", required=True, help="decision rule, stamped verbatim")
    b.add_argument("--seed", type=int, default=SAMPLE_SEED_DEFAULT)
    b.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    b.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    b.add_argument("--out", type=Path, default=DEFAULT_OUT)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="score a filled worksheet")
    s.add_argument("--dir", type=Path, default=DEFAULT_OUT)
    s.set_defaults(func=cmd_score)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except HandCheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
