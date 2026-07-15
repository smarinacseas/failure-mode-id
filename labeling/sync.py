"""Build the dashboard labeling bundle for a run's judge-validation sample.

`main.py validate --mode sample` writes runs/<slug>/judge_validation.json with
60 rows whose `response_excerpt` is truncated to 800 chars. The labeling page
(dashboard/label.html) needs the FULL candidate response, and GitHub Pages
only serves the dashboard/ folder — runs/ is gitignored. So this script joins
each validation row with its full response from runs/<slug>/responses/
<model>.jsonl and writes a static bundle the page can fetch:

    dashboard/labeling/<slug>.json

The bundle keeps every original judge_validation.json field (the page needs
them to reconstruct a `validate --mode score`-compatible file on commit) and
adds one field per row: `full_response`. Blinding is enforced by the UI —
judge_verdict / judge_reason are never rendered before the human votes — not
by stripping them here; the repo is public either way.

Idempotent. Rerun after any `validate --mode sample`:

    uv run python labeling/sync.py --experiment E08-llama3-2-3b-cc75
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BUNDLE_SCHEMA = "labeling-bundle-1"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_bundle(slug: str) -> Path:
    run_dir = REPO_ROOT / "runs" / slug
    src = run_dir / "judge_validation.json"
    if not src.exists():
        raise SystemExit(
            f"{src} missing — run `main.py validate --experiment {slug} --mode sample` first."
        )
    rows = json.loads(src.read_text(encoding="utf-8"))

    # Load full responses once per model referenced by the sample.
    responses: dict[str, dict[str, str]] = {}
    for model in sorted({r["model"] for r in rows}):
        path = run_dir / "responses" / f"{model}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} missing — cannot bundle full responses.")
        responses[model] = {r["id"]: r.get("response", "") for r in _read_jsonl(path)}

    out_rows = []
    missing = []
    for r in rows:
        full = responses.get(r["model"], {}).get(r["id"])
        if full is None:
            missing.append((r["model"], r["id"]))
            full = r.get("response_excerpt", "")
        out_rows.append({**r, "full_response": full})
    if missing:
        print(f"WARN: no full response for {len(missing)} rows: {missing[:5]}…", file=sys.stderr)

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "run": slug,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"runs/{slug}/judge_validation.json",
        "n_rows": len(out_rows),
        "rows": out_rows,
    }
    dst = REPO_ROOT / "dashboard" / "labeling" / f"{slug}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"labeling sync: wrote {len(out_rows)} rows → {dst.relative_to(REPO_ROOT)}")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", default="E08-llama3-2-3b-cc75",
                    help="Run slug under runs/ (default: E08-llama3-2-3b-cc75).")
    args = ap.parse_args()
    build_bundle(args.experiment)


if __name__ == "__main__":
    main()
