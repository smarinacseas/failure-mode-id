"""Sync experiment deliverables into the ConstraintLens dashboard folder.

The dashboard is the unpacked Claude Design output at `dashboard/index.html`
+ `dashboard/support.js`. Its embedded Logic fetches `./runs.json` on load
and, on selection, fetches each run's own JSON at `./<path>` (all relative
to `index.html`). So this script simply:

  · Copies every `outputs/experiments/<slug>.json` → `dashboard/<slug>.json`.
  · Rebuilds `dashboard/runs.json` in the shape the design's Logic expects:
    `{"runs": [{"id", "label", "date", "path"}, …]}` — ordered by
    experiment number ascending, newest last so the top-of-list stays
    stable.

Idempotent. Safe to run at the end of every aggregate step (`pipeline/
aggregate.py` calls it automatically for tagged runs) or manually as
`uv run python scripts/dashboard_sync.py`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pipeline.run_config import track_for_slug  # noqa: E402 — needs REPO_ROOT on sys.path

DEFAULT_SRC = REPO_ROOT / "outputs" / "experiments"
DEFAULT_DST = REPO_ROOT / "dashboard"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_files(src: Path) -> list[Path]:
    """Every experiment deliverable (excludes the upstream `index.json`)."""
    if not src.exists():
        return []
    return sorted(p for p in src.glob("*.json") if p.name != "index.json")


def _run_entry(payload: dict, filename: str) -> dict:
    """Compact `{id, label, date, path}` — what the design's dropdown renders.

    Label bakes the axis under investigation (`slug`) plus the shape of the
    run (`Np × Mm`) so the dropdown is scannable without opening each run.
    Date is truncated to `YYYY-MM-DD` so the mono-spaced meta line stays
    compact.
    """
    meta = payload.get("meta", {})
    experiment = meta.get("experiment", {})
    counts = meta.get("counts", {})

    slug = experiment.get("slug") or filename.rsplit(".json", 1)[0]
    n_prompts = counts.get("n_prompts", 0)
    n_models = counts.get("n_models", 0)
    description = (experiment.get("description") or "").strip()
    axis = description.split(".", 1)[0] if description else slug

    return {
        "id": slug,
        "label": f"{slug} — {axis[:80]}" if description else slug,
        "date": (experiment.get("run_date") or "")[:10],
        "path": f"./{filename}",
        "n_prompts": n_prompts,
        "n_models": n_models,
    }


_INDEX_FILES = {"runs.json", "training.json"}


def _clean_stale(dst: Path, keep: set[str]) -> list[str]:
    """Remove `dashboard/*.json` files that don't correspond to a current
    experiment. Preserves the run-index files (runs.json, training.json)."""
    removed: list[str] = []
    for jf in dst.glob("*.json"):
        if jf.name in _INDEX_FILES or jf.name in keep:
            continue
        jf.unlink()
        removed.append(jf.name)
    return removed


def sync(src: Path = DEFAULT_SRC, dst: Path = DEFAULT_DST) -> dict:
    """Copy each deliverable and rewrite `runs.json`. Returns a small
    result dict for logging by the caller."""
    dst.mkdir(parents=True, exist_ok=True)
    files = _experiment_files(src)

    entries: list[dict] = []
    kept_names: set[str] = set()
    for src_path in files:
        payload = _load_json(src_path)
        dst_name = src_path.name
        shutil.copyfile(src_path, dst / dst_name)
        entries.append(_run_entry(payload, dst_name))
        kept_names.add(dst_name)

    removed = _clean_stale(dst, kept_names)

    def _write_index(name: str, track: str) -> None:
        rows = sorted((e for e in entries if track_for_slug(e["id"]) == track),
                      key=lambda e: e["id"])
        (dst / name).write_text(
            json.dumps({"runs": rows, "synced_at": datetime.now(timezone.utc).isoformat()},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    _write_index("runs.json", "analysis")
    _write_index("training.json", "training")

    return {
        "copied": len(files),
        "removed": removed,
        "target": str(dst),
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dashboard_sync.py",
        description="Sync outputs/experiments/*.json → dashboard/ (index.html + runs.json).",
    )
    p.add_argument("--experiments-dir", default=str(DEFAULT_SRC))
    p.add_argument("--target", default=str(DEFAULT_DST))
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = sync(Path(args.experiments_dir), Path(args.target))
    if not args.quiet:
        msg = f"dashboard_sync: copied {result['copied']} experiment(s) → {result['target']}"
        if result["removed"]:
            msg += f" (removed stale: {', '.join(result['removed'])})"
        print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
