"""Pull committed human labels back into the local run folder.

dashboard/label.html commits the labeled judge_validation.json to the
`labels` branch on GitHub (runs/ is gitignored, so labels live on their own
branch rather than in the working tree's history). This script downloads that
file and writes it to runs/<slug>/judge_validation.json — after which the
normal scoring step works:

    uv run python labeling/pull.py --experiment E08-llama3-2-3b-cc75
    uv run python main.py validate --experiment E08-llama3-2-3b-cc75 --mode score

The previous local file is kept as judge_validation.json.bak.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = {"judge", "model", "id", "criterion_index", "criterion_text",
                 "prompt_text", "response_excerpt", "judge_verdict",
                 "judge_reason", "human"}


def pull(owner: str, repo: str, branch: str, slug: str) -> None:
    path = f"runs/{slug}/judge_validation.json"
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as res:
            raw = res.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise SystemExit(
            f"GET {url} → {e.code}. No labels committed yet? "
            f"(label.html creates the `{branch}` branch on first commit)"
        )

    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Downloaded file is not a non-empty JSON array — refusing to write.")
    for i, r in enumerate(rows):
        missing = REQUIRED_KEYS - set(r)
        if missing:
            raise SystemExit(f"Row {i} missing keys {sorted(missing)} — refusing to write.")
        if str(r.get("human", "")).strip().upper() not in {"", "PASS", "FAIL"}:
            raise SystemExit(f"Row {i} has invalid human verdict {r['human']!r} — refusing to write.")

    filled = sum(1 for r in rows if str(r["human"]).strip())
    dst = REPO_ROOT / path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.with_suffix(".json.bak").write_text(dst.read_text(encoding="utf-8"), encoding="utf-8")
    dst.write_text(raw, encoding="utf-8")
    print(f"labeling pull: {filled}/{len(rows)} human verdicts → {path} "
          f"(previous file saved as judge_validation.json.bak)")
    print(f"next: uv run python main.py validate --experiment {slug} --mode score")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiment", default="E08-llama3-2-3b-cc75",
                    help="Run slug under runs/ (default: E08-llama3-2-3b-cc75).")
    ap.add_argument("--repo", default="smarinacseas/failure-mode-id",
                    help="owner/name (default: smarinacseas/failure-mode-id).")
    ap.add_argument("--branch", default="labels",
                    help="Branch the labels were committed to (default: labels).")
    args = ap.parse_args()
    owner, _, repo = args.repo.partition("/")
    if not repo:
        sys.exit("--repo must be owner/name")
    pull(owner, repo, args.branch, args.experiment)


if __name__ == "__main__":
    main()
