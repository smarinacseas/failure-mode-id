"""Sidecar degenerate-output probe for E07 (spec 2026-07-08, section 7).

Repeat-draws CIF-012 under E07's frozen treatment on the two models that have
hosted degenerate_output (qwen-9b in E05, qwen-35b in E06), then runs decode
forensics over the draws, E07's own responses, and E05's original CIF-012
response. Writes ONLY sidecar paths under runs/<slug>/probe/ — never the
pipeline-owned responses/ files — so the frozen run and its resume logic are
untouched. Deliberately does not call run_config.resolve(): on a missing slug
resolve() would freeze DEFAULT params; this script must fail instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from pipeline._decode_health import (  # noqa: F401 — re-exported for tests/tools
    MIN_REPEATS,
    NGRAM,
    WINDOW,
    detect_repetition_loop,
)
from pipeline._io import append_jsonl, read_jsonl
from pipeline.generate import _generate_one
from pipeline.run_config import RunConfig

SLUG = "E07-reasoning-full75"
PROMPT_ID = "CIF-012"
PROBE_MODELS = ("qwen-9b", "qwen-35b")
DRAWS_PER_MODEL = 5


def forensic_row(source: str, model: str, rec: dict) -> dict:
    """Mechanical decode forensics for one stored response record."""
    reasoning = rec.get("reasoning", "")
    content = rec.get("response", "")
    return {
        "source": source,
        "model": model,
        "id": rec.get("id", PROMPT_ID),
        "finish_reason": rec.get("finish_reason"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "content_loop": detect_repetition_loop(content),
        "reasoning_loop": detect_repetition_loop(reasoning),
    }


def load_frozen_cfg(slug: str) -> RunConfig:
    """Read-only load of a slug's frozen params. Hard-fails if absent —
    calling resolve() here would freeze DEFAULT params for the slug."""
    path = config.RUNS_DIR / slug / "experiment.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run the E07 pipeline first (its first "
            "invocation freezes the params this probe must reuse)."
        )
    frozen = json.loads(path.read_text(encoding="utf-8"))["params"]
    return RunConfig.from_json_dict(slug, frozen)


def run_draws(cfg: RunConfig, prompt: str, out_path: Path,
              models: tuple[str, ...] = PROBE_MODELS,
              k: int = DRAWS_PER_MODEL) -> int:
    """k same-config draws per model; resume-safe on (model, draw);
    continue-on-error per draw, mirroring the pipeline's item semantics."""
    done = {(r["model"], r["draw"]) for r in read_jsonl(out_path)}
    rejected_path = out_path.parent / "rejected.jsonl"
    made = 0
    for key in models:
        model_id = cfg.candidates[key]
        for draw in range(1, k + 1):
            if (key, draw) in done:
                continue

            def _capture(fields: dict, key=key, draw=draw) -> None:
                append_jsonl(rejected_path,
                             {"model": key, "draw": draw, **fields})

            try:
                fields = _generate_one(cfg, model_id, prompt,
                                       on_reject=_capture)
            except Exception as e:  # noqa: BLE001 — per-draw continue-on-error
                print(f"probe draw {key}#{draw} failed: {type(e).__name__}: {e}")
                continue
            append_jsonl(out_path, {"model": key, "draw": draw, **fields})
            made += 1
    return made


def print_markdown(rows: list[dict]) -> None:
    print("| source | model | id | finish | reasoning chars | content chars "
          "| loop | period | onset |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        loop = (r["content_loop"] if r["content_loop"]["looping"]
                else r["reasoning_loop"])
        print(f"| {r['source']} | {r['model']} | {r['id']} "
              f"| {r['finish_reason']} | {r['reasoning_chars']:,} "
              f"| {r['content_chars']:,} "
              f"| {'YES' if loop['looping'] else 'no'} "
              f"| {loop['period'] if loop['period'] is not None else '—'} "
              f"| {loop['onset'] if loop['onset'] is not None else '—'} |")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Degenerate-output probe (sidecar to E07; spec section 7)")
    ap.add_argument("--slug", default=SLUG)
    ap.add_argument("--skip-generate", action="store_true",
                    help="No API calls; forensics over existing artifacts only.")
    args = ap.parse_args()

    cfg = load_frozen_cfg(args.slug)
    probe_dir = cfg.run_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    draws_path = probe_dir / "draws.jsonl"

    if not args.skip_generate:
        record = next(r for r in read_jsonl(config.DATA_JSONL)
                      if r["id"] == PROMPT_ID)
        made = run_draws(cfg, record["prompt"], draws_path)
        print(f"probe: {made} new draw(s) -> {draws_path}")

    rows = [forensic_row(f"probe-draw-{r['draw']}", r["model"], r)
            for r in read_jsonl(draws_path)]
    # E07's own responses: CIF-012 always; anything truncated or looping too.
    for key in cfg.candidates:
        for r in read_jsonl(cfg.responses_path(key)):
            fr = forensic_row(args.slug, key, r)
            if (r["id"] == PROMPT_ID or fr["finish_reason"] != "stop"
                    or fr["content_loop"]["looping"]
                    or fr["reasoning_loop"]["looping"]):
                rows.append(fr)
    # E05's original degenerate response, for side-by-side comparison.
    e05_path = (config.RUNS_DIR / "E05-reasoning-rand20p"
                / "responses" / "qwen-9b.jsonl")
    rows.extend(forensic_row("E05-reasoning-rand20p", "qwen-9b", r)
                for r in read_jsonl(e05_path) if r["id"] == PROMPT_ID)

    out = probe_dir / "forensics.json"
    payload = {
        "prompt_id": PROMPT_ID,
        "heuristic": {"window": WINDOW, "ngram": NGRAM,
                      "min_repeats": MIN_REPEATS},
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"probe: forensics -> {out}\n")
    print_markdown(rows)


if __name__ == "__main__":
    main()
