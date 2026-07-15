"""E08 degenerate_output ablation (synthesis Rec 2, discrepancy_drill).

E08 diagnosed 107 degenerate_output criteria concentrated in 13 prompts.
Rec 2: rerun those prompts at temperature 0.2 and 0.6 with finish_reason
logging and check — via the deterministic decode-health detector
(pipeline/_decode_health.detect_repetition_loop) — whether degenerate output
is a decoding artifact rather than a stable model behavior.

Scope: generation + decode telemetry ONLY. No judge, panel, or classifier
calls. ~26 candidate calls at 1536 max_tokens ≈ $0.02.

Design notes
------------
* Reuses the pipeline's generation call verbatim: pipeline.generate._generate_one
  (streamed, deadline-guarded, retry()-wrapped, finish_reason + per-chunk
  provider capture, on_reject hook). Importing the private helper is a
  deliberate experiments/-side choice: the frozen treatment code path must be
  IDENTICAL to the census run, and main.py has no prompt-ID subset capability.
* All decode params except temperature come from the frozen E08 config
  (runs/E08-llama3-2-3b-cc75/experiment.json): max_tokens=1536, reasoning off,
  seed=20260715, timeout 300 s. Holding max_tokens at 1536 is intentional —
  11/13 affected prompts finished with finish_reason=length in the census, so
  raising the cap would confound the temperature ablation.
* Provider pin (§0.2, bf16/fp16): probed at startup with a 16-token call.
  The pin may no longer resolve (the census was served by Parasail; the
  endpoint lineup drifts). On probe failure the arms run UNPINNED — best
  available route — and the manifest + every response record log which
  provider actually served each call. OpenRouter does not expose the serving
  quantization for this model (endpoint listing says "unknown"), so provider
  identity is the only precision signal available.
* Layout mirrors runs/<slug>/ conventions, one directory per arm:
      experiments/E08/ablation-degenerate/
        ablation_manifest.json
        responses/<arm>/llama-3b.jsonl      # {"id", "response", "finish_reason",
                                            #  "provider"?, "loop_failure"?}
        rejected/<arm>/llama-3b.jsonl       # doomed attempts (on_reject hook)
        decode_health.json                  # detector census: baseline + arms
        findings.md                         # auto-generated summary
* Resumable exactly like pipeline/generate.py: done ids are re-derived from
  the response files, so a rerun only fills gaps.

Usage (from repo root):
    uv run python experiments/E08/ablation_degenerate.py            # generate + analyze
    uv run python experiments/E08/ablation_degenerate.py --analyze-only
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from pipeline._decode_health import (  # noqa: E402
    MIN_REPEATS, NGRAM, WINDOW, detect_repetition_loop,
)
from pipeline._io import append_jsonl, read_jsonl  # noqa: E402
from pipeline.generate import _generate_one  # noqa: E402
from pipeline.run_config import RunConfig  # noqa: E402

SLUG = "E08-llama3-2-3b-cc75"
CANDIDATE_KEY = "llama-3b"
ARMS: dict[str, float] = {"t02": 0.2, "t06": 0.6}
ABLATION_DIR = ROOT / "experiments" / "E08" / "ablation-degenerate"
# 1536-token completions stream in well under this; generous vs the census
# per-call p-max, tight vs the 2700 s census deadline sized for 64k budgets.
DEFAULT_DEADLINE_S = 600.0


# --------------------------------------------------------------------------
# Inputs derived from the frozen census artifacts (never hardcoded).
# --------------------------------------------------------------------------

def load_frozen_cfg() -> RunConfig:
    frozen = json.loads(
        (config.RUNS_DIR / SLUG / "experiment.json").read_text(encoding="utf-8")
    )["params"]
    return RunConfig.from_json_dict(SLUG, frozen)


def degenerate_prompt_ids(cfg: RunConfig) -> dict[str, int]:
    """{prompt_id: n degenerate_output-diagnosed criteria} from the census
    diagnosis artifact — the selection is re-derived, not hardcoded."""
    counts: Counter[str] = Counter()
    for rec in read_jsonl(cfg.diagnosis_path(CANDIDATE_KEY)):
        for d in rec.get("diagnoses", []):
            if d.get("root_cause") == "degenerate_output":
                counts[rec["id"]] += 1
    if not counts:
        raise RuntimeError(f"no degenerate_output diagnoses found for {SLUG}")
    return dict(sorted(counts.items()))


def load_prompts(ids: set[str]) -> list[dict]:
    records = [r for r in read_jsonl(config.DATA_JSONL) if r["id"] in ids]
    missing = ids - {r["id"] for r in records}
    if missing:
        raise RuntimeError(f"prompt ids absent from {config.DATA_JSONL}: {sorted(missing)}")
    return records


# --------------------------------------------------------------------------
# Provider pin probe (§0.2 contingency).
# --------------------------------------------------------------------------

def probe_provider_pin(cfg: RunConfig) -> dict:
    """One 16-token pinned call. Returns {"resolved": bool, "provider": str|None,
    "error": str|None}. A routing failure (404 'no endpoints') means the
    bf16/fp16 pin is unservable today → arms run unpinned, reported loudly."""
    client, send_model = config.resolve_candidate_transport(
        cfg.candidates[CANDIDATE_KEY])
    try:
        resp = client.chat.completions.create(
            model=send_model,
            temperature=0.0,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            extra_body=cfg.extra_body,
            timeout=60.0,
            **({"seed": cfg.seed} if cfg.seed is not None else {}),
        )
        provider = getattr(resp, "provider", None) or (
            (getattr(resp, "model_extra", None) or {}).get("provider"))
        return {"resolved": True, "provider": provider, "error": None}
    except Exception as e:  # noqa: BLE001 — any routing failure disables the pin
        return {"resolved": False, "provider": None,
                "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# Generation (13 prompts × 2 temps, resumable, continue-on-error).
# --------------------------------------------------------------------------

def responses_path(arm: str) -> Path:
    return ABLATION_DIR / "responses" / arm / f"{CANDIDATE_KEY}.jsonl"


def rejected_path(arm: str) -> Path:
    return ABLATION_DIR / "rejected" / arm / f"{CANDIDATE_KEY}.jsonl"


def run_generation(base_cfg: RunConfig, records: list[dict], pinned: bool,
                   deadline_s: float, workers: int) -> list[str]:
    """Fan (arm × prompt) items over a small pool; per-item continue-on-error
    (errors are returned, printed, and left absent from the response file so a
    rerun retries them)."""
    arm_cfgs = {
        arm: dataclasses.replace(
            base_cfg, temperature=temp,
            provider_quantizations=base_cfg.provider_quantizations if pinned else None,
        )
        for arm, temp in ARMS.items()
    }
    items = []
    for arm, cfg in arm_cfgs.items():
        done = {r["id"] for r in read_jsonl(responses_path(arm))}
        items += [(arm, cfg, rec) for rec in records if rec["id"] not in done]

    errors: list[str] = []

    def _one(arm: str, cfg: RunConfig, rec: dict) -> None:
        def _capture_reject(fields: dict) -> None:
            append_jsonl(rejected_path(arm), {"id": rec["id"], **fields})

        try:
            fields = _generate_one(cfg, cfg.candidates[CANDIDATE_KEY],
                                   rec["prompt"], deadline_s=deadline_s,
                                   on_reject=_capture_reject)
        except Exception as e:  # noqa: BLE001 — post-retry failure; keep going
            errors.append(f"{arm} {rec['id']}: {type(e).__name__}: {e}")
            return
        append_jsonl(responses_path(arm),
                     {"id": rec["id"], "temperature": cfg.temperature, **fields})
        print(f"  {arm} {rec['id']}: finish={fields.get('finish_reason')} "
              f"provider={fields.get('provider')} chars={len(fields.get('response', ''))}")

    print(f"generate: {len(items)} calls pending "
          f"({len(records)} prompts × {len(ARMS)} arms, resume-aware)")
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="ablation") as pool:
        futures = [pool.submit(_one, arm, cfg, rec) for arm, cfg, rec in items]
        try:
            for f in as_completed(futures):
                f.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return errors


# --------------------------------------------------------------------------
# Decode-health census (pure computation; mirrors _decode_health tallies).
# --------------------------------------------------------------------------

def _row(rec: dict) -> dict:
    content = rec.get("response", "")
    reasoning = rec.get("reasoning", "")
    c_loop = detect_repetition_loop(content)
    r_loop = detect_repetition_loop(reasoning)
    finish = rec.get("finish_reason")
    return {
        "finish_reason": finish,
        "provider": rec.get("provider"),
        "content_chars": len(content),
        "content_loop": c_loop,
        "reasoning_loop": r_loop if reasoning else None,
        "loop_any": bool(c_loop["looping"] or r_loop["looping"]),
        "nonstop_finish": finish != "stop",
        "stored_loop_failure": rec.get("loop_failure"),
    }


def _tally(by_prompt: dict[str, dict]) -> dict:
    rows = list(by_prompt.values())
    n_loop = sum(1 for r in rows if r["loop_any"])
    return {
        "n_responses": len(rows),
        "n_loop_content": sum(1 for r in rows if r["content_loop"]["looping"]),
        "n_loop_reasoning": sum(
            1 for r in rows if (r["reasoning_loop"] or {}).get("looping")),
        "n_loop_any": n_loop,
        "n_loop_escaped": sum(
            1 for r in rows if r["loop_any"] and r["finish_reason"] == "stop"),
        "n_loop_died": sum(
            1 for r in rows if r["loop_any"] and r["finish_reason"] != "stop"),
        "n_nonstop_finish": sum(1 for r in rows if r["nonstop_finish"]),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "providers": dict(Counter(r["provider"] for r in rows)),
    }


def analyze(base_cfg: RunConfig, degen_counts: dict[str, int],
            pin_probe: dict | None) -> dict:
    ids = set(degen_counts)
    baseline_by_prompt = {
        r["id"]: _row(r)
        for r in read_jsonl(base_cfg.responses_path(CANDIDATE_KEY))
        if r["id"] in ids
    }
    arms = {}
    for arm, temp in ARMS.items():
        by_prompt = {r["id"]: _row(r) for r in read_jsonl(responses_path(arm))}
        arms[arm] = {
            "temperature": temp,
            "complete": len(by_prompt) == len(ids),
            "tallies": _tally(by_prompt),
            "rejected_attempts": len(read_jsonl(rejected_path(arm))),
            "by_prompt": by_prompt,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_run": SLUG,
        "detector": {"window": WINDOW, "ngram": NGRAM, "min_repeats": MIN_REPEATS},
        "verdict_note": ("any repetition loop counts as a failure (user ruling "
                         "2026-07-09); degenerate_output diagnoses are "
                         "judge-assigned and broader than mechanical loops — "
                         "the detector census is the like-for-like signal here"),
        "prompt_ids": sorted(ids),
        "degenerate_criteria_per_prompt": degen_counts,
        "n_degenerate_criteria": sum(degen_counts.values()),
        "provider_pin_probe": pin_probe,
        "baseline": {
            "temperature": base_cfg.temperature,
            "seed": base_cfg.seed,
            "max_tokens": base_cfg.max_tokens,
            "tallies": _tally(baseline_by_prompt),
            "by_prompt": baseline_by_prompt,
        },
        "arms": arms,
    }


# --------------------------------------------------------------------------
# Findings (auto-generated; numbers first, conditional reading second).
# --------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def write_findings(health: dict) -> None:
    ids = health["prompt_ids"]
    base = health["baseline"]
    arms = health["arms"]
    n = len(ids)
    lines = [
        "# E08 degenerate_output ablation — findings",
        "",
        f"_Auto-generated by `experiments/E08/ablation_degenerate.py` "
        f"at {health['generated_at']}. Synthesis Rec 2 (discrepancy_drill)._",
        "",
        f"**Setup.** The {n} prompts carrying all "
        f"{health['n_degenerate_criteria']} degenerate_output-diagnosed "
        f"criteria from {SLUG} were rerun at temperature "
        f"{arms['t02']['temperature']} and {arms['t06']['temperature']} with "
        f"finish_reason + provider logging; all other decode params match the "
        f"frozen census config (max_tokens={base['max_tokens']}, reasoning "
        f"off, seed={base['seed']}). The t06 arm replicates the census "
        f"temperature ({base['temperature']}), so it doubles as a "
        f"same-config/different-day (and possibly different-provider) "
        f"replication. Generation + decode telemetry only — no judge calls, "
        f"so 'degenerate' here is the mechanical decode-health signal "
        f"(repetition loops, non-stop finish_reason), not re-judged criteria.",
        "",
    ]
    probe = health.get("provider_pin_probe")
    if probe is None:
        lines += ["**Provider pin.** Probe skipped (analyze-only invocation).", ""]
    elif probe["resolved"]:
        lines += [f"**Provider pin.** bf16/fp16 pin resolved "
                  f"(probe served by {probe['provider']}).", ""]
    else:
        lines += [
            "**Provider pin.** The frozen bf16/fp16 quantization pin did NOT "
            "resolve at run time; arms ran on OpenRouter's best available "
            "route (per-response `provider` field records who served each "
            "call; serving quantization is not exposed for this model). "
            f"Probe error: `{probe['error']}`. The census was served by "
            "Parasail — a provider change is itself a confound to weigh "
            "when reading the deltas below.", ""]

    lines += ["## Decode-health tallies", "",
              "| metric | census (t=" + str(base["temperature"]) + ") | " +
              " | ".join(f"{a} (t={arms[a]['temperature']})" for a in ARMS) + " |",
              "|---|---|" + "---|" * len(ARMS)]
    for metric in ("n_loop_content", "n_loop_any", "n_loop_escaped",
                   "n_loop_died", "n_nonstop_finish"):
        row = [f"| {metric} | {_pct(base['tallies'][metric], n)} "]
        row += [f"| {_pct(arms[a]['tallies'][metric], arms[a]['tallies']['n_responses'])} "
                for a in ARMS]
        lines.append("".join(row) + "|")
    lines += [
        "",
        "| arm | finish_reasons | providers | rejected attempts |",
        "|---|---|---|---|",
        f"| census | {json.dumps(base['tallies']['finish_reasons'])} | "
        f"{json.dumps(base['tallies']['providers'])} | — |",
    ]
    for a in ARMS:
        t = arms[a]["tallies"]
        lines.append(f"| {a} | {json.dumps(t['finish_reasons'])} | "
                     f"{json.dumps(t['providers'])} | {arms[a]['rejected_attempts']} |")

    lines += ["", "## Per-prompt finish_reason / loop", "",
              "| prompt | degen criteria | census | " +
              " | ".join(ARMS) + " |",
              "|---|---|---|" + "---|" * len(ARMS)]

    def _cell(row: dict | None) -> str:
        if row is None:
            return "MISSING"
        loop = "LOOP" if row["loop_any"] else "clean"
        return f"{row['finish_reason']}/{loop}/{row['content_chars']}ch"

    for pid in ids:
        cells = [_cell(arms[a]["by_prompt"].get(pid)) for a in ARMS]
        lines.append(
            f"| {pid} | {health['degenerate_criteria_per_prompt'][pid]} | "
            f"{_cell(base['by_prompt'].get(pid))} | " + " | ".join(cells) + " |")

    # Conditional reading — computed, not editorialized.
    complete = all(arms[a]["complete"] for a in ARMS)
    lines += ["", "## Reading", ""]
    if not complete:
        lines += ["Arms are INCOMPLETE — rerun the script to fill gaps before "
                  "reading anything into the numbers.", ""]
    else:
        b_len = base["tallies"]["n_nonstop_finish"]
        deltas = []
        for a in ARMS:
            t = arms[a]["tallies"]
            deltas.append(
                f"- At t={arms[a]['temperature']}: length-capped "
                f"{_pct(t['n_nonstop_finish'], n)} vs census {_pct(b_len, n)}; "
                f"content loops {_pct(t['n_loop_content'], n)} vs census "
                f"{_pct(base['tallies']['n_loop_content'], n)}.")
        lines += deltas + [""]
        t02, t06 = arms["t02"]["tallies"], arms["t06"]["tallies"]
        if t02["n_loop_any"] < t06["n_loop_any"]:
            lines.append(
                "Loops thin out at the lower temperature → consistent with a "
                "decoding artifact (sampling-entropy-sensitive), not a stable "
                "model behavior; a repetition-penalty / decode-config fix is "
                "plausible before concluding anything about the model.")
        elif t02["n_loop_any"] > t06["n_loop_any"]:
            lines.append(
                "Loops are MORE frequent at t=0.2 → the classic "
                "low-temperature repetition failure of small models; the "
                "census t=0.6 setting was not the cause, and the "
                "degenerate_output mass looks model-intrinsic at this size.")
        else:
            lines.append(
                "Loop counts are flat across temperatures → temperature is "
                "not the lever; if per-prompt overlap with the census is also "
                "high, degenerate output looks prompt-driven and "
                "model-intrinsic rather than a decoding artifact.")
        lines += [
            "",
            "Caveats: single sample per (prompt, temp) — treat rate deltas "
            "smaller than ~2 prompts as noise; provider route may differ from "
            "the census (see pin note); degenerate_output diagnoses include "
            "judge-visible degeneracy (e.g. mid-list gibberish) that the "
            "mechanical loop detector does not flag, so census loop counts "
            "being lower than 13 is expected, not a contradiction.",
        ]
    (ABLATION_DIR / "findings.md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip generation (and the pin probe); recompute "
                         "decode_health.json + findings.md from stored files.")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deadline-s", type=float, default=DEFAULT_DEADLINE_S)
    args = ap.parse_args(argv)

    base_cfg = load_frozen_cfg()
    degen_counts = degenerate_prompt_ids(base_cfg)
    print(f"{len(degen_counts)} degenerate_output prompts / "
          f"{sum(degen_counts.values())} criteria: {', '.join(degen_counts)}")

    pin_probe = None
    errors: list[str] = []
    if not args.analyze_only:
        pin_probe = probe_provider_pin(base_cfg)
        if pin_probe["resolved"]:
            print(f"provider pin bf16/fp16 OK (probe: {pin_probe['provider']})")
        else:
            print(f"⚠ provider pin bf16/fp16 did NOT resolve — running "
                  f"unpinned on best available route.\n  {pin_probe['error']}",
                  file=sys.stderr)
        records = load_prompts(set(degen_counts))
        errors = run_generation(base_cfg, records, pin_probe["resolved"],
                                args.deadline_s, args.workers)
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001 — manifest metadata only
            commit = None
        manifest = {
            "ablation": "E08 degenerate_output temp ablation (synthesis Rec 2)",
            "source_run": SLUG,
            "run_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "prompt_ids": sorted(degen_counts),
            "arms": {a: {"temperature": t} for a, t in ARMS.items()},
            "frozen_params_held": {
                "max_tokens": base_cfg.max_tokens,
                "reasoning": base_cfg.reasoning,
                "seed": base_cfg.seed,
                "timeout_s": base_cfg.timeout_s,
                "model": base_cfg.candidates[CANDIDATE_KEY],
            },
            "provider_pin": {
                "requested": list(base_cfg.provider_quantizations or ()),
                **pin_probe,
            },
            "deadline_s": args.deadline_s,
            "errors": errors,
        }
        ABLATION_DIR.mkdir(parents=True, exist_ok=True)
        (ABLATION_DIR / "ablation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

    health = analyze(base_cfg, degen_counts, pin_probe)
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    (ABLATION_DIR / "decode_health.json").write_text(
        json.dumps(health, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_findings(health)
    print(f"wrote {ABLATION_DIR / 'decode_health.json'}")
    print(f"wrote {ABLATION_DIR / 'findings.md'}")
    for e in errors:
        print(f"ERROR (rerun to retry): {e}", file=sys.stderr)
    for arm in ARMS:
        got = len(read_jsonl(responses_path(arm)))
        print(f"{arm}: {got}/{len(degen_counts)} responses stored")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
