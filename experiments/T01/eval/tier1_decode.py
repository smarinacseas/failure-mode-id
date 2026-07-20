"""T1.4 Tier-1 holdout decode runner (PREREG §5).

Decodes the 400-prompt holdout (200/cause) with k=3 samples per prompt at the
frozen eval regime — temperature 0.6, top_p 1.0, seeds 20260715/16/17 (one per
decode index, PREREG §7) — for one arm per invocation:

    0   base model, untrained                      (reference)
    P   base + enumerate-then-verify system prompt (reference, $0)
    SA SB RA RB   base + final LoRA adapter (step-300 top-level weights)

Formatting matches training exactly (datasets_t01): a bare user message through
the chat template, no system message — Arm P is the single exception, adding
SYSTEM_PROMPT_P. Decode is local HF generate() (same stack training validated;
PREREG's "via vLLM" was explicitly deferred to T1.4 and declined — no new
serving stack for a 3B model / 7,200 decodes).

max_new_tokens=2048 matches the T1.2 difficulty-calibration regime that set the
30-70% band (datagen/calibrate.py), and exceeds training's 1536 completion
budget so no arm is clipped relative to its training regime.

Sampling params are passed explicitly to generate() so the model's bundled
generation_config defaults (Llama-3.2 ships temp 0.6 / top_p 0.9) cannot
silently shape the eval regime.

Resumable: existing (prompt_id, decode_idx) pairs in the output JSONL are
skipped, so a killed run continues where it left off.

Run (system python == frozen training env):
    python3 experiments/T01/eval/tier1_decode.py --arm RA
    python3 experiments/T01/eval/tier1_decode.py --arm 0 --limit 5 --k 1   # smoke
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve()
_T01 = _HERE.parents[1]
_REPO = _T01.parents[1]
sys.path.insert(0, str(_T01 / "training"))   # common (MODEL_PATH, loaders)

from common import load_model, load_tokenizer  # noqa: E402

ARMS = ("0", "P", "SA", "SB", "RA", "RB")
CAUSES = ("coverage", "precision")
DECODE_SEEDS = (20260715, 20260716, 20260717)   # PREREG §7, one per decode index
EVAL_TEMPERATURE = 0.6                          # PREREG §5
EVAL_TOP_P = 1.0                                # matches T1.2 calibration (API default)
MAX_NEW_TOKENS = 2048                           # matches T1.2 calibration regime
ADAPTER_DIR = _REPO / "results" / "adapters"
OUT_DIR = _REPO / "results" / "eval_t1_4" / "decodes"

# Arm P (PREREG §3: "base + enumerate-then-verify system prompt"). PREREG names
# the arm but never froze the text; defined here at T1.4, before any T1.4 grading
# was run. The ===FINAL=== convention mirrors the trained scaffold so only the
# final answer is graded (extract_final) — otherwise the visible checklist would
# trivially satisfy mention-type constraints and inflate this reference arm.
SYSTEM_PROMPT_P = (
    "You are a careful assistant. Before answering, work through the task in "
    "three explicit stages: (1) ENUMERATE — list every requirement and "
    "constraint in the task as a numbered checklist; (2) DRAFT — write a "
    "response that attempts to satisfy all of them; (3) VERIFY — go through "
    "the checklist item by item against the draft and fix any violation. "
    "Then write ===FINAL=== on its own line, followed by the final response "
    "only (no checklist, no commentary)."
)


def load_holdout(pools: list[str]) -> list[dict]:
    rows: list[dict] = []
    for cause in pools:
        path = _T01 / "data" / "holdout" / f"{cause}.jsonl"
        with path.open(encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def build_model(arm: str):
    model = load_model()
    if arm in ("SA", "SB", "RA", "RB"):
        from peft import PeftModel
        path = ADAPTER_DIR / f"T01-{arm}"
        model = PeftModel.from_pretrained(model, str(path))
        print(f"[arm {arm}] LoRA adapter loaded from {path}")
    else:
        print(f"[arm {arm}] base model (no adapter)")
    model.eval().cuda()
    return model


def messages_for(arm: str, prompt: str) -> list[dict]:
    msgs = []
    if arm == "P":
        msgs.append({"role": "system", "content": SYSTEM_PROMPT_P})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--pool", default="both", choices=("both",) + CAUSES)
    ap.add_argument("--k", type=int, default=3, help="decodes per prompt (<= len(DECODE_SEEDS))")
    ap.add_argument("--limit", type=int, default=None, help="per-pool prompt cap (smoke)")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch

    pools = list(CAUSES) if args.pool == "both" else [args.pool]
    rows = load_holdout(pools)
    if args.limit:
        by_cause: dict[str, list[dict]] = {c: [] for c in pools}
        for r in rows:
            by_cause[r["cause"]].append(r)
        rows = [r for c in pools for r in by_cause[c][: args.limit]]
    rows.sort(key=lambda r: r["id"])   # fixed order → reproducible batch composition

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}.jsonl"
    done: set[tuple[str, int]] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    done.add((d["prompt_id"], d["decode_idx"]))
        print(f"[resume] {len(done)} (prompt, decode) pairs already in {out_path.name}")

    tokenizer = load_tokenizer()
    tokenizer.padding_side = "left"
    model = build_model(args.arm)

    total = len(rows) * args.k
    n_new = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as out_f:
        for decode_idx in range(args.k):
            seed = DECODE_SEEDS[decode_idx]
            todo = [r for r in rows if (r["id"], decode_idx) not in done]
            if not todo:
                print(f"[decode {decode_idx}] all {len(rows)} prompts already done")
                continue
            torch.manual_seed(seed)   # one seed per decode index (PREREG §7)
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i : i + args.batch_size]
                texts = [
                    tokenizer.apply_chat_template(
                        messages_for(args.arm, r["prompt"]),
                        tokenize=False, add_generation_prompt=True)
                    for r in batch
                ]
                enc = tokenizer(texts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=EVAL_TEMPERATURE,
                        top_p=EVAL_TOP_P,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                new_tokens = gen[:, enc["input_ids"].shape[1]:]
                for r, toks in zip(batch, new_tokens):
                    text = tokenizer.decode(toks, skip_special_tokens=True)
                    n_gen = int((toks != tokenizer.pad_token_id).sum())
                    out_f.write(json.dumps({
                        "arm": args.arm,
                        "prompt_id": r["id"],
                        "cause": r["cause"],
                        "decode_idx": decode_idx,
                        "seed": seed,
                        "temperature": EVAL_TEMPERATURE,
                        "top_p": EVAL_TOP_P,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "gen_tokens": n_gen,
                        "len_capped": bool(n_gen >= MAX_NEW_TOKENS),
                        "completion": text,
                    }) + "\n")
                    n_new += 1
                out_f.flush()
                el = time.time() - t0
                done_total = len(done) + n_new
                print(f"[arm {args.arm}] decode {decode_idx} | {done_total}/{total} "
                      f"({n_new} new, {el:.0f}s, {n_new / el:.2f} dec/s)", flush=True)

    summary = {
        "arm": args.arm, "pools": pools, "k": args.k, "n_prompts": len(rows),
        "n_decodes_total": total, "n_decodes_new": n_new,
        "seeds": DECODE_SEEDS[: args.k], "temperature": EVAL_TEMPERATURE,
        "top_p": EVAL_TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / f"{args.arm}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
