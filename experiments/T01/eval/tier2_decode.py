"""Tier-2 CC-75 local decode — HF generate() responses for one arm on the CC-75
corpus. Mirrors eval/tier1_decode.py (same local stack the T1.4 eval validated).

Regime matches E08's base decode where it must (temp 0.6, max_new_tokens 1536,
reasoning off = bare chat, seed 20260715) and T1.4's majority-vote where it helps
(k=3 by default, one seed per decode index). Arm 0 is re-decoded LOCALLY here for
§10-#7 stack parity (E08's Arm 0 ran on OpenRouter).

    0            base model, untrained
    SA SB RA RB  base + final LoRA adapter (step-300 top-level weights)

Formatting: bare user message through the chat template, no system message —
identical to training / T1.4 (Arm P is out of Tier-2 scope). Resumable:
existing (prompt_id, decode_idx) pairs are skipped.

Run:
    python3 experiments/T01/eval/tier2_decode.py --arm 0 --subset          # spot-check
    python3 experiments/T01/eval/tier2_decode.py --arm SA                  # full (all 75)
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
sys.path.insert(0, str(_T01 / "training"))
sys.path.insert(0, str(_HERE.parent))          # tier2_labels

from common import load_model, load_tokenizer          # noqa: E402
from tier2_labels import load_corpus, select_subset     # noqa: E402

ARMS = ("0", "SA", "SB", "RA", "RB")
DECODE_SEEDS = (20260715, 20260716, 20260717)          # one per decode index (T1.4 convention)
EVAL_TEMPERATURE = 0.6                                  # E08 base regime
EVAL_TOP_P = 1.0
MAX_NEW_TOKENS = 1536                                   # E08 base regime (not T1.4's 2048)
ADAPTER_DIR = _REPO / "results" / "adapters"
OUT_DIR = _REPO / "results" / "eval_t2" / "responses"


def build_model(arm: str):
    model = load_model()
    if arm != "0":
        from peft import PeftModel
        path = ADAPTER_DIR / f"T01-{arm}"
        model = PeftModel.from_pretrained(model, str(path))
        print(f"[arm {arm}] LoRA adapter loaded from {path}")
    else:
        print(f"[arm {arm}] base model (no adapter)")
    model.eval().cuda()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--subset", action="store_true", help="spot-check subset (18 prompts, seed 20260715)")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts by id (smoke)")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch

    corpus = load_corpus()
    ids = sorted(corpus)
    if args.subset:
        ids = select_subset()
    if args.limit:
        ids = ids[: args.limit]
    rows = [corpus[i] for i in ids]

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.arm + ("_subset" if args.subset else "")
    out_path = out_dir / f"{tag}.jsonl"
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
            torch.manual_seed(seed)
            for i in range(0, len(todo), args.batch_size):
                batch = todo[i : i + args.batch_size]
                texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": r["prompt"]}],
                        tokenize=False, add_generation_prompt=True)
                    for r in batch
                ]
                enc = tokenizer(texts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(
                        **enc, do_sample=True, temperature=EVAL_TEMPERATURE,
                        top_p=EVAL_TOP_P, max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.pad_token_id)
                new_tokens = gen[:, enc["input_ids"].shape[1]:]
                for r, toks in zip(batch, new_tokens):
                    text = tokenizer.decode(toks, skip_special_tokens=True)
                    n_gen = int((toks != tokenizer.pad_token_id).sum())
                    out_f.write(json.dumps({
                        "arm": args.arm, "prompt_id": r["id"], "decode_idx": decode_idx,
                        "seed": seed, "temperature": EVAL_TEMPERATURE, "top_p": EVAL_TOP_P,
                        "max_new_tokens": MAX_NEW_TOKENS, "gen_tokens": n_gen,
                        "len_capped": bool(n_gen >= MAX_NEW_TOKENS), "response": text,
                    }) + "\n")
                    n_new += 1
                out_f.flush()
                el = time.time() - t0
                print(f"[arm {args.arm}{'/subset' if args.subset else ''}] decode {decode_idx} | "
                      f"{len(done) + n_new}/{total} ({n_new} new, {el:.0f}s)", flush=True)

    print(f"[done] arm {args.arm}: {len(done) + n_new} (prompt,decode) responses in {out_path.name}")


if __name__ == "__main__":
    main()
