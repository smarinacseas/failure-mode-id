"""Per-constraint-TYPE base-pass rate on the coverage pool (recalibration input).
Identifies which coverage constraint types the base 3B aces, so hardening targets
them precisely instead of guessing. GPU; run after the sweep.

  python measure_coverage_by_type.py --n 60 --k 4 [--pool train]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifiers"))

import torch  # noqa: E402

import reward  # noqa: F401,E402 — registers verifier pools
from base import check  # noqa: E402
from common import MODEL_PATH, extract_final, load_model, load_tokenizer  # noqa: E402
from datasets_t01 import DATA_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--pool", default="train")
    ap.add_argument("--seed", type=int, default=20260715)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = load_tokenizer(); tok.padding_side = "left"
    model = load_model().to("cuda").eval()
    rows = [json.loads(l) for l in (DATA_DIR / args.pool / "coverage.jsonl").open()][:args.n]

    per_type = defaultdict(lambda: [0, 0])   # type -> [passed, total]
    prompt_fracs = []
    for r in rows:
        enc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to("cuda")
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=args.temp, num_return_sequences=args.k,
                                 pad_token_id=tok.eos_token_id)
        for g in gen:
            ans = extract_final(tok.decode(g[plen:], skip_special_tokens=True))
            passed = 0
            for s in r["specs"]:
                ok = check(ans, s).passed
                per_type[s["type"]][0] += int(ok); per_type[s["type"]][1] += 1
                passed += int(ok)
            prompt_fracs.append(passed / len(r["specs"]))

    print(f"\ncoverage base per-TYPE pass rate ({len(rows)} prompts x k={args.k}, temp {args.temp}):")
    out = {}
    for t, (p, tot) in sorted(per_type.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        rate = p / tot
        out[t] = {"pass_rate": round(rate, 3), "n": tot}
        flag = "  <= EASY (harden)" if rate > 0.75 else ("  <= hard" if rate < 0.45 else "")
        print(f"  {t:20s} {rate:5.1%}  (n={tot}){flag}")
    mean = sum(prompt_fracs) / len(prompt_fracs)
    print(f"\n  MEAN per-criterion pass = {mean:.1%}  (target band 30-70%)")
    Path("/workspace/failure-mode-id/results/probe/coverage_by_type.json").write_text(
        json.dumps({"model": MODEL_PATH, "mean_frac": mean, "by_type": out}, indent=2))
    print("[saved] results/probe/coverage_by_type.json")


if __name__ == "__main__":
    main()
