"""Fast local recalibration loop: compose FRESH coverage prompts with the current
compose.py, run the LOCAL base model (k rollouts, temp 0.9), report mean
per-criterion pass + per-type breakdown vs the 30-70% band. No file writes, no
API — the iteration loop for hardening the coverage generator (edit compose.py,
re-run this) before committing to a full regenerate + teacher spend.

  python calibrate_local.py --n 60 --k 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifiers"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datagen"))

import torch  # noqa: E402

import reward  # noqa: F401,E402 — registers verifier pools
from base import check  # noqa: E402
from common import extract_final, load_model, load_tokenizer  # noqa: E402
from compose import compose_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--cause", default="coverage")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    recs = [compose_prompt(args.cause, rng, f"CAL-{i:03d}") for i in range(args.n)]

    tok = load_tokenizer(); tok.padding_side = "left"
    model = load_model().to("cuda").eval()

    per_type = defaultdict(lambda: [0, 0])
    fracs, ks = [], []
    for r in recs:
        ks.append(len(r["specs"]))
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
            fracs.append(passed / len(r["specs"]))

    mean = sum(fracs) / len(fracs)
    print(f"\n[calibrate_local] cause={args.cause} n={args.n} k_rollouts={args.k} "
          f"mean_constraints/prompt={sum(ks)/len(ks):.1f}")
    print("per-type base pass:")
    for t, (p, tot) in sorted(per_type.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        rate = p / tot
        flag = "  <=EASY" if rate > 0.75 else ("  <=hard" if rate < 0.45 else "")
        print(f"  {t:20s} {rate:5.1%} (n={tot}){flag}")
    band = "IN BAND ✓" if 0.30 <= mean <= 0.70 else ("TOO EASY" if mean > 0.70 else "TOO HARD")
    print(f"\n  MEAN per-criterion pass = {mean:.1%}  ->  {band}  (target 30-70%)")


if __name__ == "__main__":
    main()
