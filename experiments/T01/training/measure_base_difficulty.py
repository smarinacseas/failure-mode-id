"""Base-model difficulty probe (reviewer items 3 + 5). Loads the UNTRAINED subject
and, per cause, generates k rollouts (temp 0.9, the GRPO rollout temp) on a sample
of train prompts, then reports the reward decomposition:
  * verifier fraction   = mean constraint pass-rate on extract_final(rollout)
  * malformed rate      = fraction of empty rollouts (format penalty trigger)
  * overlength rate     = fraction over the M-char cap (length penalty trigger)
against the PREREG §5 30-70% base-difficulty band, split by whether the prompt is
an ACCEPTED (in data/sft) or REJECTED training prompt (survivor-composition covariate).

Run AFTER the sweep (needs the GPU):
  python measure_base_difficulty.py --n 40 --k 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                 # training/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verifiers"))  # base, reward, pools

import torch

import reward  # noqa: F401,E402: importing registers the verifier pools
from base import check  # noqa: E402
from common import CAUSES, MODEL_PATH, extract_final, load_model, load_tokenizer
from datasets_t01 import DATA_DIR

MAX_CHARS = 2800


def accepted_id_set(cause: str) -> set[str]:
    return {json.loads(l)["id"] for l in (DATA_DIR / "sft" / f"{cause}.jsonl").open()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="train prompts sampled per cause")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=20260715)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = load_tokenizer(); tok.padding_side = "left"
    model = load_model().to("cuda").eval()
    out = {}
    for cause in CAUSES:
        acc = accepted_id_set(cause)
        rows = [json.loads(l) for l in (DATA_DIR / "train" / f"{cause}.jsonl").open()][:args.n]
        recs = []
        for r in rows:
            msgs = [{"role": "user", "content": r["prompt"]}]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to("cuda")
            plen = enc["input_ids"].shape[1]
            fracs, malformed, over = [], 0, 0
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=True,
                                     temperature=args.temp, num_return_sequences=args.k,
                                     pad_token_id=tok.eos_token_id)
            for g in gen:
                txt = tok.decode(g[plen:], skip_special_tokens=True)
                ans = extract_final(txt)
                if not ans.strip():
                    malformed += 1
                if len(ans) > MAX_CHARS:
                    over += 1
                fracs.append(sum(check(ans, s).passed for s in r["specs"]) / len(r["specs"]))
            recs.append({"id": r["id"], "accepted": r["id"] in acc,
                         "verifier_frac": sum(fracs) / len(fracs),
                         "malformed": malformed / args.k, "over": over / args.k})
        def agg(sub):
            if not sub:
                return None
            return {"n": len(sub),
                    "verifier_frac": round(sum(x["verifier_frac"] for x in sub) / len(sub), 3),
                    "malformed_rate": round(sum(x["malformed"] for x in sub) / len(sub), 3),
                    "overlength_rate": round(sum(x["over"] for x in sub) / len(sub), 3)}
        out[cause] = {"all": agg(recs),
                      "accepted": agg([x for x in recs if x["accepted"]]),
                      "rejected": agg([x for x in recs if not x["accepted"]])}
        a = out[cause]["all"]
        band = "IN 30-70% band" if 0.30 <= a["verifier_frac"] <= 0.70 else "OUT OF BAND"
        flag = "  <<< FLAG: verifier fraction > 0.70" if a["verifier_frac"] > 0.70 else ""
        print(f"[{cause}] base verifier_frac={a['verifier_frac']} malformed={a['malformed_rate']} "
              f"over={a['overlength_rate']} -> {band}{flag}")
        print(f"    accepted-prompts: {out[cause]['accepted']}")
        print(f"    rejected-prompts: {out[cause]['rejected']}")
    Path("/workspace/failure-mode-id/results/probe/base_difficulty.json").write_text(
        json.dumps({"model": MODEL_PATH, "k": args.k, "temp": args.temp, "by_cause": out}, indent=2))
    print("[saved] results/probe/base_difficulty.json")


if __name__ == "__main__":
    main()
