"""T3 — Tier-3 general-capability regression guard (PREREG §4, H3).

H3 is a REGRESSION GUARD, not a hypothesis to confirm: no *trained* arm may score
more than 3 points below Arm 0 on a general-capability battery. PREREG §4 defines
the guard but never froze a battery; the composition below was chosen at H3-launch
(2026-07-20) and operator-approved before any arm was scored.

Battery (operator-approved 2026-07-20):
    MMLU, 0-shot, deterministic answer-letter log-likelihood scoring.
    Fixed stratified subset of N=1000 questions across all 57 subjects, selection
    seed 20260715 (project seed). Same subset for every arm (paired items).

Scoring is the Hendrycks / lm-eval-harness MMLU protocol adapted to a chat model:
format the question + A/B/C/D as a user turn, run ONE forward pass, and compare the
model's next-token log-probability of the four bare letter tokens (A,B,C,D) at the
first assistant position (argmax = prediction). No sampling, no generation, so the
score is fully deterministic — the arm-vs-base delta is a clean paired difference,
not a temp-0.6 decode like T1.4. This reads the immediate answer distribution, so a
LoRA arm whose *formatting* drifted is not penalised for anything but lost knowledge.

Arms (one per invocation, like tier1_decode):
    0   base model, untrained                      (reference — the H3 denominator)
    P   base + enumerate-then-verify system prompt (reference, $0; NOT guard-gated)
    SA SB RA RB   base + final LoRA adapter (step-300 top-level weights)  [guard-gated]

Arm P is scored WITH its scaffold system prompt (that IS Arm P's identity). The
scaffold is meaningless for knowledge MC and depresses the first-token letter mass,
so P's MMLU number is a reference for the SB-vs-P discussion, not a capability claim
— and P is not a trained arm, so the >3pt guard does not apply to it (PREREG §4).

Resumable: existing qids in the output JSONL are skipped. The fixed subset is built
once to results/eval_t3/mmlu_subset.jsonl and reused, so every arm scores identical
items even if the upstream dataset changes.

Run (system python == frozen training env):
    python3 experiments/T01/eval/tier3_mmlu.py --arm 0
    python3 experiments/T01/eval/tier3_mmlu.py --arm 0 --limit 4   # smoke
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

_HERE = pathlib.Path(__file__).resolve()
_T01 = _HERE.parents[1]
_REPO = _T01.parents[1]
sys.path.insert(0, str(_T01 / "training"))   # common (MODEL_PATH, loaders)

from common import load_model, load_tokenizer  # noqa: E402

ARMS = ("0", "P", "SA", "SB", "RA", "RB")
TRAINED_ARMS = ("SA", "SB", "RA", "RB")        # the arms H3 actually gates
LETTERS = "ABCD"
SUBSET_SEED = 20260715                          # PREREG project seed (subset selection only)
SUBSET_N = 1000
ADAPTER_DIR = _REPO / "results" / "adapters"
OUT_DIR = _REPO / "results" / "eval_t3"
SUBSET_PATH = OUT_DIR / "mmlu_subset.jsonl"

# Arm P's system prompt — byte-identical to eval/tier1_decode.py's SYSTEM_PROMPT_P
# (imported below so the two cannot drift). Arm P = base + this scaffold.
from tier1_decode import SYSTEM_PROMPT_P  # noqa: E402


def build_subset() -> list[dict]:
    """Fixed stratified N=1000 MMLU subset across all 57 subjects, seed 20260715.

    Deterministic: per-subject quota is fixed (17 or 18), and within each subject
    the questions are drawn after a seeded shuffle. qids encode the ORIGINAL
    per-subject index, so they are stable regardless of shuffle order.
    """
    from datasets import get_dataset_config_names, load_dataset

    subjects = [s for s in sorted(get_dataset_config_names("cais/mmlu"))
                if s not in ("all", "auxiliary_train")]
    assert len(subjects) == 57, f"expected 57 MMLU subjects, got {len(subjects)}"

    n_sub = len(subjects)
    base = SUBSET_N // n_sub                      # 17
    rem = SUBSET_N - base * n_sub                 # 31 -> first 31 subjects get +1
    quotas = {s: base + (1 if i < rem else 0) for i, s in enumerate(subjects)}

    rng = random.Random(SUBSET_SEED)
    rows: list[dict] = []
    for s in subjects:
        ds = load_dataset("cais/mmlu", s, split="test")
        items = [{"qid": f"{s}::{oi}", "subject": s, "question": r["question"],
                  "choices": list(r["choices"]), "gold": int(r["answer"])}
                 for oi, r in enumerate(ds)]
        rng.shuffle(items)
        rows.extend(items[: quotas[s]])
    assert len(rows) == SUBSET_N, f"subset size {len(rows)} != {SUBSET_N}"
    return rows


def load_or_build_subset() -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if SUBSET_PATH.exists():
        with SUBSET_PATH.open(encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        print(f"[subset] loaded {len(rows)} questions from {SUBSET_PATH.name}")
        return rows
    rows = build_subset()
    with SUBSET_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[subset] built {len(rows)} questions (seed {SUBSET_SEED}) -> {SUBSET_PATH.name}")
    return rows


def format_question(r: dict) -> str:
    subject = r["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question (with answer) about {subject}.",
             "", r["question"].strip()]
    for letter, choice in zip(LETTERS, r["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer with the letter of the correct choice.")
    return "\n".join(lines)


def messages_for(arm: str, question_block: str) -> list[dict]:
    msgs = []
    if arm == "P":
        msgs.append({"role": "system", "content": SYSTEM_PROMPT_P})
    msgs.append({"role": "user", "content": question_block})
    return msgs


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--limit", type=int, default=None, help="cap total questions (smoke)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    import torch

    rows = load_or_build_subset()
    rows.sort(key=lambda r: r["qid"])            # fixed order -> reproducible batching
    if args.limit:
        rows = rows[: args.limit]

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arm}.jsonl"
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["qid"])
        print(f"[resume] {len(done)} questions already scored in {out_path.name}")

    tokenizer = load_tokenizer()
    tokenizer.padding_side = "left"              # last position == real final token for every row
    # Bare letter token ids (probe-verified: A/B/C/D -> 32/33/34/35 as the first
    # assistant content token; asserted single-token so scoring can't silently break).
    letter_ids = []
    for L in LETTERS:
        ids = tokenizer.encode(L, add_special_tokens=False)
        assert len(ids) == 1, f"letter {L!r} is not a single token: {ids}"
        letter_ids.append(ids[0])
    letter_ids_t = torch.tensor(letter_ids, device="cuda")

    model = build_model(args.arm)

    todo = [r for r in rows if r["qid"] not in done]
    n_new = 0
    t0 = time.time()
    with out_path.open("a", encoding="utf-8") as out_f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            texts = [
                tokenizer.apply_chat_template(
                    messages_for(args.arm, format_question(r)),
                    tokenize=False, add_generation_prompt=True)
                for r in batch
            ]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            add_special_tokens=False).to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :]        # [B, V] final-position logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            letter_lp = logprobs[:, letter_ids_t]             # [B, 4] true logprobs
            preds = letter_lp.argmax(dim=-1)                  # [B]
            for r, lp, pi in zip(batch, letter_lp.tolist(), preds.tolist()):
                out_f.write(json.dumps({
                    "arm": args.arm,
                    "qid": r["qid"],
                    "subject": r["subject"],
                    "gold": r["gold"],
                    "pred": int(pi),
                    "correct": bool(int(pi) == r["gold"]),
                    "logprobs": {L: round(v, 4) for L, v in zip(LETTERS, lp)},
                }) + "\n")
                n_new += 1
            out_f.flush()
            el = time.time() - t0
            print(f"[arm {args.arm}] {len(done) + n_new}/{len(rows)} "
                  f"({n_new} new, {el:.0f}s, {n_new / max(el, 1e-9):.1f} q/s)", flush=True)

    # Recompute the summary over the FULL output file (resume-safe).
    scored = []
    with out_path.open(encoding="utf-8") as f:
        scored = [json.loads(l) for l in f if l.strip()]
    n = len(scored)
    n_correct = sum(1 for d in scored if d["correct"])
    by_subject: dict[str, list[int]] = {}
    for d in scored:
        by_subject.setdefault(d["subject"], []).append(1 if d["correct"] else 0)
    subj_acc = {s: {"n": len(v), "correct": sum(v), "acc": round(sum(v) / len(v), 4)}
                for s, v in sorted(by_subject.items())}
    summary = {
        "arm": args.arm, "battery": "mmlu-0shot-letterloglik",
        "n": n, "n_correct": n_correct, "accuracy": round(n_correct / n, 4) if n else None,
        "subset_seed": SUBSET_SEED, "subset_n": SUBSET_N, "deterministic": True,
        "n_new_this_run": n_new, "elapsed_s": round(time.time() - t0, 1),
        "per_subject": subj_acc,
    }
    (out_dir / f"{args.arm}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] arm {args.arm}: accuracy {summary['accuracy']} "
          f"({n_correct}/{n})  [{n_new} new this run]")


if __name__ == "__main__":
    main()
