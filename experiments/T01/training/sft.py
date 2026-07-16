"""SFT runner — Arms SA (coverage) and SB (precision). PREREG §5 + amendment
2026-07-16 (c).

Completion-only LoRA SFT: the user prompt is masked, the loss is on the teacher
scaffold + ===FINAL=== + answer, so the student learns the enumerate/compute-
then-verify behaviour from the bare task. Identical LoRA for every arm; only the
data differs (data/sft/{coverage,precision}.jsonl).

Run (probe / full):
  python sft.py --arm SA --cause coverage --limit 50 --epochs 1 --sanity
  python sft.py --arm SA --cause coverage --epochs 2
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from trl import SFTConfig, SFTTrainer

from callbacks import CSVLogger
from common import CAUSES, MODEL_PATH, load_model, load_tokenizer, lora_config
from datasets_t01 import DATA_DIR as T01_DATA, load_sft_dataset

WORKSPACE = Path("/workspace/failure-mode-id/results")

# Frozen SFT knobs (PREREG amendment (c)); execution defaults, estimand-immune.
LR = 1e-4
LR_SCHED = "cosine"
WARMUP_RATIO = 0.03
PER_DEVICE_BS = 4
GRAD_ACCUM = 4          # effective batch 16
MAX_LENGTH = 4096       # bump to 8192 only if probe truncation > 1%
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 3


def report_token_lengths(ds, tokenizer, max_length: int, out_path: Path | None = None) -> dict:
    """Token-length distribution of the full (prompt+completion) sequences and
    the truncation rate at `max_length` (PREREG amendment (c): flag if > 1%)."""
    lens = []
    for row in ds:
        ids = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"], tokenize=True, return_dict=False)
        lens.append(len(ids))
    lens.sort()
    n = len(lens)
    trunc = sum(x > max_length for x in lens)
    stats = {
        "n": n, "min": lens[0], "median": statistics.median(lens),
        "p95": lens[int(0.95 * n)] if n else 0, "max": lens[-1],
        "max_length": max_length, "n_truncated": trunc,
        "truncation_rate": round(trunc / n, 4) if n else 0.0,
    }
    print(f"[token-lengths] n={n} min={stats['min']} med={stats['median']:.0f} "
          f"p95={stats['p95']} max={stats['max']} | >{max_length}: {trunc} "
          f"({stats['truncation_rate']*100:.2f}%)")
    if stats["truncation_rate"] > 0.01:
        print(f"[FLAG] truncation {stats['truncation_rate']*100:.2f}% > 1% — bump max_length to 8192")
    if out_path:
        out_path.write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["SA", "SB"])
    ap.add_argument("--cause", required=True, choices=list(CAUSES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH)
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--sanity", action="store_true",
                    help="probe: report lengths, run, but write logs under results/probe/")
    ap.add_argument("--no-parity", action="store_true",
                    help="skip the SFT parity down-sample manifest (probes only)")
    args = ap.parse_args()

    out_dir = args.output_dir or str(WORKSPACE / "adapters" / f"T01-{args.arm}")
    log_dir = WORKSPACE / ("probe" if args.sanity else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sft_{args.arm}_{args.cause}{('_' + args.run_tag) if args.run_tag else ''}"
    csv_path = str(log_dir / f"{tag}.csv")

    print(f"== SFT {args.arm} ({args.cause}) | model={MODEL_PATH} | lr={args.lr} "
          f"| epochs={args.epochs} | out={out_dir} ==")

    # Parity down-sample (PREREG amendment DRAFT 2026-07-16): the real arms train
    # on the manifest's target_n ids so SA and SB have equal n. --no-parity for probes.
    only_ids = None
    if not args.no_parity:
        manifest_path = T01_DATA / "sft_manifests" / f"{args.arm}.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            only_ids = m["selected_ids"]
            print(f"[parity] {args.arm}: manifest target_n={m['target_n']} "
                  f"(of {m['source']['accepted']} accepted) <- {manifest_path.name}")
        else:
            print(f"[parity] no manifest at {manifest_path} — using full accepted set")

    tokenizer = load_tokenizer()
    ds = load_sft_dataset(args.cause, limit=args.limit, only_ids=only_ids)
    print(f"[data] {len(ds)} SFT pairs from data/sft/{args.cause}.jsonl"
          f"{' (parity-filtered)' if only_ids else ''}")
    report_token_lengths(ds, tokenizer, args.max_length,
                         out_path=log_dir / f"{tag}_lengths.json")

    cfg = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=args.lr,
        lr_scheduler_type=LR_SCHED,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        logging_steps=args.logging_steps,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=args.seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=load_model(),
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(),
        callbacks=[CSVLogger(csv_path)],
    )
    result = trainer.train()
    trainer.save_model(out_dir)
    print(f"[done] {tag} | final loss={result.training_loss:.4f} | csv={csv_path}")
    print(f"[done] adapter -> {out_dir}")


if __name__ == "__main__":
    main()
