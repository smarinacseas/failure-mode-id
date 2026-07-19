"""GRPO runner — Arms RA (coverage) and RB (precision). PREREG §5 + amendment
2026-07-16 (b)(c)(d).

On-policy verifier-feedback RL with LoRA (identical adapter to SFT; only the data
differs). Rollouts run on the stock `model.generate()` backend —
`GRPOConfig(use_vllm=False)` — per amendment (d) (no vLLM; speed-only, same
sampling distribution). Reward = fraction of the prompt's constraints its final
answer satisfies, minus a malformed penalty, length-capped (verifiers/reward.py
via reward_adapter). k (num_generations) is frozen at 6 — NOT a sweep axis
(amendment (b)); the probe sweeps LR only.

Run (probe / full):
  python grpo.py --arm RA --cause coverage --limit 50 --epochs 2 --lr 1e-5 \
                 --time-budget-sec 10800 --run-tag lr1e-5
  python grpo.py --arm RA --cause coverage --epochs 2 --lr <frozen>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

from callbacks import CSVLogger, GenerateMeter, TimeBudget
from common import CAUSES, MODEL_PATH, load_model, load_tokenizer, lora_config
from datasets_t01 import load_grpo_dataset
from reward_adapter import make_constraint_reward

WORKSPACE = Path("/workspace/failure-mode-id/results")

# Frozen GRPO knobs (PREREG §5 design table + amendment (b)(c)); estimand-immune.
NUM_GENERATIONS = 6           # k — FROZEN (amendment (b)), not swept
ROLLOUT_TEMP = 0.9
BETA = 0.04                   # TRL default KL coeff; change only if unstable, then log
MAX_COMPLETION_LENGTH = 1536  # must fit answer before ===FINAL===; report cap-hit at probe
MAX_CHARS = 2800              # length cap M — just above longest teacher answer (2617 ch)
PER_DEVICE_BS = 6             # one k-group per device batch
GRAD_ACCUM = 2                # -> generation_batch_size 12 = 2 unique prompts/step
WARMUP_RATIO = 0.03
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 3
# NOTE: TRL 1.8.0 GRPOConfig has NO `max_prompt_length` field (amendment (c) named
# 512). Deviation is inert: coverage/precision prompts are <=166 tokens (measured),
# so nothing is truncated. Logged, not silently dropped (PREREG §9).


def _tail_metrics(log_history: list[dict], keys) -> dict:
    out = {}
    for k in keys:
        vals = [h[k] for h in log_history if k in h]
        if vals:
            out[k] = round(float(vals[-1]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["RA", "RB"])
    ap.add_argument("--cause", required=True, choices=list(CAUSES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--max-completion-length", type=int, default=MAX_COMPLETION_LENGTH)
    ap.add_argument("--time-budget-sec", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="hard step ceiling (overrides epochs); e.g. 150 for the §9 checkpoint")
    ap.add_argument("--resume-from-checkpoint", default=None,
                    help="path to a checkpoint-N dir to resume optimizer/scheduler/RNG/step from")
    ap.add_argument("--continuous-batching", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()

    out_dir = args.output_dir or str(WORKSPACE / "adapters" / f"T01-{args.arm}")
    log_dir = WORKSPACE / ("probe" if args.sanity else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"grpo_{args.arm}_{args.cause}{('_' + args.run_tag) if args.run_tag else ''}"
    csv_path = str(log_dir / f"{tag}.csv")

    print(f"== GRPO {args.arm} ({args.cause}) | lr={args.lr} beta={args.beta} k={NUM_GENERATIONS} "
          f"temp={ROLLOUT_TEMP} max_comp={args.max_completion_length} M={args.max_chars} ==")
    print(f"== use_vllm=False continuous_batching={args.continuous_batching} "
          f"budget={args.time_budget_sec}s out={out_dir} ==")

    tokenizer = load_tokenizer()
    ds = load_grpo_dataset(args.cause, limit=args.limit)
    print(f"[data] {len(ds)} GRPO prompts from data/train/{args.cause}.jsonl")

    cfg = GRPOConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        max_completion_length=args.max_completion_length,
        temperature=ROLLOUT_TEMP,
        beta=args.beta,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=WARMUP_RATIO,
        use_vllm=False,
        use_transformers_continuous_batching=args.continuous_batching,
        bf16=True,
        # GC ON (memory) + force use_cache=True in generation (throughput). GC's
        # gradient_checkpointing_enable() sets config.use_cache=False, which would
        # disable the KV cache during rollouts (~2.6x slowdown: 192 vs 500 tok/s);
        # generation_kwargs re-enables it ONLY for generate() (533 tok/s measured),
        # while the training forward still runs checkpointed. GC-off OOM'd at
        # ~75 GB on lr=1e-5 (length spikes), so checkpointing is required for headroom.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        generation_kwargs={"use_cache": True},
        logging_steps=args.logging_steps,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=SAVE_TOTAL_LIMIT,
        log_completions=True,
        seed=args.seed,
        report_to="none",
    )

    time_cb = TimeBudget(args.time_budget_sec) if args.time_budget_sec else None
    callbacks = [CSVLogger(csv_path)] + ([time_cb] if time_cb else [])

    trainer = GRPOTrainer(
        model=load_model(),
        reward_funcs=[make_constraint_reward(max_chars=args.max_chars)],
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config(),
        callbacks=callbacks,
    )

    # Measure real generate() rollout throughput on the underlying causal LM
    # (peft_model.base_model.model) — the object TRL calls .generate() on.
    meter = GenerateMeter()
    policy = getattr(getattr(trainer.model, "base_model", trainer.model), "model", trainer.model)
    meter.attach(policy)

    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(out_dir)

    tail = _tail_metrics(trainer.state.log_history,
                         ["reward", "reward_std", "completions/mean_length",
                          "completions/max_length", "completions/clipped_ratio", "kl", "loss"])
    summary = {
        "tag": tag, "arm": args.arm, "cause": args.cause, "lr": args.lr, "beta": args.beta,
        "epochs": args.epochs, "n_prompts": len(ds), "steps": trainer.state.global_step,
        "throughput": meter.summary(),
        "final_metrics": tail,
        "cap_hit_ratio_last": tail.get("completions/clipped_ratio"),
        "time_budget_hit": bool(time_cb and time_cb.hit),
        "continuous_batching": args.continuous_batching,
        "max_completion_length": args.max_completion_length, "max_chars": args.max_chars,
    }
    (log_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    print("[summary]", json.dumps(summary, indent=2))
    print(f"[done] {tag} | csv={csv_path} | adapter -> {out_dir}")


if __name__ == "__main__":
    main()
