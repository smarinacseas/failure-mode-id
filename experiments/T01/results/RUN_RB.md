# Arm RB — GRPO · precision (run record) — ✅ COMPLETE

Real-arm GRPO run, T1.3. Frozen config: `config/t1_3_frozen.md` (GRPO block). Adapter +
logs on the `/workspace` volume (gitignored); this is the committed provenance.

- **Date:** 2026-07-19
- **Env:** A100-SXM4-80GB · torch 2.8.0+cu128 · trl 1.8.0 · transformers 5.14.1 · peft 0.19.1.
- **Data:** `data/train/precision.jsonl` (300 prompts + per-prompt `specs`).
- **Config (as run):** LR 7.5e-6 `constant_with_warmup`; k=6 (frozen); rollout temp 0.9; β 0.04;
  max_completion_length 1536; reward cap M=2800; per_device 6 × grad_accum 2; bf16 + gradient
  checkpointing + `generation_kwargs{use_cache:True}`; `use_vllm=False`; seed 20260715; 2 epochs.
- **Ran in a single pass** (~1h41m, ~11–20s/step) — precision rollouts are short (mean_len ~150–370
  tok vs RA's ~430–490), so no phasing/3h-cap issue. `time_budget_hit=false`.
- **No pre-committed step-50 PAUSE** (that gate was coverage-specific; precision was in-band in the
  probe). The shared **§9** GRPO-stall gate was checked non-blocking at step 150.

## §9-at-150 (non-blocking health check) → weakly trending up (not a stall)

OLS slope +0.00049/step, **t = 1.89**; first10 0.5111 → last10 0.6493 (Δ +0.138); thirds Δ +0.042.
Every estimator positive → **not** the flat/stalled condition the §9 kill-switch targets (contrast
RA's genuinely-flat step-50, t=−0.21). RB continued to 300; no kill-switch. (`RB_step150_s9.json`.)

## ✅ RB COMPLETE — 2 full epochs (300 steps), GT3 GRPO bar PASSED

Full 1–300 trajectory (`results/logs/RB_full_run.json`):

| estimator (full run) | value |
|---|---|
| **OLS slope over 300 steps** | **+0.00026/step, t = 2.79** — significant positive trend |
| first10 → last10 (windowed) | 0.5111 → **0.7465** (Δ **+0.235**) |
| first-third → last-third | 0.5926 → 0.6460 (Δ +0.053) |
| overall | mean 0.622 ± 0.141 |
| summary final metrics | reward 0.7222, KL 0.006, **cap-hit 0%**, format_ok 1.0, mean_len 367 |
| throughput | 212 tok/s rollout (300 opt-steps in one pass) |

**GT3 GRPO pass bar PASSED** (reward trending up, not flat). RB learns precision, reaching a final
reward comparable to RA (~0.72–0.75), but with a **weaker/noisier trend** (t=2.79 vs RA's t=20.24).

## ⚠️ Preliminary observation — NOT the estimand

The training-reward trend is stronger/cleaner for coverage (RA) than precision (RB) under identical
GRPO. If it survives eval, that hints at a **method×cause interaction**. But the actual estimand is
measured on the **Tier-1 holdout (T1.4)**, not from training reward — this is a signal to test, not
a conclusion. Final adapter: `results/adapters/T01-RB/adapter_model.safetensors` (step 300).
