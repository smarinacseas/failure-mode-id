# Arm RA — GRPO · coverage (run record) — ⏸️ PAUSED at step-50 gate

Real-arm GRPO run, T1.3. Frozen config: `config/t1_3_frozen.md` (GRPO block).
**Halted by the pre-committed step-50 windowed-trend gate** (flat → PAUSE). Adapter
checkpoints + logs on the `/workspace` volume (gitignored); this is the committed
provenance.

- **Date:** 2026-07-18
- **Env:** A100-SXM4-80GB · torch 2.8.0+cu128 · trl 1.8.0 · transformers 5.14.1 · peft 0.19.1.
- **Data:** `data/train/coverage.jsonl` (300 prompts, hardened/recal pool + per-prompt `specs`).
- **Config (as run):** LR 7.5e-6, `constant_with_warmup` (warmup 0.03); k=6 (frozen); rollout
  temp 0.9; β 0.04; max_completion_length 1536; reward length cap M=2800; per_device 6 ×
  grad_accum 2 (2 unique prompts/step); bf16 + gradient checkpointing + `generation_kwargs
  {use_cache:True}`; `use_vllm=False`; seed 20260715; `--time-budget-sec 10800`.
- **Training health (through step 56):** format_ok 1.0 throughout; KL ~1e-3 (tiny); length
  noisy-flat ~350 tok (no runaway), cap-hit ~0–8%; GPU ~28.7 GB (comfortable). No instability.

## Pre-committed step-50 gate → **PAUSE (FLAT — no reward learning at 50-step scale)**

Rule (`config/t1_3_frozen.md`, PREREG amendment 2026-07-16): *at step 50 compute
mean(reward[1:10]) vs mean(reward[41:50]); flat-or-declining → PAUSE and report; clear
positive slope → proceed.*

| estimator | value | reading |
|---|---|---|
| pre-committed window Δ | **+0.0137** (0.4944 → 0.5081) | positive but **0.16× the end-window noise SD (0.0833)** — single-window artifact |
| OLS slope over 50 steps | **−0.00018/step (t = −0.21)** | statistically zero (slightly negative) |
| halves (mean[1:25] vs mean[26:50]) | 0.5080 vs 0.5076 (**Δ −0.0004**) | dead flat |
| overall | mean 0.5078 ± 0.087 (min 0.339, max 0.671) | noisy-flat around ~0.51 |
| probe reference (hardened pool) | Δ −0.032 (declining) | real run is **not declining**, but **not clearly learning** either |

**Verdict:** FLAT → the pre-committed **PAUSE** branch. The lone positive number (the
+0.0137 window delta) is the single-endpoint-noise artifact the whole windowed methodology
exists to avoid; every robust estimator (OLS over all 50 points, first-half vs second-half)
says no trend. This **confirms the probe's health flag** ("no reward learning at the 50-step
probe scale on hardened coverage") on the real run.

## Action taken

- RA **halted at step 56** (SIGTERM; a straggler orphaned worker under the `nohup` parent was
  also reaped — GPU confirmed freed, 0 MiB).
- **checkpoint-50 preserved** (`results/adapters/T01-RA/checkpoint-50/` — adapter_model.safetensors
  + trainer_state.json), the pre-committed decision-point weights. Steps 51–56 discarded.
- Gate record: `results/logs/RA_gate_step50.json` (with robust_analysis); live log
  `results/logs/grpo_RA_coverage.{jsonl,csv}`; pause marker `results/logs/RA.status`.

## ⚠️ Operator decision required (RA not resumed; RB not started)

This is a substantive decision point, not a mechanical retry. Options — see the report / my
message for the full trade-offs. Nothing further runs until the operator chooses.
