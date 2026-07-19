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
- **checkpoint-50 preserved at pause time** (`results/adapters/T01-RA/checkpoint-50/` —
  adapter_model.safetensors + trainer_state.json), the pre-committed decision-point weights.
  Steps 51–56 discarded. ⚠️ **This checkpoint was later rotated out** by `save_total_limit=3`
  during the final 150→300 phase — see "Provenance note" at the end of this record.
- Gate record: `results/logs/RA_gate_step50.json` (with robust_analysis); live log
  `results/logs/grpo_RA_coverage.{jsonl,csv}`; pause marker `results/logs/RA.status`.

## Operator decision (2026-07-19): RESUME to the §9 checkpoint (~150 steps)

The step-50 gate was the *early warning*; the pre-registered failure path is the **§9
GRPO-stall kill-switch at ~150 steps** (reward not trending up by ~150 → both RL cells → RFT).
50 steps may be too short to detect learning (the frozen config itself flags "no reward
learning *at the 50-step scale*"). So RA is resumed to let §9 adjudicate:

- **Resumed from `checkpoint-50`** (clean HF resume — optimizer.pt / scheduler.pt / rng_state.pth /
  global_step 50 all restored; LR stays constant **7.5e-6**, past warmup). Runner gained
  `--resume-from-checkpoint` + `--max-steps` (execution-only, estimand-neutral).
- **`--max-steps 150`** → runs steps 51→150 then stops cleanly (save_model + summary). At ~40s/step
  this is ~65 min; well inside the 3h cap, so **no truncation risk for this phase**.
- Live log rebuilt to a continuous 1→150 trajectory (steps 1–50 from the original run that produced
  checkpoint-50; discarded 51–56 archived to `grpo_RA_coverage_pre-resume_raw.jsonl`).

**Next decision at step 150 (§9):** if reward is trending up → RA continues toward completion; if
still flat → the §9 kill-switch converts **both** RL cells (RA, RB) from GRPO to RFT. I will report
the §9 read and confirm before any RFT conversion.

## §9 checkpoint (step 150) — ✅ PASS: reward TRENDING UP

Over the full 1–150 trajectory (robust estimators, not single endpoints;
`results/logs/RA_step150_s9.json`):

| estimator | value | reading |
|---|---|---|
| OLS slope over 150 steps | **+0.00115/step, t = 6.29** | strongly significant positive trend (gain ~+0.17) |
| first10 → last10 | 0.4944 → **0.7016** (Δ +0.207) | clear rise |
| first-third → last-third | 0.5078 → 0.6108 (Δ +0.103) | clear rise |
| step-150 summary final reward | **0.6806** (std 0.092, cap-hit 0%, KL 0.013) | healthy |

**RA is learning coverage.** The step-50 flat was simply too short a window — exactly the
limitation the frozen config flagged ("no reward learning *at the 50-step scale*"). **§9 GT3 bar
PASSED; no kill-switch fired.**

## Resumed to full 2 epochs (150 → 300)

Per the operator's "if trending up, RA continues" path, RA was resumed from **checkpoint-150**
(`--max-steps 300`) to complete the frozen 2-epoch spec.

- **Truncation concern resolved by phasing:** 150→300 = 150 steps at ~44s/step ≈ **110 min**, well
  inside the 3h cap → **no truncation, RA reaches the full 300 steps (2 epochs)**, symmetric with
  RB's eventual 2 epochs. (The earlier partial-run risk applied only to a single 3h run from step 0;
  phased resumes each fit the budget.)
- checkpoints preserved (200/250/300 retained; earlier rotated by save_total_limit=3); log continuous 1→300.

## ✅ RA COMPLETE — 2 full epochs (300 steps), GT3 GRPO bar PASSED

Completed cleanly, **no truncation** (`time_budget_hit=false`; ran ~1h42m in the final phase).
Full 1–300 trajectory (`results/logs/RA_full_run.json`):

| estimator (full run) | value |
|---|---|
| **OLS slope over 300 steps** | **+0.00123/step, t = 20.24** — decisive upward trend |
| first10 → last10 (windowed) | 0.4944 → **0.8004** (Δ **+0.306**) |
| first-third → last-third | 0.5197 → 0.7686 (Δ +0.249) |
| overall | mean 0.649 ± 0.140 |
| summary final metrics | reward 0.6746 (last noisy step), KL 0.016, **cap-hit 0%**, format_ok 1.0, mean_len 492 (no runaway) |
| throughput | 199 tok/s rollout (150 opt-steps this phase) |

**GT3 GRPO pass bar (reward trending up, not flat by ~150): PASSED decisively.** The coverage arm
learns strongly under GRPO — the step-50 flat was purely a window-length artifact. Final adapter:
`results/adapters/T01-RA/adapter_model.safetensors` (step 300). RB (precision GRPO) starts next.

## ⚠️ Provenance note (2026-07-19): decision-point checkpoints rotated out

`checkpoint-50` (the step-50 PAUSE weights) and `checkpoint-150` (the §9 decision-point weights)
**no longer exist on disk** — both were rotated out by `save_total_limit=3` during the final
150→300 phase. Only `checkpoint-{200,250,300}` + the final step-300 adapter remain (verified
`ls results/adapters/T01-RA/` → checkpoint-200/250/300 only; checkpoint-50 and checkpoint-150 both
`No such file or directory`).

- **Impact on training / T1.4: none.** The final adapter (step 300) is the artifact carried
  forward; the intermediate checkpoints were only decision waypoints.
- **Provenance gap:** the exact adapter weights at steps 50 and 150 can no longer be
  re-instantiated. (The earlier "checkpoint-50 preserved" line above described its state *at the
  pause*, before the 150→300 phase rotated it out.)
- **What IS still preserved — the evidentiary basis is intact.** Both the PAUSE and §9 calls were
  made on the *logged reward trajectory*, not by re-evaluating checkpoint weights, and every input
  to those decisions survives: `results/logs/RA_gate_step50.json` (step-50 robust analysis),
  `results/logs/RA_step150_s9.json` (§9 read), the continuous per-step live logs
  `grpo_RA_coverage.{jsonl,csv}`, and the full 1→300 `log_history` embedded in
  `checkpoint-300/trainer_state.json`. The numbers the decisions rested on are fully recoverable;
  only the weight snapshots are not.
