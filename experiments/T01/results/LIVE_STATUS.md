# T1.3 — LIVE STATUS (auto-generated, read-only)

_Generated: **2026-07-19 06:34:22 UTC** · batch start ≈ 2026-07-18 21:09:31 · source: `results/logs/` (volume) + `config/t1_3_frozen.md`. Regenerate: `python experiments/T01/results/live_status.py`._

> This file is regenerated at checkpoints (arm completion / gate fire), not by continuous polling — it reads training output only and never touches the training processes.

---

## 1 · Run header

- **Frozen config** (`config/t1_3_frozen.md`): SFT LR **1e-4** (cosine, 2 ep) · GRPO LR **7.5e-6** · k=**6** (frozen) · rollout temp 0.9 · β 0.04 · seed **20260715** · 2 epochs · identical LoRA (r16/α32) all arms.
- **Running:** RB  ·  **Queued:** —  ·  **Done:** SA, SB, RA
- **Estimand:** method (SFT vs GRPO) × cause (coverage vs precision). SA/RA = coverage, SB/RB = precision.

---

## 2 · Per-arm status

| arm | method · cause | status | step / total | elapsed | ETA | last loss/reward |
|---|---|---|---|---|---|---|
| **RB** | GRPO · precision | 🟢 running | 5 / 300 (~300→cap) | 1m41s | 1h36m | 0.4833 (reward(last10)) |
| **SA** | SFT · coverage | ✅ done | 16 / 16 | 0m54s | done | 1.6394 (loss) |
| **SB** | SFT · precision | ✅ done | 16 / 16 | 0m28s | done | 2.1714 (loss) |
| **RA** | GRPO · coverage | ✅ done | 300 / 300 | 1h42m | done | 0.6746 (reward(final)) |

---

## 3 · Per-arm diagnostics

### SA — SFT · coverage

| epoch | steps | first loss | last loss | mean loss | Δ | tok-acc (first→last) |
|---|---|---|---|---|---|---|
| 1 | 1–8 | 1.9145 | 1.6108 | 1.7818 | -0.3037 | 0.603→0.639 |
| 2 | 9–16 | 1.5832 | 1.4309 | 1.4970 | -0.1523 | 0.639→0.667 |

**Overall loss slope:** mean(first3) 1.9349 → mean(last3) 1.4253 (Δ **-0.5096**, decreasing ✅).
**Truncation @4096:** 0.00% (ok) | lengths n=123 med 773 p95 982 max 1096.
**Masking sanity:** completion_only_loss=True (user prompt → -100; loss on scaffold+answer); label-build step present in run log ✅.

### SB — SFT · precision

| epoch | steps | first loss | last loss | mean loss | Δ | tok-acc (first→last) |
|---|---|---|---|---|---|---|
| 1 | 1–8 | 2.5435 | 2.2189 | 2.3384 | -0.3246 | 0.496→0.536 |
| 2 | 9–16 | 2.0018 | 2.0417 | 2.0044 | +0.0399 | 0.548→0.530 |

**Overall loss slope:** mean(first3) 2.5162 → mean(last3) 1.9933 (Δ **-0.5229**, decreasing ✅).
**Truncation @4096:** 0.00% (ok) | lengths n=123 med 344 p95 475 max 621.
**Masking sanity:** completion_only_loss=True (user prompt → -100; loss on scaffold+answer); label-build step present in run log ✅.

### RA — GRPO · coverage

**Windowed reward (first10 vs last10 — probe methodology, not single-endpoint):**

| window | steps | mean reward |
|---|---|---|
| first10 | 1–10 | 0.4944 |
| last10 | 291–300 | 0.8004 |
| **Δ** |  | **+0.3060** (rising) |

**Health metrics (last step / summary):**

| metric | value |
|---|---|
| reward (last step, noisy — see windowed table above) | 0.6746 |
| reward_std | 0.2494 |
| kl | 0.0159 |
| format_ok | 1.000 |
| length drift (mean_len first→last) | 430→492 (cap 1536) |
| cap-hit % (last / mean) | 0.0% / 0.9% |
| rollout tok/s | 199.1 |
| step_time (last) | 60.4s |

**3h hardcap behavior + partial-run note.**
> On current pace RA is projected to reach the full step count within the 3h cap. If pace slows (length drift up), truncation risk returns — this note will flip to 🟠. At any cap-stop the adapter still saves (`save_model` + `save_steps=50`).

#### 🔺 RA step-50 windowed-trend gate (pre-committed decision point)

> **🛑 GATE RESULT — PAUSE (FLAT, no learning at 50-step scale).** The pre-committed window delta is **Δ=+0.0137** (mean[1:10]=0.4944 → mean[41:50]=0.5081), but that is only **0.16× the noise SD** (0.0833) — a single-window artifact.
>
> **Robust estimators over all 50 steps confirm flat:** OLS slope **-0.00018/step (t=-0.21)**; halves mean[1:25]=0.508 vs mean[26:50]=0.5076 (**Δ-0.0004**); overall mean 0.5078 ± 0.087. Not declining like the probe (Δ-0.032), but **not the clear positive slope the rule requires to proceed.**
>
> **Action:** per the pre-committed *flat-or-declining → PAUSE* rule, RA was halted at step 56; checkpoint-50 preserved. (My watcher's naive `delta>0→proceed` label was overridden — the rule needs a *clear* slope, not any positive delta.)
>
> ✅ **Operator decision:** RESUME to ~150 steps (the §9 GRPO-stall checkpoint) — 2026-07-19; if reward still flat by ~150, §9 kill-switch converts both RL cells to RFT. LR unchanged 7.5e-6.

#### ✅ §9 checkpoint (step 150) — PASS: reward TRENDING UP

> Over the full 1–150 trajectory: **OLS slope +0.00115/step, t=6.29** (gain ~+0.172 over 150); first10 0.4944 → last10 0.7016; first-third 0.5078 → last-third 0.6108 (Δ+0.1030). **Clear, significant learning — the step-50 flat was a too-short window, exactly as the frozen config anticipated. §9 GT3 bar PASSED; no kill-switch. RA resumed from checkpoint-150 to full 2 epochs (300 steps; 150→300 ≈110 min, fits the 3h cap → no truncation).**

### RB — GRPO · precision

**Windowed reward (first10 vs last10 — probe methodology, not single-endpoint):**

| window | steps | mean reward |
|---|---|---|
| first10 | 1–10 | 0.4833 |
| last10 | 1–5 | 0.4833 |
| **Δ** |  | **+0.0000** (flat/declining) |

**Health metrics (last step / summary):**

| metric | value |
|---|---|
| reward (last step, noisy — see windowed table above) | 0.3542 |
| reward_std | 0.1709 |
| kl | 0.0006031 |
| format_ok | 1.000 |
| length drift (mean_len first→last) | 229→322 (cap 1536) |
| cap-hit % (last / mean) | 0.0% / 0.0% |
| rollout tok/s | n/a until completion |
| step_time (last) | 29.4s |

**3h hardcap behavior + partial-run note.**
> On current pace RA is projected to reach the full step count within the 3h cap. If pace slows (length drift up), truncation risk returns — this note will flip to 🟠. At any cap-stop the adapter still saves (`save_model` + `save_steps=50`).

---

## 4 · Flags / health

- ✅ **RA §9 checkpoint (step 150) PASSED — reward TRENDING UP** (OLS slope +0.00115/step, t=6.29; first10 0.4944→last10 0.7016). The step-50 flat was a too-short window; RA is learning coverage. Resumed to full 2 epochs (300); 150→300 fits the 3h cap (no truncation).

---

## 5 · Artifacts (durable, on /workspace volume — gitignored)

| arm | adapter | live log | summary |
|---|---|---|---|
| SA | ✅ `results/adapters/T01-SA/` | `results/logs/sft_SA_coverage.jsonl` | `RUN_SA.md` (SFT) |
| SB | ✅ `results/adapters/T01-SB/` | `results/logs/sft_SB_precision.jsonl` | `RUN_SB.md` (SFT) |
| RA | ✅ `results/adapters/T01-RA/` | `results/logs/grpo_RA_coverage.jsonl` | `results/logs/grpo_RA_coverage_summary.json` |
| RB | — `results/adapters/T01-RB/` | `results/logs/grpo_RB_precision.jsonl` | — |

_Gate JSON: `results/logs/RA_gate_step50.json` (present)._
