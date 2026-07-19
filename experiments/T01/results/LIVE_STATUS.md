# T1.3 — LIVE STATUS (auto-generated, read-only)

_Generated: **2026-07-19 03:33:32 UTC** · batch start ≈ 2026-07-18 21:09:31 · source: `results/logs/` (volume) + `config/t1_3_frozen.md`. Regenerate: `python experiments/T01/results/live_status.py`._

> This file is regenerated at checkpoints (arm completion / gate fire), not by continuous polling — it reads training output only and never touches the training processes.

---

## 1 · Run header

- **Frozen config** (`config/t1_3_frozen.md`): SFT LR **1e-4** (cosine, 2 ep) · GRPO LR **7.5e-6** · k=**6** (frozen) · rollout temp 0.9 · β 0.04 · seed **20260715** · 2 epochs · identical LoRA (r16/α32) all arms.
- **Running:** RA  ·  **Queued:** RB  ·  **Done:** SA, SB
- **Estimand:** method (SFT vs GRPO) × cause (coverage vs precision). SA/RA = coverage, SB/RB = precision.

---

## 2 · Per-arm status

| arm | method · cause | status | step / total | elapsed | ETA | last loss/reward |
|---|---|---|---|---|---|---|
| **RA** | GRPO · coverage | 🟢 running | 53 / 150 (~150→cap) | 39m44s | 1h15m | 0.5187 (reward(last10)) |
| **SA** | SFT · coverage | ✅ done | 16 / 16 | 0m54s | done | 1.6394 (loss) |
| **SB** | SFT · precision | ✅ done | 16 / 16 | 0m28s | done | 2.1714 (loss) |
| **RB** | GRPO · precision | ⚪ queued | 0 / 300 | 0m00s | — | — (reward(last10)) |

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
| last10 | 44–53 | 0.5187 |
| **Δ** |  | **+0.0242** (rising) |

**Health metrics (last step / summary):**

| metric | value |
|---|---|
| reward (last step, noisy — see windowed table above) | 0.5417 |
| reward_std | 0.1759 |
| kl | 0.0006505 |
| format_ok | 1.000 |
| length drift (mean_len first→last) | 430→332 (cap 1536) |
| cap-hit % (last / mean) | 0.0% / 1.4% |
| rollout tok/s | n/a until completion |
| step_time (last) | 28.6s |

**3h hardcap behavior + partial-run note.**
> On current pace RA is projected to reach the full step count within the 3h cap. If pace slows (length drift up), truncation risk returns — this note will flip to 🟠. At any cap-stop the adapter still saves (`save_model` + `save_steps=50`).

#### 🔺 RA step-50 windowed-trend gate (pre-committed decision point)

> **🛑 GATE RESULT — PAUSE (FLAT, no learning at 50-step scale).** The pre-committed window delta is **Δ=+0.0137** (mean[1:10]=0.4944 → mean[41:50]=0.5081), but that is only **0.16× the noise SD** (0.0833) — a single-window artifact.
>
> **Robust estimators over all 50 steps confirm flat:** OLS slope **-0.00018/step (t=-0.21)**; halves mean[1:25]=0.508 vs mean[26:50]=0.5076 (**Δ-0.0004**); overall mean 0.5078 ± 0.087. Not declining like the probe (Δ-0.032), but **not the clear positive slope the rule requires to proceed.**
>
> **Action:** per the pre-committed *flat-or-declining → PAUSE* rule, RA was halted at step 56; checkpoint-50 preserved. (My watcher's naive `delta>0→proceed` label was overridden — the rule needs a *clear* slope, not any positive delta.)
>
> ✅ **Operator decision:** RESUME to ~150 steps (the §9 GRPO-stall checkpoint) — 2026-07-19; if reward still flat by ~150, §9 kill-switch converts both RL cells to RFT. LR unchanged 7.5e-6. **Resumed from checkpoint-50 → running to step 150; the §9 GRPO-stall kill-switch adjudicates there** (reward trending up by ~150 → continue; still flat → both RL cells to RFT).

### RB — GRPO · precision

_no reward rows yet._

---

## 4 · Flags / health

- 🔄 **RA step-50 gate read FLAT (OLS slope -0.00018/step, t=-0.21); operator resumed to the §9 checkpoint (step 150).** If reward is still flat by ~150, the §9 GRPO-stall kill-switch converts both RL cells to RFT; if trending up, RA continues. Watch this space.

---

## 5 · Artifacts (durable, on /workspace volume — gitignored)

| arm | adapter | live log | summary |
|---|---|---|---|
| SA | ✅ `results/adapters/T01-SA/` | `results/logs/sft_SA_coverage.jsonl` | `RUN_SA.md` (SFT) |
| SB | ✅ `results/adapters/T01-SB/` | `results/logs/sft_SB_precision.jsonl` | `RUN_SB.md` (SFT) |
| RA | — `results/adapters/T01-RA/` | `results/logs/grpo_RA_coverage.jsonl` | — |
| RB | — `results/adapters/T01-RB/` | — | — |

_Gate JSON: `results/logs/RA_gate_step50.json` (present)._
