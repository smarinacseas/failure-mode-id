# T1.3 — Method × Cause interaction: SFT vs GRPO × coverage vs precision · FINAL REPORT

> **STATUS: SCAFFOLD (2026-07-18).** Structure + stable methodology/provenance are
> in place; every *results* and *eval* cell is marked **⏳ PENDING** and will be
> populated only from real artifacts after all four arms **and** the Tier-1 holdout
> eval (T1.4) complete. No number in this report is invented — PENDING means not yet
> measured. Live progress meanwhile: [`LIVE_STATUS.md`](./LIVE_STATUS.md).

Front-door report for T1.3 training + T1.4 eval. It reads as a standalone summary
and **links** to `PREREG.md`, the amendment log, manifests, and probe files rather
than re-deriving them. Pre-registration is frozen in
[`../PREREG.md`](../PREREG.md); the coverage-recal + sft-parity amendments are commit
`c2348c6` on `e08-t01`.

---

## 1 · Executive summary

⏳ PENDING (after T1.4 eval). Will state, plainly and in one place:

- **Headline interaction (the estimand):** does the SFT→GRPO improvement differ by
  cause (coverage vs precision)? Direction + magnitude of the method×cause
  interaction on Tier-1 holdout criterion-pass. `⏳ effect size ± CI`.
- **Per-cause method gaps:** GRPO−SFT on coverage `⏳`, on precision `⏳`, each with
  the pre-registered **cluster/cluster-bootstrap CI** (§6).
- **Pre-registered significance call** per §8 outcome-interpretation table `⏳`.
- **The load-bearing caveats up front** (see §5): coverage was recalibrated
  mid-T1.3; the GRPO probe flagged *no reward learning at 50-step scale on hardened
  coverage* — whether that resolved on the full RA run is a headline input.

_TL;DR bullets (E08-census style) to be written last, from the §3/§4 tables._

---

## 2 · Methodology recap

One paragraph, no re-derivation — see [`../PREREG.md`](../PREREG.md) §§3–7 for the
frozen design. Briefly: a 2×2 — method {SFT (teacher-distilled, completion-only
LoRA), GRPO (on-policy verifier-reward LoRA)} × cause {coverage = CAUSE_A,
precision = CAUSE_B}, all four arms sharing an **identical LoRA** (r16/α32, all
attn+MLP linears) on Llama-3.2-3B-Instruct, trained on disjoint per-cause pools and
scored on a reserved 200/cause holdout. Estimand = the method×cause interaction on
holdout criterion-pass; analysis = pre-registered bootstrap (§6). Execution knobs
(estimand-immune) are frozen in [`../config/t1_3_frozen.md`](../config/t1_3_frozen.md);
GRPO LR 7.5e-6 was probe-selected ([`../config/t1_3_grpo_probe.md`](../config/t1_3_grpo_probe.md)).

---

## 3 · Per-arm results (training)

Final training metrics + diagnostics summary; adapters/logs are durable on the
`/workspace` volume (gitignored `/results/`), per-arm provenance in `RUN_*.md`.

| Arm | Method · Cause | n (parity) | Steps | Final loss / reward | Learning (GT3) | Deviations / flags | Record |
|-----|----------------|-----------|-------|---------------------|----------------|--------------------|--------|
| **SA** | SFT · coverage | 123/123 | 16 | loss 1.935→1.425 (Δ−0.510) | ✅ PASS | none | [`RUN_SA.md`](./RUN_SA.md) |
| **SB** | SFT · precision | 123/270 | 16 | loss 2.516→1.993 (Δ−0.523) | ✅ PASS | none | [`RUN_SB.md`](./RUN_SB.md) |
| **RA** | GRPO · coverage | 300 prompts | ⏳ | ⏳ windowed reward | ⏳ | **step-50 gate: ⏳**; probe "no-learning@50" flag — resolved? ⏳ | ⏳ `RUN_RA.md` |
| **RB** | GRPO · precision | 300 prompts | ⏳ | ⏳ windowed reward | ⏳ | ⏳ | ⏳ `RUN_RB.md` |

**GT3 pass bar (§ amendment (a)):** all four arms show learning — SFT loss clearly
decreasing (SA/SB ✅), GRPO mean reward trending up by ~150 steps (RA/RB ⏳).
**RA step-50 windowed-trend gate** (pre-committed decision point): ⏳ — result +
whether the "no reward learning at probe scale on hardened coverage" health flag
(probe Δ=−0.032) reproduced or resolved on the full run.

---

## 4 · Tier-1 holdout eval results (the estimand)

⏳ PENDING (T1.4). From the reserved 200/cause holdout, decoded per §5 (temp 0.6,
k=3, seed 20260715), graded by the frozen verifiers:

| Cause | SFT pass % | GRPO pass % | GRPO−SFT gap | Cluster-bootstrap 95% CI |
|-------|-----------|-------------|--------------|--------------------------|
| coverage (SA vs RA) | ⏳ | ⏳ | ⏳ | ⏳ |
| precision (SB vs RB) | ⏳ | ⏳ | ⏳ | ⏳ |
| **interaction (Δgap)** | | | ⏳ | ⏳ |

- **Cross-cause comparison / interaction:** `(RA−SA) − (RB−SB)` with the
  pre-registered CI (§6) `⏳`.
- **Base-subject reference** (untrained) per cause for context `⏳`.
- Decode/cap-hit health at eval `⏳`.

---

## 5 · Threats to validity

Links to [`../PREREG.md`](../PREREG.md) §10 + the amendment log; summarized here.

- **Coverage recalibration (mid-T1.3).** CAUSE_A difficulty was recalibrated into
  the 30–70% band (commit `c2348c6`; [`../config/recalibration_stopping_rule.md`](../config/recalibration_stopping_rule.md)):
  hardened generators, retired near-free binary types, regenerated pools. Effect on
  interpretation: `⏳` — coverage remains disjoint from precision (estimand intact),
  but absolute coverage numbers are post-recal and not comparable to pre-recal.
- **Teacher-yield asymmetry → parity down-sample.** Coverage kept 123/300 vs
  precision 270/300; both SFT cells down-sampled to target_n=123 to avoid confounding
  the interaction with accepted-count (manifests below). `⏳` residual risk.
- **Survivor-composition bias.** The accepted SFT pairs are the teacher's *solved*
  subset; per-type composition differs across causes. `⏳`.
- **Coverage headroom / GRPO learnability.** Probe: no reward learning at 50-step
  scale on hardened coverage (Δ=−0.032). Whether RA learns on the full run is both a
  result (§3) and a validity question for the coverage arm. `⏳`.
- **New threats surfaced during training:** `⏳` (e.g. env re-provision on a fresh
  pod container — versions re-verified against the frozen lock; length drift / KL
  behavior; cap-hit).

---

## 6 · Provenance appendix

Pointers, not duplicates.

- **Pre-registration:** [`../PREREG.md`](../PREREG.md) (§§3–10 design; amendment log).
- **Amendments (append):** commit `c2348c6` (coverage-recal + sft-parity);
  drafts [`../config/DRAFT_amendment_2026-07-16_coverage-recal.md`](../config/DRAFT_amendment_2026-07-16_coverage-recal.md),
  [`../config/DRAFT_amendment_2026-07-16_sft-parity.md`](../config/DRAFT_amendment_2026-07-16_sft-parity.md).
- **Frozen training config:** [`../config/t1_3_frozen.md`](../config/t1_3_frozen.md);
  GRPO LR probe [`../config/t1_3_grpo_probe.md`](../config/t1_3_grpo_probe.md);
  stopping rule [`../config/recalibration_stopping_rule.md`](../config/recalibration_stopping_rule.md).
- **Manifests (parity):** `data/sft_manifests/SA.json` (123/123), `SB.json` (123/270), seed 20260715.
- **Runners:** `training/sft.py`, `training/grpo.py`, `training/callbacks.py`, `training/reward_adapter.py`.
- **Per-arm records:** [`RUN_SA.md`](./RUN_SA.md), [`RUN_SB.md`](./RUN_SB.md), ⏳ `RUN_RA.md`, ⏳ `RUN_RB.md`.
- **Durable artifacts (volume, gitignored):** `results/adapters/T01-{SA,SB,RA,RB}/`,
  `results/logs/{sft_*,grpo_*}.{jsonl,csv}`, `results/logs/grpo_*_summary.json`,
  `results/logs/RA_gate_step50.json`.
- **Probe evidence:** `results/probe/base_difficulty.json`, `results/probe/coverage_by_type.json`,
  `results/probe/grpo_RA_coverage_lr*_hard*`.
- **Live status generator:** [`live_status.py`](./live_status.py) → [`LIVE_STATUS.md`](./LIVE_STATUS.md).

---

_Generated by hand as a scaffold; §§1,3(RA/RB),4 to be filled from real artifacts at
T1.3/T1.4 completion. This report is the readable front door — PREREG.md remains the
authority._
