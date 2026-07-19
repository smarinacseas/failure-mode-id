# DRAFT amendment — coverage difficulty recalibration (for reviewer approval)

> Staging file. Do NOT append to `PREREG.md` until wording is approved.
> Companion to the stopping rule in `recalibration_stopping_rule.md`.

---

### 2026-07-16 — CAUSE_A (coverage) difficulty recalibration

**Trigger (process metric, pre-training).** The base subject's coverage
criterion-pass was **out of the PREREG §5 30–70% band**: 86.4% at the GRPO
rollout regime (temp 0.9) and ~75% at the eval regime (temp 0.6) — measured
before any real-arm training. Per the pre-committed T1.3 STOP rule (verifier
component >0.70 ⇒ recalibrate before real arms), CAUSE_A is recalibrated.
Precision (CAUSE_B) was in band at the rollout regime (61.1% temp 0.9, k=6;
`results/probe/base_difficulty.json`) — below the >0.70 recalibration trigger — so it
is **untouched**. No temp-0.6 acceptance measurement is taken for precision: the
acceptance-regime protocol (`recalibration_stopping_rule.md`) governs only the cause
being recalibrated (coverage).

**Stopping rule (committed before any acceptance number was read).** Acceptance
regime = eval regime **temp 0.6, k=3, N=60, seed 20260715** (matches §5 Tier-1
decode and the original T1.2 calibrate.py). Cap 3 generator versions; accept the
**first** version in [30%,70%]; no within-band selection by preference; one
measurement per version; escalate if none of 3 land in band. Full rule + version
log: `config/recalibration_stopping_rule.md`.

**What changed (coverage generator only).**
- Hardened count-based generators (harder for the base 3B, still teacher-tractable):
  keyword_include 4–6 → **7–10** keywords; required_sections 3–5 → **6–8**;
  keyword_exclude 1–2 → **3–5**.
- Retired the two near-free binary types from **sampling** — no_placeholders
  (base ~98–100% pass) and title (~92%) — which held the mean above band. Their
  verifiers/generators remain registered + archetype-assigned (provenance).
- Added a stricter coverage type **start_phrase** (exact opening sentence; base
  passes ~8% — mirror of end_phrase).
- Constraints per coverage prompt: 5–6 → **6–7** (k capped at |sample pool| = 7).
- Coverage train (300) + holdout (200) **regenerated** (build_pools --cause
  coverage, split seed 20260715, full 13-gram external screen vs CC-75 + IFEval:
  **0 contamination hits**, 0 dups). Precision pools byte-identical (md5 verified).

**Acceptance result.** v2 (the change set above) landed at **61.0%** eval-regime
criterion-pass on the fresh-composed sample AND **61.0%** on the regenerated train
pool (identical — same seed 20260715, 0 dedup rejects) — in band, first version at
the acceptance regime → accepted (no v3, no tuning). v1 (count-hardening only) was
74.4% at temp 0.9 (exploratory) — logged, not used. Per-type on the regenerated
pool: start_phrase 7.5, keyword_include 38.6, end_phrase 47.3, keyword_exclude
67.3, casing 80.9, no_commas 93.3, required_sections 95.0.

**Downstream (forced by the recount).**
- Coverage SFT teacher data **regenerated** from the new train pool (teacher_gen
  --cause coverage). New accepted count = **123** (was 204); GT2 yield **41.0% —
  PASS (≥30%)** (~3.0 attempts/prompt; teacher tokens 319.2k prompt + 1.455M completion).
  Pre-recal 204-record file archived at `results/probe/coverage_sft_pre-recal_204.jsonl`.
- Parity target_n recomputed = min(123 coverage, 270 precision) = **123** (was 204);
  SFT manifests regenerated (SA coverage 123/123 identity, SB precision 123/270,
  seed 20260715). The sft-parity amendment's target_n=204 is **superseded**.
- GRPO LR re-probed on the hardened coverage pool (the 7.5e-6 pick was made on the
  easy pool): **7.5e-6 re-confirmed, no revision** — reward now starts ~0.435 (was
  ~0.77; real headroom), but the re-probe **does not discriminate**: on the hardened
  pool all three LRs are flat-to-declining over 50 steps and within noise (windowed
  Δ mean[40:50]−mean[0:10]: 5e-6 −0.018, 7.5e-6 −0.032, 1e-5 −0.037; windowed
  end-std 0.196 / 0.202 / 0.188), so there is no signal to revise 7.5e-6 (mid-range,
  single frozen LR both arms, no per-arm tuning). Health flag carried to the STOP
  report: **no reward learning at the 50-step probe scale on hardened coverage**
  (contrast the easy pool's +0.08 climb; length did not drift up). A **pre-committed
  RA step-50 windowed-trend check** is added: at step 50 of the real RA run compute
  mean(reward[0:10]) vs mean(reward[40:50]) — if flat-or-declining, PAUSE and report;
  if clear positive slope, proceed. Full analysis + table: `t1_3_grpo_probe.md`
  § "Re-probe on the HARDENED coverage pool".

**Unchanged.** Estimand, hypotheses (§§3–4), sample sizes, seeds (§7), analysis
(§6), precision pools/holdout/SFT, and the 2-epoch schedule. Coverage remains
coverage (disjoint from precision) — the interaction estimand is intact.
