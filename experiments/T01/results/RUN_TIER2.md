# Tier-2 — CC-75 transfer (exploratory/descriptive, PREREG §6) — run record

Does the trained-arm behavior seen on T01's synthetic verifier-checkable pools
(SFT weak+breaking on coverage, GRPO retaining, the H3 general-capability
regression) **transfer** to the naturalistic CC-75 corpus that E08 originally
diagnosed? **Exploratory/descriptive only** (PREREG §6): no confirmatory test, no
CI-excludes-0 verdict, does **not** re-test H1. Raw artifacts on the `/workspace`
volume (`results/eval_t2/` — `responses/`, `grades/`, `spotcheck.json`,
`analysis.json`) are gitignored; this record + the `eval/tier2_*.py` harness are
the committed provenance.

- **Reuses E08** (`dashboard/E08-llama3-2-3b-cc75.json`, `data/complexconstraints.jsonl`):
  the CC-75 corpus (75 prompts, 1,559 criteria) and its diagnosed base-failure labels.
- **Harness (tracked):** `eval/tier2_labels.py` (E08 label loader + subset selector),
  `eval/tier2_decode.py` (local HF decode, mirrors tier1_decode), `eval/tier2_grade.py`
  (opus-4.8 single judge, reuses `pipeline.grade._grade_one` = E08's exact judge
  protocol), `eval/tier2_spotcheck.py` (stack-parity), `eval/tier2_analysis.py` (recovery).

## STEP 1 — E08 reuse validated (2026-07-20)

- **Row-level base data is complete and reusable.** `prompts[].criteria[].results.llama-3b`
  gives base pass/fail + panel votes for all 1,559 criteria; `failure_analysis.rows` gives
  the **254 coverage** (`constraint_unaddressed`) + **321 precision** (`execution_slip`)
  base-failed criteria (text + votes) — the recovery denominators, matching `pareto.json`.
- **⚠️ Indexing trap caught:** `failure_analysis.rows.criterion_index` is **1-BASED**
  (the incoming E08 commit was titled "1-based consensus-FAIL invariant"). A naive 0-based
  positional join mislabels **199/575** denominator rows and drops 29. With `idx0 =
  criterion_index − 1`: **0 unmapped, text present 575/575, base-fail confirmed 575/575,
  0 vote-mismatches** against `prompts[].criteria[]`. Corpus↔dashboard criteria byte-align
  (0 mismatches / 1,559), so corpus and dashboard indices are interchangeable.
- **E08 Arm-0 decode config (the "before"):** temp 0.6, **k=1**, seed 20260715, max 1536,
  reasoning off, via **OpenRouter** (provider Parasail, bf16/fp16 pin); 3-judge panel
  consensus (opus48 + gemini-pro + gpt56sol, opus tie-break).

## STEP 2 — stack-parity spot-check (18 prompts, ~$1.3) → ⚠️ reproduction error is NOT negligible

Local Arm-0 re-decode (opus-4.8 **single** judge, **k=3** majority) vs E08's OpenRouter
**panel** labels, on the same subset criteria (`results/eval_t2/spotcheck.json`; 18 prompts,
seed 20260715, stratified 56 coverage + 62 precision base-failed criteria):

| metric | value |
|---|---|
| **Agreement** (local-opus vs E08-panel base) | **77.1%** (Cohen κ **0.54**, "moderate") |
| Aggregate pass rate | local 49.7% vs E08 48.9% (aggregate-calibrated) |
| **Arm-0 false-recovery, coverage** | **13/56 = 23.2%** of base-fails "pass" locally, zero training |
| **Arm-0 false-recovery, precision** | **12/62 = 19.4%** |

**Interpretation (the standing Tier-2 caveat).** Aggregate pass rates match, but per-criterion
the two graders/stacks disagree ~23% of the time, and **~20% of E08's base-failures flip to
"pass" on an *untrained* local Arm-0** — purely from stack (local HF vs OpenRouter) + grader
(single-opus vs 3-judge panel) + k (3 vs 1). The check cannot decompose the three, but its
**size** is the point: using E08's panel labels directly as the "before" would put a **~20-point
false-recovery floor** under every trained-arm recovery number. This is flagged, not assumed
negligible (per the Step-2 mandate).

**Consequence for the design (adopted):** re-grade a **local Arm-0** on the full 75 with the
*identical* opus/k=3 protocol, and measure recovery against **it** — not E08's panel labels.
Recovery is then computed on the **local-Arm-0-fails ∩ E08-cause-labeled** set, which cancels
the stack+grader+k bias (numerator and reference graded identically) and satisfies §10 threat
#7's local-Arm-0 stack-parity requirement. Cost of doing so is ~$6 (opus is $5/$25 per M).

## STEP 3 — full-pass config (running)

- **Arms (5):** local **0** (baseline reference) + **SA, SB, RA, RB**. Arm P excluded (not a
  trained arm; not central to "post-training capability improvement").
- **Decode:** all 5 × 75 CC-75 prompts, **k=3**, temp 0.6, seed 20260715, max 1536, local HF.
- **Grade:** opus-4.8 single judge (same protocol as Step 2).
- **Denominator:** {E08 cause-labeled base-fail} **∩** {local-Arm-0 majority-fails}.
- **Metric:** `Rec_CC75(arm, cause)` = fraction of that set the arm passes (opus/k=3 majority);
  + breakage complement; prompt-clustered bootstrap CIs (10k, seed 20260715).
- **Budget:** operator ceiling $35; estimate ~$31 (measured ~2.7k in / ~0.5k out tokens/call ×
  ~1,125 calls × opus $5/$25 per M). Spot-check already spent ~$1.3.

## Results — ✅ full 5-arm run complete (2026-07-21)

Ran 2026-07-20 23:31 → 2026-07-21 04:58 UTC (~5.4 h; 5 arms × 75 prompts × k=3 =
1,125 decodes + 1,125 opus grades). Integrity: every arm 225/225 (prompt,decode)
cells graded, **0 judge errors**, raw pass rates 0.41–0.46 (`analysis.json`).

**⚠️ Cost overrun — recorded for provenance.** Actual ~**$121** for the full run
(+~$6 spot-check) vs an estimated ~$31 and an operator ceiling of $35 — a ~3.5×
overrun. Cause: the reused E08 judge protocol runs opus with adaptive thinking
(`pipeline/_judge_llm.py:45`), ~3,761 output tok/call (~7.5× the ~500 estimated),
billed at $25/M → ~$106 of the $121 is reasoning tokens. The estimate omitted
them and the spot-check's *actual* API cost was never measured. Results are valid
and maximally E08-comparable (thinking-on = census protocol). Process fix adopted:
thinking-off by default + per-arm spend checkpoint against any ceiling (see the
`judge-run-cost-protocol` operator memory).

**Denominator (stack+grader+k bias canceled):** coverage **212** (of 254 E08-labeled),
precision **266** (of 321) — the local-Arm-0 intersection dropped ~17%, matching the
Step-2 false-recovery floor (the ~20% of E08 base-fails the untrained local base passes).

**Recovery matrix** `Rec [95% prompt-clustered bootstrap CI]` (10k, seed 20260715):

| arm | Rec(coverage) | Rec(precision) | Breakage |
|-----|---------------|----------------|----------|
| SA (SFT·cov)  | 0.052 [.021,.096] | 0.083 [.047,.128] | **0.228** |
| SB (SFT·prec) | 0.052 [.023,.094] | 0.094 [.047,.156] | **0.260** |
| RA (GRPO·cov) | 0.076 [.036,.130] | 0.068 [.036,.106] | 0.153 |
| RB (GRPO·prec)| 0.085 [.045,.138] | 0.113 [.062,.175] | 0.154 |

**What transfers, what doesn't** (descriptive; Tier-2 is underpowered for the
interaction by design, PREREG §5):

- **The Tier-1 coverage-GRPO advantage does NOT transfer.** RA recovered **74%** of
  Tier-1 *synthetic*-coverage failures; on *naturalistic* CC-75 it recovers **7.6%**.
  Recovery is uniformly low (5–11%) across all four arms. The Tier-2 interaction is
  **≈ −0.005** (vs Tier-1's −0.34) — indistinguishable from 0, CIs overlapping. Read
  plainly: **T1.4's headline coverage effect appears tied to T01's synthetic
  constraint distribution, not a generalizing capability** (corroborates §6(d)'s
  constraint-type-family / pool-authoring confound).
- **The SFT-more-destructive pattern DOES transfer.** SFT arms break 22.8% / 26.0%
  of base-passed criteria vs GRPO's 15.3% / 15.4%. This is now the **third
  independent measurement** of the same behavior: §5.6 (Tier-1 on-task breakage),
  H3 (off-task general-capability regression), and Tier-2 (CC-75 naturalistic
  breakage). It is the robust, transfer-stable finding.

**Stack-parity caveat (standing):** all recovery is measured local-vs-local (opus/k=3
both sides), which cancels the Step-2 bias; but the 77.1% agreement / κ0.54 (§2)
means CC-75 grades carry moderate single-judge/stack noise — held as a qualifier on
the absolute numbers, not the qualitative transfer/non-transfer reading.

