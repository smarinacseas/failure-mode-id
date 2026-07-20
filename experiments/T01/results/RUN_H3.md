# H3 — Tier-3 general-capability regression guard — ❌ FAIL (marginal: SA −3.2 pp, CI spans the −3 line)

Pre-registered guard (PREREG **§4, H3**): *"no trained arm scores more than 3
points below Arm 0 on the Tier-3 general-capability battery. Violations are
reported prominently."* H3 is a **regression guard, not a hypothesis to confirm**
— it protects against fine-tuning silently wrecking general capability.

> ## ❌ VERDICT: FAIL by the pre-registered point-estimate rule — but a **marginal** one.
> **Arm SA is 3.2 pp below Arm 0** (49.2% vs 52.4%), past the 3-point line, so the
> literal guard (a point-estimate rule, no CI) returns FAIL. **But SA's Δ 95% CI is
> [−5.3, −1.1], which includes the −3.0 threshold** — stated precisely: *the point
> estimate fails, the CI includes the threshold.* The regression is real (CI
> excludes 0) but its crossing of the 3-point line is within sampling noise, and SA
> (−3.2) is statistically indistinguishable from SB (−3.0, which "passes"). The
> other three trained arms pass: SB −3.0 (on the line, not *past* it), RA −0.1,
> RB +0.9. Raw artifacts on the `/workspace` volume
> (`results/eval_t3/` — per-arm `*.jsonl`, `*_summary.json`, `mmlu_subset.jsonl`,
> `delta_ci.json`, `verdict.json`, `run.log`) — gitignored by design; this tracked
> record is the committed provenance.

- **Run:** 2026-07-20 20:18–20:22 UTC (~3m15s, all 6 arms + aggregate; single A100-80GB).
- **Battery (operator-approved 2026-07-20, *not* pre-frozen in PREREG — §4 defines the
  guard but never froze a battery; composition chosen and approved before any arm was
  scored, so the battery is not selected on results):** **MMLU, 0-shot, deterministic
  answer-letter log-likelihood scoring.** Fixed stratified subset **N=1000** across all
  57 subjects, selection seed **20260715** (project seed), identical subset for every arm.

## 1. Raw scores — all 6 arms (paired on the same 1000 questions)

| arm | role | accuracy | correct/N | **Δpp vs Arm 0** | 95% CI (paired bootstrap) | gated by H3? |
|-----|------|----------|-----------|------------------|---------------------------|--------------|
| **0**  | base (denominator) | **52.4%** | 524/1000 | 0.0 | — | — |
| P   | base + enumerate/verify sys-prompt | 38.0% | 380/1000 | −14.4 | [−18.0, −10.9] | no (not a trained arm) |
| **SA** | SFT · coverage | **49.2%** | 492/1000 | **−3.2** | [−5.3, −1.1] | **yes → ❌ VIOLATION** |
| SB  | SFT · precision | 49.4% | 494/1000 | −3.0 | [−5.4, −0.5] | yes (pass, boundary) |
| RA  | GRPO · coverage | 52.3% | 523/1000 | −0.1 | [−0.9, +0.7] | yes (pass) |
| RB  | GRPO · precision | 53.3% | 533/1000 | +0.9 | [+0.1, +1.8] | yes (pass) |

Bootstrap: 10,000 paired resamples of the 1000 items, seed 20260715 (`delta_ci.json`).
Verdict rule applied in `tier3_aggregate.py`: FAIL iff any *trained* arm's Δ point
estimate `< −3.0`. SA (−3.2) is the sole trigger.

## 2. The violation — Arm SA — reported plainly, with its uncertainty

**SA (SFT·coverage) scores 49.2% vs Arm 0's 52.4% → −3.2 pp, past the 3-point
guard. Per the pre-registered rule, H3 FAILS.** This is stated prominently, not buried.

Two honest qualifications that do **not** overturn the literal verdict but bound what it means:

1. **The crossing is within sampling noise.** SA's Δ 95% CI is [−5.3, −1.1] — it
   excludes 0 (SA *did* genuinely regress) but comfortably **includes −3.0**. SA (−3.2)
   and SB (−3.0) are statistically indistinguishable from each other and from the
   threshold. The SA-fails / SB-passes split is a knife-edge: it rests on a 0.2 pp point
   estimate gap against a hard line, not on a real capability difference between the two
   SFT arms. H3 has **no pre-registered CI** — it is a point-estimate guard — so the
   literal verdict stands at FAIL, but honesty requires stating that the exact crossing is
   inside the noise band.
2. **This is a *knowledge* regression, not SA's generation pathology.** MMLU here is
   letter-loglik — one forward pass, no generation — so it is immune to the degenerate
   repetition/truncation loops that hurt SA at the T1.4 decode (RUN_T1.4 §2–4). SA's
   −3.2 pp is a real drop in the model's next-token answer knowledge under the LoRA,
   independent of and additional to its decode-time pathology.

## 3. The robust pattern — method, not cause, drives general-capability retention

Setting aside the exact 3-point line, the four trained arms split cleanly by **method**:

| | coverage | precision | method summary |
|---|---|---|---|
| **SFT** (SA, SB) | −3.2 | −3.0 | **both regress ~3 pt** (CIs both cap below −0.5) |
| **GRPO** (RA, RB) | −0.1 | +0.9 | **both retain** (RA flat; RB CI excludes 0 on the *positive* side) |

Both SFT arms lost general capability; neither GRPO arm did (RB modestly improved).
This is consistent with the mechanism: GRPO ran at a **tiny KL** (RA/RB KL ~0.006–0.016
throughout training — RUN_RA/RUN_RB), staying near the base policy, whereas SFT
completion-only imitation on 123 narrow-format examples overwrites more broadly. The
*cause* (coverage vs precision) does **not** predict retention within a method — it is the
method axis that separates the arms. (Descriptive observation; carried into
`T1_3_FINAL_REPORT.md` discussion, not a confirmatory claim.)

## 4. Arm P — the −14.4 is a scaffold measurement artifact, not capability loss

Arm P shares Arm 0's exact weights (it is base + the enumerate/verify **system prompt**),
so its *knowledge* is identical to Arm 0's by construction. It scores 38.0% only because
letter-loglik reads the **first** assistant token, and under the scaffold that token is the
opening of a checklist ("(1)…", "ENUMERATE…"), not a bare answer letter — so the letter
mass is depressed. P's number is recorded because P is a live comparison point elsewhere
(the SB-vs-Arm-P discussion in the final report), **not** as a capability claim, and P is
correctly **not** gated by H3 (§4 gates only *trained* arms).

## 5. Methodology (reproducible)

- **Scoring:** Hendrycks / lm-eval-harness MMLU protocol adapted to a chat model. Question
  + A/B/C/D formatted as a user turn; chat template with `add_generation_prompt=True`;
  one forward pass; compare the log-softmax log-probability of the four **bare** letter
  tokens (A/B/C/D → ids 32/33/34/35, probe-asserted single-token) at the final
  (left-padded) position; argmax = prediction. **Fully deterministic** — no sampling, no
  decode seed — so the arm−base delta is a clean paired difference (unlike T1.4's temp-0.6
  decode). Reads the immediate answer distribution, so a LoRA arm whose *formatting*
  drifted is penalised only for lost knowledge, not format.
- **Subset:** all 57 MMLU subjects (`cais/mmlu`, ungated); per-subject quota 17–18
  (31 subjects @18 + 26 @17 = 1000); within-subject selection by seeded shuffle
  (`random.Random(20260715)`); qids encode the original per-subject index (stable across
  shuffles). Built once to `mmlu_subset.jsonl`, reused by every arm → identical paired
  items. Gold-label balance across A/B/C/D: 225/253/268/254 (no degenerate skew).
- **Arms:** one process per arm (base, or base+LoRA via `PeftModel`, or base+sys-prompt for
  P), reusing `training/common.py::load_model/load_tokenizer` (local Llama-3.2-3B at
  `/workspace/models/llama-3.2-3b-instruct`). Resumable: existing qids skipped.
- **Seeds logged:** subset-selection & bootstrap seed 20260715. Scoring is seedless
  (deterministic). Base sanity: Arm 0 = 52.4% — a sound 0-shot chat-MMLU figure for this
  model (0-shot runs below Meta's 5-shot ~63%), well above the 25% chance floor.

## 6. Artifacts

- **Volume (gitignored):** `results/eval_t3/{mmlu_subset.jsonl, 0,P,SA,SB,RA,RB}.jsonl`
  (per-question: gold, pred, correct, 4 letter logprobs), `*_summary.json` (per-arm +
  per-subject accuracy), `delta_ci.json`, `verdict.json`, `run.log`.
- **Tracked:** `eval/tier3_mmlu.py` (scorer), `eval/tier3_aggregate.py` (verdict),
  `eval/run_tier3.sh` (sequencer), this record.
- **Reproduce:** `bash experiments/T01/eval/run_tier3.sh` (smoke → 6 arms → verdict).
