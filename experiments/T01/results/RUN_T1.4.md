# T1.4: Tier-1 holdout eval: FINAL RESULT

Full 6-arm holdout decode + grade + pre-registered analysis (PREREG §§3, 5-8),
plus the truncation sensitivity analysis that closes the one open confound.
**This record is final: T1.3 training and T1.4 evaluation are complete; no
further runs are planned for this tier.** Raw artifacts live on the `/workspace`
volume (`results/eval_t1_4/`: decodes, `criteria.jsonl`, `decode_meta.jsonl`,
`analysis.json`, `truncation_sensitivity.json`, `run.log`), gitignored by
design; this tracked record is the committed provenance.

- **Eval run:** 2026-07-19 22:21 UTC → 2026-07-20 01:12 UTC (~2h50m; single unattended
  run, completed in background across an operator disconnect). Sensitivity analysis:
  2026-07-20.
- **Pipeline:** `eval/run_tier1.sh` → smoke (arm 0, 5 prompts) → `eval/tier1_decode.py`
  per arm in order 0, P, SA, SB, RA, RB → `eval/grade_tier1.py` → `analysis/tier1_analysis.py`
  → `analysis/tier1_truncation_sensitivity.py`. No errors/tracebacks in `run.log`;
  all phase banners present.
- **Regime (frozen):** 400 holdout prompts (200/cause), k=3 decodes/prompt,
  temp 0.6 / top_p 1.0 (passed explicitly), seeds 20260715/16/17 by decode index,
  max_new_tokens 2048, local HF generate. 7,200 decodes total; `decode_meta.jsonl` = 7,200 rows.
- **Grading:** answer = `extract_final(completion)`; no marker → whole completion
  (the path that actually applies to arm 0 and all four trained arms; see §4).
- **Base sanity:** arm-0 criterion pass rate 61.4% (coverage) / 59.7% (precision), inside
  the T1.2 30 to 70 difficulty band. Denominators: 503 failed criteria (coverage),
  284 (precision).

## 1. Headline: H1 **NOT CONFIRMED**; interaction significantly *negative*

Interaction = [Rec(RB,B) − Rec(SB,B)] − [Rec(RA,A) − Rec(SA,A)] = **−0.3391**,
95% cluster-bootstrap CI **[−0.4086, −0.2687]** (10k resamples, seed 20260715).
The pre-registered rule (confirmed iff CI excludes 0 AND interaction > 0) →
`confirmed: false`. The CI lies entirely below zero: **GRPO's advantage over SFT
is larger for coverage than for precision; the opposite of the pre-registered
direction.** The effect is driven by Rec(RA,coverage):

| Rec | coverage | precision |
|-----|----------|-----------|
| 0   | 0.000    | 0.000     |
| P   | 0.179    | 0.268     |
| SA  | 0.284    | 0.194     |
| SB  | 0.270    | 0.190     |
| RA  | **0.740**| 0.218     |
| RB  | 0.181    | **0.306** |

This is the final T1.4 result. The truncation confound flagged at first read of
`analysis.json` (SA/coverage hit the 2048-token cap on 16.5% of decodes: 99/600,
81/200 prompts, ~10× any other cell, and Rec(SA,cov) enters H1 with a negative
sign, so truncation-induced SA failures push the interaction *toward* −0.34) was
investigated in full and does **not** explain the finding (§2).

## 2. Truncation sensitivity (V0-V3): conclusive; finding robust

`analysis/tier1_truncation_sensitivity.py` → `results/eval_t1_4/truncation_sensitivity.json`.
Same estimand code imported from `tier1_analysis.py`; same 10k bootstrap/seed;
treatments applied symmetrically to all arms (capped decodes elsewhere: SB 8 cov / 9 prec,
RA 2 cov, SA 1 prec, P 1 prec).

**What the capped decodes are:** degenerate repetition/enumeration loops (e.g. repeated
"onboarding … onboarding" n-grams; numbered lists running past item 100), not long good
answers. An SA (SFT·coverage) generation pathology at the eval regime; the eval cap (2048)
exceeds the training completion budget (1536), so this is not a harness clipping artifact.

| Variant | Treatment of capped decodes | Rec(SA,cov) | Rec(RA,cov) | Rec(SB,prec) | Rec(RB,prec) | Interaction | 95% CI |
|---------|-----------------------------|-------------|-------------|--------------|--------------|-------------|--------|
| V0 | as graded (reproduces `analysis.json` exactly) | 0.2843 | 0.7396 | 0.1901 | 0.3063 | −0.3391 | [−0.4086, −0.2687] |
| V1 | dropped from majority vote | 0.2744 | 0.7396 | 0.1866 | 0.3063 | −0.3455 | [−0.4151, −0.2754] |
| V2 | affected prompts excluded (81 cov, 9 prec) | 0.3277 | 0.7399 | 0.1868 | 0.3077 | −0.2913 | [−0.3719, −0.2108] |
| V3 | counted as PASS on every criterion (upper bound) | 0.4254 | 0.7396 | 0.1901 | 0.3063 | **−0.1979** | **[−0.2714, −0.1244]** |

Every variant leaves the interaction negative with the CI excluding zero.
**V3 is the decisive robustness check:** it grants SA a pass on all 489 criterion
checks of its capped decodes; the maximally generous treatment, physically
impossible to exceed; and the interaction is *still* −0.1979 with CI
[−0.2714, −0.1244], entirely below zero. Even if every truncated SA decode had
been a perfect answer, H1's sign and significance would not flip. Rec(SA,cov) is
bracketed [0.274, 0.425] under any treatment vs RA's 0.740: truncation cannot
explain the RA−SA coverage gap. −0.3391 (V0) stands as the headline per prereg;
V1-V3 are robustness.
(V1 note: COV-H-020 has zero valid SA decodes and counts as non-recovered there,
1 prompt, immaterial.)

## 3. end_phrase truncation effect: narrow, real, not decisive

The one criterion type where the cap directly causes failure: SA/coverage
end_phrase fail rate is **28.6% uncapped (134/469) → 100% capped (98/98)**, a
truncated decode by construction cannot end on the required phrase. This is a
real, mechanically-attributable artifact, and it is why the confound had to be
run down. But it is narrow: required_sections shows a smaller gap (61% → 83%)
and keyword_include none (44% → 47%), and 55% of capped-decode criterion
failures (271/489) sit on truncation-**insensitive** types (casing 98% → 100%,
start_phrase 92% → 98%, no_commas, keyword_exclude) that the cap cannot cause,
SA fails those near-totally, capped or not. Most capped-decode failures were not
caused by the cap; the V0-V3 bracket (§2) bounds the total possible contribution
and it is not decisive.

## 4. Marker forensics: SA's `===FINAL===` absence is general behavior, not truncation

Separate observation, independent of §2-3: SA emits the full `===FINAL===`
marker in **0/1100 uncapped decodes and 0/100 capped decodes**; its near-total
non-use of the scaffold marker is a general trained behavior, not something
truncation cut off. No completion ends mid-marker (fragment scan: 0 in both
strata; loose "FINAL" substring: 26/1100 uncapped, 6/100 capped, never the full
marker). RA is identical in kind: 0/1198 uncapped, 0/2 capped. All four trained
arms emit the marker at rate 0, so whole-text fallback grading applies uniformly
and matches training-time behavior; arm P (prompted scaffold) emits it 10.3%
(coverage) / 88.8% (precision). RA also has no milder version of SA's length
pathology: 2/600 capped (isolated, 2 prompts), gen-token p99 = 635 vs
SA/coverage p90 = 2048 (bimodal: median 521, then a mass at the cap).

## 5. H2 (exploratory): null across all variants

SFT data-targeting DiD = [Rec(SA,A)−Rec(SB,A)] − [Rec(SA,B)−Rec(SB,B)]:
baseline (V0) 0.0104, 95% CI [−0.0567, 0.0762]; null. Point estimates under
the sensitivity variants: −0.0011 (V1), 0.0566 (V2), 0.1385 (V3). The confound
biases H2 *downward* (SA enters H2 positively), so the V3 value is a
physically-unattainable upper bound, not evidence of an effect; no variant CI
was computed because H2 is exploratory and no variant produces a confirmable
signal. H2 remains null.

## Artifacts

- Volume: `results/eval_t1_4/{analysis.json, truncation_sensitivity.json, criteria.jsonl,`
  `decode_meta.jsonl, run.log, decodes/*.jsonl, decodes/*_summary.json}` (+ smoke_* files).
- Tracked: `eval/tier1_decode.py`, `eval/grade_tier1.py`, `eval/run_tier1.sh`,
  `analysis/tier1_analysis.py`, `analysis/tier1_truncation_sensitivity.py`, this record.

No config deviation from PREREG §5/§7 regime. The one PREREG deviation (local HF generate
in place of "via vLLM") was pre-declared at T1.4 launch in `tier1_decode.py`'s header.
