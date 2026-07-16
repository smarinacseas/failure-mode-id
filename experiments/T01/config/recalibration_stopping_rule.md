# Coverage recalibration — pre-committed stopping rule (2026-07-16)

Stated BEFORE the acceptance measurement of any version is read, to remove
researcher degrees of freedom from the difficulty recalibration. (Retroactive
disclosure: v1 and v2 were developed/measured at the *training-rollout* regime
temp 0.9 / k=4 as an exploratory iteration signal — logged below — before this
rule was written; the ACCEPTANCE decision uses only the eval-regime protocol.)

## Acceptance regime (the band is defined here)
- **temp 0.6, k=3 decodes**, N=60 freshly composed coverage prompts, seed 20260715.
- Matches PREREG §5 Tier-1 eval decode (temp 0.6, k=3) and the original T1.2
  `calibrate.py` (temp 0.6). Difficulty measured at the training-rollout regime
  (temp 0.9) is NOT the acceptance metric — it can understate pass rate.

## Rule
1. **Cap: 3 generator versions.**
2. **Accept the FIRST version whose mean criterion-pass ∈ [30%, 70%] at the
   acceptance regime.** No selection within-band by closeness to any preferred
   point (no "pick the one nearest 50%").
3. **One acceptance measurement per version** — no re-rolling a version's
   measurement to obtain a nicer number.
4. If none of the 3 versions lands in-band, **STOP and escalate to the reviewer**
   (do not keep iterating).
5. **Every version and its regime is logged in the dated PREREG amendment,**
   regardless of which is used.

## Version log
- **v1** (harden count-types: keyword_include 4-6→7-10, required_sections 3-5→6-8,
  keyword_exclude 1-2→3-5; k 5-6→6-8): exploratory temp-0.9/k=4 mean = **74.4%**.
- **v2** (v1 + retire no_placeholders/title from sampling, add start_phrase; k 6-7):
  exploratory temp-0.9/k=4 mean = **59.8%** (per-type: start_phrase 8.6, keyword_include
  42.1, end_phrase 45.5, casing 63.9, keyword_exclude 73.2, required_sections 92.5,
  no_commas 95.0); **acceptance temp-0.6/k=3 mean = 61.0% → IN BAND → ACCEPTED**
  (per-type: start_phrase 7.5, keyword_include 38.6, end_phrase 47.3, keyword_exclude
  67.3, casing 80.9, no_commas 93.3, required_sections 95.0; 6.5 constraints/prompt).
- v3: not run — v2 is the first version measured at the acceptance regime and is
  in-band, so the rule accepts it (no within-band tuning, no further versions).

## Outcome
**v2 accepted** at 61.0% eval-regime criterion-pass (was 86.4%/~75% pre-recal).
9-point margin below the 70% ceiling. Fresh-composed sample; the regenerated
train pool is re-measured at the same regime to confirm before teacher_gen.
