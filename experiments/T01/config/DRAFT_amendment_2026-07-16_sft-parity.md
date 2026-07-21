# DRAFT amendment: for reviewer approval, NOT yet committed to PREREG.md

> Staging file. Do not append to `PREREG.md` until the wording is approved at the
> T1.3 STOP. Verbatim candidate entry follows.

---

> **Superseded in part (2026-07-16, same day) by the coverage recalibration.** The
> coverage difficulty recal regenerated the coverage SFT pool, dropping coverage
> accepted **204 → 123** and therefore `target_n` from min(204,270)=204 to
> **min(123,270)=123 per cause**. This entry records the *original* T1.2 down-sample
> decision and its regime-independent mechanism; the counts below (204 / 68% coverage
> yield) are the **pre-recal** state. Operative manifests now: SA coverage 123/123,
> SB precision 123/270 (seed 20260715). See
> `DRAFT_amendment_2026-07-16_coverage-recal.md`.

### 2026-07-16: SFT training-set down-sample to cross-cause parity

**Decision, and when it was made.** Before any real-arm training (SA/SB/RA/RB)
had been run, the SFT accepted-pair sets are down-sampled to equal size across
causes. The decision is based **solely on datagen process metrics** (the teacher
acceptance yield measured at T1.2) and **no outcome/eval data was consulted**
(no arm has been scored; the GRPO LR probe informs only training-method
hyperparameters, not this decision).

**What changed.** §5 specifies "300 filtered pairs" of SFT data per cause, but the
teacher-acceptance yield is cause-dependent: coverage kept **204/300 (68%)**,
precision **270/300 (90%)** (datagen holds 300 prompts fixed per cause and keeps
one 100%-verifier-pass teacher answer per prompt when found within ≤4 attempts).
Training SA on 204 pairs and SB on 270 would confound the method×cause interaction
with an accepted-count asymmetry. Both SFT cells are therefore down-sampled to
**target_n = min(204, 270) = 204** accepted pairs. The effective SFT n changes
from "≤300 filtered pairs" to **target_n = 204 per cause**.

**Mechanics (symmetric, deterministic).** Each SFT cell selects target_n ids from
its **own** accepted set: sort by prompt_id, then `random.Random(20260715).sample`
(repo seed convention). Coverage is the min-count cell, so its selection is the
identity (204 of 204). Precision selects 204 of 270. Selections are materialized
as ids-only manifests under `data/sft_manifests/{SA,SB}.json` (tracked; never any
response text). Epochs stay at **2** (§5); reduced n is **not** compensated with
extra epochs/steps, which would re-introduce the asymmetry as a tuning degree of
freedom.

**What is explicitly unchanged.** GRPO prompt pools (300/cause), all holdout data
(`data/holdout/`, 200/cause), the 2-epoch schedule, the primary estimand and
hypotheses (§§3-4), sample sizes for eval (§5), seeds (§7), and the statistical
analysis (§6). Within-row data-fairness is preserved: both methods in a row still
draw from the same cause's data; only the SFT cell size is equalized across rows.

**Provenance.** Manifests: `data/sft_manifests/SA.json` (target_n 204, seed
20260715, source yield 68.0%), `data/sft_manifests/SB.json` (target_n 204, seed
20260715, source yield 90.0%) (pre-recal; current manifest files are SA 123/123, SB
123/270). Generator + determinism/parity/isolation tests:
`datagen/downsample_sft.py`, `datagen/test_downsample_sft.py`.
