# Does failure type determine training method? SFT vs GRPO across coverage and precision failures (T01 · Tier-1 Final Report)

> **STATUS: FINAL for Tier-1 (2026-07-20).** Covers T1.3 (training, all four arms)
> and T1.4 (Tier-1 holdout eval + truncation sensitivity analysis), both complete.
> The **Tier-3 general-capability guard (H3) was run 2026-07-20 and FAILED by the
> point-estimate rule** (Arm SA −3.2 pp vs base, a *marginal* fail; its CI [−5.3,
> −1.1] spans the −3 line; [`RUN_H3.md`](./RUN_H3.md)), incorporated in §6
> (post-finalization additions), §7, and §8. **Tier-2** (CC-75 transfer) has now
> also run (2026-07-21; [`RUN_TIER2.md`](./RUN_TIER2.md)): the coverage effect does
> not transfer to the naturalistic corpus; see §6 addition (h). Every number
> in this report is taken from committed run records or volume artifacts named in
> §9; the pre-registration ([`../PREREG.md`](../PREREG.md)) remains the authority
> on the design. Detailed run records: [`RUN_SA.md`](./RUN_SA.md),
> [`RUN_SB.md`](./RUN_SB.md), [`RUN_RA.md`](./RUN_RA.md), [`RUN_RB.md`](./RUN_RB.md),
> [`RUN_T1.4.md`](./RUN_T1.4.md), [`RUN_H3.md`](./RUN_H3.md).

---

## Abstract

We test whether the *type* of failure a language model exhibits should determine
the *method* used to train it out of that failure. From a prior failure census
(E08) of `meta-llama/Llama-3.2-3B-Instruct` on constraint-following tasks, the two
dominant root causes were bound as **coverage** failures (`constraint_unaddressed`,
28.8%, the model never engages a stated requirement) and **precision** failures
(`execution_slip`, 36.4%, the model engages a requirement but executes it
incorrectly). We pre-registered a 2×2 factorial: training method {SFT, GRPO} ×
training-data cause {coverage, precision}, with one confirmatory hypothesis:
**H1**, that GRPO's advantage over SFT is larger for precision than for coverage
(Interaction > 0), on the theory that imitation cheaply installs a coverage
*template* while precision requires on-policy practice against a verifier. All four
LoRA arms trained to their pre-registered learning bars. On the 400-prompt reserved
holdout, **H1 was not confirmed: the interaction is significantly *negative*:
−0.3391, 95% cluster-bootstrap CI [−0.4086, −0.2687]**; GRPO's advantage over SFT
is far *larger* for coverage than for precision, the opposite of the pre-registered
direction. The result is driven by the GRPO·coverage arm recovering 74.0% of
base-model failures while the SFT·coverage arm recovered 28.4% *and* broke 55.2% of
the criteria the base model passed. A truncation confound affecting 16.5% of the
SFT·coverage arm's decodes was investigated with a four-variant sensitivity
analysis and ruled out: even under the maximally generous treatment the interaction
remains −0.1979, CI [−0.2714, −0.1244]. The exploratory data-targeting hypothesis
(H2) is null. We report the reversal honestly and offer three graded post-hoc
mechanisms, the best-evidenced being a train/eval grading asymmetry specific to the
SFT pipeline: trained arms never adopt the teacher's `===FINAL===` answer marker
(0/4800 decodes), so their imitated scaffold prose (exempt from constraint
checking during teacher-data acceptance) is graded at eval, where it violates
formatting constraints wholesale. A post-hoc general-capability guard (H3, MMLU)
points at the same fault line from the other side: **both SFT arms regress ≈3 points
of unrelated general capability** (SA −3.2, a *marginal* miss of the guard's 3-point
rule, its CI [−5.3, −1.1] spans the line; SB −3.0) **while both GRPO arms retain it**
(RA −0.1, RB +0.9). That is the *same* method split as the interaction, and it is
plausibly one story rather than two: at this scale the SFT recipe broadly overwrites
the base model: breaking about as much as it fixes on-task (§5.6) and regressing
off-task, while GRPO's KL-tethered on-policy updates change little beyond the target.
Read that way the negative interaction is as much "SFT is unexpectedly destructive
here" as "GRPO is unexpectedly good at coverage." A transfer test on the naturalistic
CC-75 corpus (Tier-2) sharpens which half generalizes. The coverage-specific GRPO
advantage does **not**: RA's 74% synthetic-coverage recovery falls to 7.6%, and the
interaction vanishes (≈ −0.005), locating T1.4's headline effect in T01's *synthetic*
constraint distribution rather than a generalizing capability. The SFT-more-destructive
pattern, by contrast, transfers cleanly and is now triangulated across three
distributions (synthetic holdout breakage, MMLU general-capability regression, and
naturalistic CC-75 breakage). **The transfer-robust takeaway is therefore the
cautionary one: scaffold-distilled SFT at this scale degraded the model on every
distribution tested, while GRPO's larger gains were concentrated where a clean verifier
defined the target and did not evidently generalize past it.**

---

## 1 · Introduction

**The question.** Post-training fixes are usually chosen by availability (SFT if
you have demonstrations, RL if you have a reward) rather than by *diagnosis* of
what is actually failing. The T-series asks whether diagnosis should drive the
choice: if a model's failures are of mechanistically different kinds, do different
training methods repair different kinds preferentially? T01 is the first cell of
that program: one model, one task family, two failure causes, two methods.

**Where the causes come from.** E08 ran a blinded root-cause census over 882
consensus-FAIL criteria (3-judge panel verdicts, `claude-fable-5→opus-4-8`
classifier chain) for Llama-3.2-3B-Instruct on the CC-75 constraint-following
battery (43.4% base criterion pass, far from ceiling). The top two causes were
high-prevalence, jointly 65.2% of failures, and mechanistically distinct:

- **Coverage** (`CAUSE_A`, `constraint_unaddressed`, 254/882 = 28.8%): the model
  never engages a stated requirement at all, a failure of *noticing/tracking*.
- **Precision** (`CAUSE_B`, `execution_slip`, 321/882 = 36.4%): the model engages
  the requirement but executes it wrong, a failure of *exactness*.

The distinction matters because the two have different plausible repair
mechanisms. Coverage looks like a missing behavioral template
(enumerate → draft → verify) that imitation could install cheaply. Precision is
correctness *under the model's own output distribution*: the kind of thing
on-policy practice with verifier feedback trains directly, and off-policy
imitation of someone else's correct outputs may not.

**Pre-registration discipline.** The design, estimand, sample sizes, seeds,
decision rule, and outcome interpretations were frozen in
[`PREREG.md`](../PREREG.md) at Gate E→T (2026-07-15), before any training. The
pre-registration is append-only; every mid-experiment change (a coverage
difficulty recalibration, an SFT parity down-sample, environment freezes) is a
dated amendment in its log, and none altered the estimand, hypotheses, seeds, or
analysis. There is **exactly one confirmatory test** in the whole experiment (H1);
everything else is exploratory or descriptive by construction.

## 2 · Hypotheses

- **H1 (primary, confirmatory; the only confirmatory test).** GRPO's advantage
  over SFT is larger for precision than for coverage: `Interaction > 0` (see §3.2
  for the estimand). Confirmed iff the 95% CI excludes 0 **and** the point
  estimate is positive. *Rationale:* the template-vs-practice asymmetry above.
- **H2 (secondary, exploratory).** Within SFT, cause-matched data helps its own
  cause more than cross-cause data does (a data-targeting
  difference-in-differences). Reported with a CI, never used for confirmation.
- **H3 (regression guard, not a hypothesis).** No trained arm falls more than 3
  points below the untrained base on a general-capability battery (Tier-3).
  **Status: evaluated 2026-07-20: FAILED (point-estimate rule; marginal).** Arm SA
  is 3.2 pp below base on MMLU (0-shot, N=1000), past the 3-point line, though its
  95% CI [−5.3, −1.1] includes the threshold; the other three trained arms pass. Full
  result and the method-wise retention pattern in §6 (post-finalization additions);
  raw scores in [`RUN_H3.md`](./RUN_H3.md).

The pre-registration also committed interpretations for every outcome *before any
data existed* (PREREG §8), including the one that occurred: *"Interaction < 0, CI
excludes 0 → opposite direction; report honestly, speculate cautiously."* This
report follows that instruction: §6 separates the finding (robust) from the
mechanisms (post-hoc, graded by evidence).

## 3 · Methods

### 3.1 Subject and design

Subject: `meta-llama/Llama-3.2-3B-Instruct` (a heavily post-trained 3B model,
see §7). Design: a 2 (method) × 2 (data-cause) factorial of LoRA adapters, plus
two untrained reference arms (ceiling: 6 arms):

```
                        TRAINED WITH →
                  ┌──────────────┬──────────────┐
                  │  SFT (copy)  │  GRPO (RL)   │
 ┌────────────────┼──────────────┼──────────────┤
 │ coverage data  │   Arm SA     │   Arm RA     │
 ├────────────────┼──────────────┼──────────────┤
 │ precision data │   Arm SB     │   Arm RB     │
 └────────────────┴──────────────┴──────────────┘
 + Arm 0 : base model, untrained            (reference; defines the failure set)
 + Arm P : base + enumerate-then-verify system prompt   (the $0 baseline)
```

*Why these two methods:* SFT here is teacher distillation: imitate a strong
model's verified-correct, scaffolded answers (off-policy). GRPO (Group Relative
Policy Optimization) is on-policy RL: the model samples k=6 answers per prompt at
temperature 0.9, each is scored by the programmatic verifiers, and the policy is
updated toward the answers that scored above their group's mean (advantages are
computed *relative to the group*, which removes the need for a learned value
model), with a KL penalty (β=0.04) tethering it to the base policy. The contrast
is exactly the off-policy-imitation vs on-policy-practice distinction that H1's
rationale turns on.

### 3.2 Estimand: recovery, and why an interaction

**Recovery.** For each cause pool, Arm 0's failed-criterion set is fixed first:
a criterion counts as a base failure if Arm 0 fails it on a majority of its k=3
decodes. `Rec(arm, cause)` is the fraction of that base-failed set the arm now
passes (again by majority vote over the arm's k=3 decodes). Recovery asks the
diagnostic question directly (*of the things the base model got wrong, how many
does this intervention fix?*) rather than mixing repair with things the base
already did well. (Its blind spot, criteria the base passed and the arm breaks, is
reported descriptively in §5.6.)

**The interaction.**

```
Interaction = [Rec(RB, precision) − Rec(SB, precision)]
            − [Rec(RA, coverage)  − Rec(SA, coverage)]
```

i.e. (GRPO−SFT advantage on precision) minus (GRPO−SFT advantage on coverage).
H1 predicts it positive.

*Why an interaction rather than main effects:* GRPO consumes roughly 10× the
FLOPs of SFT here, so "GRPO beats SFT" main effects are compute-confounded and
reported as descriptive only. The interaction is structurally fairer: the compute
surplus applies to both causes equally and cancels in the difference of
differences; RL-family quirks cancel across the two RL cells (family-fair); and
both methods in a row consume the same 300-prompt pool (data-fair).

### 3.3 Task and data construction

**Task family.** Programmatically composed constrained-writing prompts, each
carrying mechanically verifiable constraints (`check(response, spec) →
pass/fail`). Coverage prompts carry 6 to 7 heterogeneous, buried constraints
(measured: 1,304 criteria over 200 holdout prompts ≈ 6.5/prompt) of types
{start_phrase, end_phrase, keyword_include, keyword_exclude, casing, no_commas,
required_sections}; precision prompts carry 2 to 4 exactness-demanding constraints
(measured: 705 ≈ 3.5/prompt). The A/B contrast is deliberate: many-buried vs
few-exact is what makes the causes sortable at a glance, without which the
interaction would collapse toward 0 by construction.

**Pools and hygiene.** Per cause: 300 training prompts + 200 holdout, split with
seed 20260715. The holdout lives under `data/holdout/` and training code is
test-enforced to never read that path. All generated prompts passed a 13-gram
contamination screen against the T01 holdout, CC-75, and IFEval originals (0 hits
on the final pools).

**Difficulty calibration (and the one mid-stream recalibration).** Pools target a
30 to 70% base-model criterion-pass band, hard enough to leave repair headroom,
easy enough to be learnable. Coverage initially landed *out of band* (86.4% base
pass at the rollout regime), tripping a pre-committed STOP rule before any real
training. Under a stopping rule committed before any acceptance number was read
(first version to land in band is accepted; cap 3 versions), the coverage
generators were hardened: more keywords/sections to include, near-free constraint
types retired from sampling, a strict `start_phrase` type added (base passes
~7.5%); and v2 was accepted at 61.0% base pass. Precision was in band (61.1%)
and untouched. Consequence for interpretation: post-recalibration coverage
constraints are *more exacting* than the E08 originals; see §6, mechanism (c).

**SFT data (teacher distillation).** A strong open instruct model (via
OpenRouter) answered each training prompt with a mandated scaffold:
coverage: INVENTORY → DRAFT → VERIFY; precision: WORKED COMPUTATION → DRAFT →
RECONCILE, ending with an `===FINAL===` marker followed by the final answer.
Acceptance graded **only the post-marker answer** with the same verifiers
(≤4 attempts/prompt; keep only 100%-pass responses). The **full completion**
(scaffold + marker + answer) is the SFT target. Yields were cause-asymmetric:
coverage 123/300 (41.0%), precision 270/300 (90%), so both SFT cells were
down-sampled to **target_n = 123** accepted pairs (deterministic, seed 20260715,
manifests tracked) before any real training, to keep the accepted-count asymmetry
out of the interaction. This grading detail (scaffold prose exempt at
acceptance, answer-only checked) becomes load-bearing in §6.

### 3.4 Training protocol (T1.3)

All four arms share an identical LoRA (r=16, α=32, dropout 0.05, all
attention+MLP linears) on a single A100-80GB; frozen config in
[`../config/t1_3_frozen.md`](../config/t1_3_frozen.md); one hyperparameter probe
per *method* (never per-arm) before freezing.

| | SFT (SA, SB) | GRPO (RA, RB) |
|---|---|---|
| data | 123 accepted pairs/cause, completion-only loss (prompt masked) | 300 prompts/cause + per-prompt `specs` |
| schedule | 2 epochs (16 steps), LR 1e-4 cosine | 2 epochs (300 steps), LR 7.5e-6 constant (probe-selected from the pre-registered 5e-6-1e-5 range) |
| sampling |  | k=6 rollouts/prompt, temp 0.9, `generate()` backend (no vLLM, GT0 env freeze, PREREG amendment (d)) |
| reward |  | verifier pass-fraction on `extract_final(completion)` − malformed penalty, length-capped (M=2800 chars); β=0.04 |
| wall-clock | 55.5 s / 29.8 s | ~3.4 h (RA, phased) / ~1.7 h (RB) |

*Reward wiring note:* GRPO's reward graded `extract_final(completion)`: the
text after a `===FINAL===` marker if present, otherwise the **whole completion**.
Rollouts essentially never contained the marker, so GRPO in practice optimized
whole-text constraint satisfaction. The Tier-1 eval grades the same way, which
makes GRPO train/eval-*consistent*, a point that returns in §6.

**Pre-committed monitoring gates.** GRPO carried two: an RA-specific step-50
windowed-trend check (added by amendment after a probe flagged "no reward learning
at 50-step scale on hardened coverage") and the pre-registered §9 stall
kill-switch at ~150 steps (reward flat → both RL cells demoted to RFT).

### 3.5 Evaluation protocol (T1.4)

Frozen regime: 400 holdout prompts (200/cause) × k=3 decodes × 6 arms = 7,200
decodes; temperature 0.6, top_p 1.0, seeds 20260715/16/17 by decode index;
max_new_tokens 2048; local HF `generate()` (the one pre-declared deviation from
PREREG's "via vLLM" (declared at T1.4 launch, before results). Grading: answer =
`extract_final(completion)`: no marker → whole completion; per-criterion records
`{arm, prompt_id, decode_idx, cause_pool, criterion_id, pass}`. Single unattended
2h50m run (2026-07-19 22:21 → 07-20 01:12 UTC), no errors in `run.log`.

**Base sanity:** Arm-0 criterion pass 61.4% (coverage) / 59.7% (precision),
inside the calibration band; base-failed criterion sets: 503 (coverage) / 284
(precision). These two sets are the denominators for every recovery number below.

### 3.6 Statistical analysis (fixed in advance)

Uncertainty is clustered at the **prompt** level: criteria within a prompt share
failures (one bad decode fails several criteria at once), so treating criteria as
independent would fabricate significance. The 95% CI comes from a prompt-level
cluster bootstrap: resample the 200 prompts with replacement *within each cause
pool*, recompute all four Rec cells and the interaction, 10,000 times (seed
20260715), take the 2.5th/97.5th percentiles. Decision rule: H1 confirmed iff the
CI excludes 0 and the interaction is positive. Power context: ~3 to 4-point
detection floor for the interaction at this n.

## 4 · Training results (T1.3)

All four arms met the pre-registered GT3 learning bar; no kill-switch fired.

| Arm | Method · cause | n | Steps | Learning signal | GT3 |
|-----|----------------|---|-------|-----------------|-----|
| SA | SFT · coverage | 123/123 | 16 | loss 1.935 → 1.425 (Δ −0.510, monotone) | ✅ |
| SB | SFT · precision | 123/270 | 16 | loss 2.516 → 1.993 (Δ −0.523) | ✅ |
| RA | GRPO · coverage | 300 prompts | 300 | reward 0.494 → 0.800; OLS +0.00123/step, **t = 20.24** | ✅ |
| RB | GRPO · precision | 300 prompts | 300 | reward 0.511 → 0.747; OLS +0.00026/step, t = 2.79 | ✅ |

**The RA gate story** (full detail in [`RUN_RA.md`](./RUN_RA.md)): at the
pre-committed step-50 check RA was genuinely flat (OLS t = −0.21) and was PAUSED
per the rule. The operator resumed it to let the pre-registered ~150-step §9 gate
adjudicate; by step 150 reward was strongly trending (t = 6.29), and the full
300-step run finished decisively (t = 20.24, no length runaway, cap-hit 0%,
KL ~0.016). The step-50 flat was a window-length artifact, but it means the
*earliest* training signal pointed the opposite way from the eventual result.
Note the direction: the training-reward trend was already *stronger for coverage
than precision* under identical GRPO (t = 20.24 vs 2.79), flagged in
[`RUN_RB.md`](./RUN_RB.md) at the time as a hint of an interaction (in the
direction opposite H1) to be tested on holdout, not concluded from training
curves. A provenance note: RA's step-50/150 decision-point *checkpoints* were
later rotated out by `save_total_limit=3`; the decisions' evidentiary basis (the
logged trajectories) is fully preserved.

## 5 · Evaluation results (T1.4)

### 5.1 Primary result: H1 not confirmed; interaction significantly negative

Recovery matrix (fraction of the base-failed criterion set now passed, majority
vote over k=3):

| Rec | coverage | precision |
|-----|----------|-----------|
| 0   | 0.000    | 0.000     |
| P   | 0.179    | 0.268     |
| SA  | 0.284    | 0.194     |
| SB  | 0.270    | 0.190     |
| RA  | **0.740**| 0.218     |
| RB  | 0.181    | **0.306** |

- GRPO−SFT advantage on **precision**: Rec(RB) − Rec(SB) = 0.306 − 0.190 = **+0.116**
- GRPO−SFT advantage on **coverage**: Rec(RA) − Rec(SA) = 0.740 − 0.284 = **+0.455**
- **Interaction = 0.116 − 0.455 = −0.3391**, 95% cluster-bootstrap CI
  **[−0.4086, −0.2687]** (10k, seed 20260715).

The pre-registered rule returns **`confirmed: false`**, and the CI lies entirely
below zero: this is not a null but a significant *reversal*. GRPO does beat SFT
on precision (consistent with H1's mechanism in isolation), but it beats SFT on
coverage by four times as much, driven by Rec(RA, coverage) = 0.740, the largest
cell in the table by a wide margin.

### 5.2 Robustness: the truncation confound, investigated and ruled out

**Why this had to be run down.** The eval's health telemetry showed SA/coverage
hitting the 2048-token generation cap on 16.5% of decodes (99/600, spread over
81/200 prompts), ~10× any other cell (next: SB/precision 9, SB/coverage 8,
RA/coverage 2). Because Rec(SA, coverage) enters H1 with a negative sign,
truncation-induced SA failures push the interaction *toward* the observed −0.34.
Inspection showed the capped decodes are degenerate repetition/enumeration loops
(repeated n-grams, numbered lists past item 100), not long good answers: a
generation pathology of the SA arm, not harness clipping (the eval cap exceeds
the 1536-token training budget).

**Sensitivity design.** Rather than argue about what the truncated text "would
have been," recompute the full pre-registered estimand (same code path, same 10k
bootstrap and seed, treatments applied symmetrically to all arms) under four
treatments of capped decodes, spanning the entire range from most punitive to
physically-impossible-to-exceed generous:

| Variant | Treatment of capped decodes | Rec(SA,cov) | Rec(RA,cov) | Rec(SB,prec) | Rec(RB,prec) | Interaction | 95% CI |
|---------|-----------------------------|-------------|-------------|--------------|--------------|-------------|--------|
| V0 | as graded (reproduces `analysis.json` exactly) | 0.2843 | 0.7396 | 0.1901 | 0.3063 | −0.3391 | [−0.4086, −0.2687] |
| V1 | dropped from the majority vote | 0.2744 | 0.7396 | 0.1866 | 0.3063 | −0.3455 | [−0.4151, −0.2754] |
| V2 | affected prompts excluded (81 cov, 9 prec) | 0.3277 | 0.7399 | 0.1868 | 0.3077 | −0.2913 | [−0.3719, −0.2108] |
| V3 | counted as PASS on **every** criterion (upper bound) | 0.4254 | 0.7396 | 0.1901 | 0.3063 | **−0.1979** | **[−0.2714, −0.1244]** |

Every variant leaves the interaction negative with the CI excluding zero. **V3 is
the decisive check:** it grants SA a pass on all 489 criterion checks of its
capped decodes (as if every truncated decode had been a perfect answer) and the
interaction is still −0.1979 with a CI entirely below zero. Rec(SA, coverage) is
bracketed [0.274, 0.425] under any treatment against RA's 0.740: truncation
cannot explain the gap, hence cannot explain the sign or significance of H1.
Per pre-registration, V0 (−0.3391) stands as the headline; V1-V3 are robustness.

**The one real truncation artifact, bounded.** `end_phrase` is the canary
criterion (a truncated decode cannot end on the required phrase) and behaves
exactly as truncation predicts: SA/coverage fail rate 28.6% uncapped (134/469) →
100% capped (98/98). But the effect is narrow: `required_sections` shows a smaller
gap (61% → 83%), `keyword_include` none (44% → 47%), and 55% of capped-decode
criterion failures (271/489) sit on truncation-*insensitive* types (casing,
start_phrase, no_commas, keyword_exclude) that SA fails near-totally capped or
not. Real, mechanically attributable, and non-decisive; the V0-V3 bracket bounds
its total possible contribution.

### 5.3 Marker forensics: a scaffold-compliance property, not truncation

A separate observation that seeds the leading mechanism in §6: SA emits the full
`===FINAL===` marker in **0/1100 uncapped and 0/100 capped decodes**; its
absence is a general trained behavior, not something truncation cut off (no
completion ends mid-marker; loose "FINAL" substrings appear 26/1100 vs 6/100,
never the full marker). RA is identical in kind (0/1198, 0/2), as are SB (0/1183
uncapped, 0/17 capped) and RB (0/1200); **all four trained arms emit the marker at
rate 0** (0/4800 decodes total; the SB/RB counts were verified from the decode
artifacts on 2026-07-20), despite the marker appearing in every SFT training target.
Notably SB, though it never once emits the marker, writes the bare word "FINAL" most
often of any arm (loose substring 85/1183 uncapped = 7.2%). Whole-text fallback
grading therefore applies uniformly to the trained arms, matching their training-time
grading. Arm P, the
*prompted* scaffold, emits it at 10.3% (coverage) / 88.8% (precision). RA also
shows no milder version of SA's length pathology: median 331 / p90 477 generated
tokens vs SA/coverage's bimodal median 521 with p90 at the 2048 cap.

### 5.4 H2 (exploratory): null across all variants

Data-targeting DiD = [Rec(SA,cov) − Rec(SB,cov)] − [Rec(SA,prec) − Rec(SB,prec)]:
**0.0104**, 95% CI [−0.0567, 0.0762]: null. Under the sensitivity variants the
point estimate stays small: −0.0011 (V1), 0.0566 (V2), 0.1385 (V3, a
physically-unattainable upper bound, since the confound biases H2 downward). No
variant produces a confirmable signal; training SFT on a cause's own data did not
detectably help that cause more than the other cause's data did.

### 5.5 The $0 baseline (Arm P, descriptive)

The enumerate-then-verify system prompt recovers 0.179 (coverage) / 0.268
(precision); on **precision it beats both SFT arms** (0.194, 0.190) and lands
within 4 points of GRPO's 0.306, for zero training cost. On coverage it trails
everything. The pre-committed interpretation ("a $0 system prompt matched
fine-tuning: the baseline nobody runs") applies to the precision column
specifically, and sharpens the practical takeaway in §6.

### 5.6 Post-hoc descriptives: what recovery hides (breakage)

*(This subsection is computed post-hoc from `criteria.jsonl`
(`analysis/tier1_descriptives.py`) and is not pre-registered; decode-level and
criterion-level rates, no CIs.)*

Recovery only counts base-failures fixed. Its complement (**breakage**, the
fraction of base-*passed* criteria an arm now majority-fails) completes the
picture (base-passed sets: 801 coverage / 421 precision):

| Arm (on its trained cause) | Rec | Brk | Net criteria (fixed − broken) |
|----------------------------|-----|-----|-------------------------------|
| SA · coverage | 0.284 | **0.552** | **−299** |
| SB · precision | 0.190 | 0.287 | −67 |
| RA · coverage | 0.740 | 0.050 | **+332** |
| RB · precision | 0.306 | 0.083 | +52 |

Both GRPO arms are net-positive on their trained cause; **both SFT arms are
net-negative: they broke more than they fixed** (raw decode-level pass rates
tell the same story: SA/coverage 38.1% vs base 60.2%; SB/precision 50.0% vs
59.4%; RA/coverage 85.2%; RB/precision 66.1%). Per-type anatomy on the coverage
pool localizes SA's damage precisely:

| type (coverage pool) | base | SA | RA |
|----------------------|------|-----|-----|
| casing | 0.714 | **0.013** | 0.740 |
| no_commas | 0.928 | **0.417** | 0.967 |
| required_sections | 0.939 | **0.356** | 0.960 |
| start_phrase | 0.046 | 0.073 | **0.951** |
| end_phrase | 0.427 | 0.591 | 0.963 |
| keyword_include | 0.388 | 0.555 | 0.590 |
| keyword_exclude | 0.747 | 0.644 | 0.785 |

SA *improved* the content-insertion types (keywords, end_phrase): the imitation
did teach something, while collapsing the formatting-preservation types (casing
71% → 1%, no_commas, required_sections). RA improved every type, including
near-mastery of `start_phrase` (4.6% → 95.1%), the hardest type in the pool. RA's
gains are also not a verbosity artifact (the pre-committed check for
length-co-moving gains): its answers are *shorter* than base (mean 1551 vs 1613
chars; SA's are 3435, inflated by the loops).

## 6 · Discussion

**What the result establishes.** Within this experiment's scope, failure type
*does* modulate method choice (significantly and robustly) but in the direction
opposite the pre-registered prediction: on-policy verifier-driven training (GRPO)
out-repaired imitation (SFT) most where the failures were coverage-shaped. The
committed interpretation table binds us to report this as an opposite-direction
finding and speculate cautiously. Three candidate mechanisms, ordered by evidence:

**(a) A train/eval grading asymmetry punished SFT specifically**
*(best-evidenced).* Teacher data was accepted by grading only the post-marker
answer: the scaffold prose (INVENTORY checklists, VERIFY sections, full of
commas, capitals, and headers) was never subject to the constraints. The students
imitated the scaffold but never adopted the marker (0/4800 trained-arm decodes,
§5.3), so at eval their whole text is graded, scaffold included. The per-type
signature matches exactly: SA collapses precisely on whole-text formatting
constraints that scaffold prose violates (casing, no_commas, required_sections)
while improving content-insertion types (§5.6). GRPO, whose reward graded
whole-text all along, was train/eval-consistent; RA never suffers this. Under
this account, part of the reversal's *magnitude* is an artifact of the SFT
pipeline's scaffold design rather than of imitation per se; note it cannot
explain why SA also trails RA on marker-independent grounds (V3 caps SA at 0.425
vs RA's 0.740 even granting all its truncated decodes) and it leaves the
precision-side GRPO advantage (+0.116, where SB has the same scaffold handicap)
untouched.

**(b) Dense verifiable reward is GRPO's home turf.** Coverage prompts carry ~6.5
mechanically checkable constraints vs precision's ~3.5, so coverage rollouts get
a smoother, denser pass-fraction gradient (more distinguishable reward levels
per group of 6) and several coverage types are the kind of formulaic surface
rules RL discovers readily (always open with the exact phrase; never emit a
comma). RA's t = 20.24 vs RB's t = 2.79 training trend and its 95% start_phrase
mastery fit this. H1's rationale implicitly assumed "coverage = noticing," but a
verifier-definable coverage constraint is *also* a perfect reward signal: the
very property that made the cause measurable made it unusually RL-trainable.

**(c) Construct drift from the recalibration** *(least evidenced, honest
caveat).* The mid-stream difficulty recalibration hardened coverage with exacting
types like `start_phrase` (exact opening sentence, base 4.6%). Arguably these
demand execution-precision as much as requirement-noticing, blurring the very
archetype contrast the design needs (PREREG's "sortable at a glance"
requirement). If post-recal coverage is partly precision-in-disguise, some of the
coverage-side GRPO advantage is the H1 mechanism itself, relabeled. We cannot
quantify this from the data; it is a design lesson for T02.

**Practical readings, within scope.** (1) If failures are verifier-checkable
(whatever their nominal archetype), on-policy training against that verifier was
the only intervention here that repaired without collateral damage (both GRPO
arms net-positive; both SFT arms net-negative, §5.6). (2) The scaffold-distilled
SFT recipe, as implemented, is actively hazardous under whole-text grading; if
retried, either train marker emission to reliability or grade training data
whole-text for consistency. (3) On precision, a free system prompt captured most
of what SFT delivered (§5.5); try prompting before training. (4) H2's null says
cause-*targeting* of SFT data bought nothing detectable here; the method, not the
data's cause-label, carried the effect.

### Post-finalization additions (2026-07-20)

*Added the same day the report was finalized, after the H3 Tier-3 guard was run
([`RUN_H3.md`](./RUN_H3.md)) and in response to four reviewer questions on the
confound structure. These extend §6's mechanisms (a)-(c) and update the H3 status
throughout (§2, STATUS, §7, §8); they change no pre-registered number and no
headline finding.*

**(d) The two causes are not fully parallel constructions: "cause" is confounded
with constraint-type family and pool-authoring epoch.** Two facts, both verified
from artifacts. *Type families are disjoint:* the coverage pool draws its
constraints from {start_phrase, end_phrase, keyword_include, keyword_exclude,
casing, no_commas, required_sections} (surface/format rules) and the precision
pool from {arithmetic_result, item_count, word/sentence/paragraph_count, ordering,
exact_repetition, keyword/caps_frequency} (counting/arithmetic rules), with **no
type in common**. *Authoring epochs differ:* the coverage train+holdout were
**regenerated 2026-07-16** (git `6d67e4f`) when the recalibration added the strict
`start_phrase` type and hardened thresholds, while the precision pools were **last
touched 2026-07-15** (`23532c1`) and carried forward untouched from T1.2. So the
design's "method×cause" is inseparable from "method×constraint-type-family", which
is true *by construction*, since the A/B contrast of many-buried-surface vs
few-exact *requires* different type families, and now also from a
pool-authoring/calibration difference (coverage re-authored to a stricter bar a day
later, precision not). The interaction is therefore not a clean archetype effect; it
is, at minimum, archetype ⊕ type-family ⊕ authoring-epoch, and these are not
separable within T01. Mechanism (b) is the benign reading of the type-family
confound (surface constraints happen to be RL-friendly); this is its sharper, less
comfortable statement: that "coverage" and "precision" may name *constraint
families authored to different standards* as much as they name failure archetypes.
Both readings survive the data. A clean archetype test needs the *same*
constraint-type families instantiated under both causes, the central design lesson
for T02, extending §6(c).

**(e) Compute-fairness reconsidered: the ~10× surplus "cancels" only if its
marginal value is constant across causes.** The interaction's compute-fairness claim
(§3.2; PREREG §10.1, "the surplus cancels") rests on an unstated assumption: that an
extra unit of GRPO compute is worth the same on coverage as on precision, so it drops
out of the difference-of-differences. But the result *is* that GRPO's advantage is
not constant across causes, so that assumption is exactly what is in doubt. If
coverage's constraint-satisfaction is the more exploration-rewarding task (denser
verifiable reward (~6.5 vs ~3.5 criteria/prompt → more distinguishable group-relative
advantage levels per rollout batch) and formulaic surface rules a policy can *find*
by sampling (start_phrase 4.6% → 95%)) then the same FLOP surplus buys more on
coverage than on precision, and it does **not** cancel. Under that account some of
the coverage-side GRPO advantage is a compute effect riding along with the archetype,
not a pure method×archetype effect. This does not overturn the finding, but it means
"the interaction is immune to the compute confound" should be stated conditionally:
immune *if* the marginal value of compute is cause-independent, an assumption this
very result undercuts. Worth naming as an alternative explanation rather than resting
on "immune."

**(f) SB underperformed Arm P on precision: real training lost to a $0 prompt, and
we cannot fully say why.** Rec(SB, precision) = 0.190 < Rec(P, precision) = 0.268:
SFT on precision data recovered *less* than the enumerate-then-verify system prompt,
at real training cost against zero. The interaction-consistent reading is that SFT
genuinely struggles on precision: exactness under the model's own distribution is
what on-policy practice trains and off-policy imitation does not, which would mean
H1's original mechanism holds on the SFT/precision cell even though the coverage side
reversed. But a competing under-implementation reading is not excluded: SB trained on
only 123 teacher demonstrations for 2 epochs (16 optimizer steps, loss 2.52 → 1.99),
a light touch that may be undertrained, or simply too few examples to install
precision behaviors. T01 cannot separate "SFT can't do precision" from "this SFT run
was too small." The honest statement is that SB losing to a free prompt is real and
its cause is unidentified, a caveat on any "SFT is bad at precision" reading drawn
from this one cell. (Consistent with imitation-of-surface-without-function: SB is
also the arm most drawn to the marker *word* (loose "FINAL" in 85/1183 decodes,
7.2%, the highest of any arm) while never once producing the actual `===FINAL===`
token, §5.3.)

**(g) H3 Tier-3 guard: FAILED, and it confirms the SFT-breakage story off-task.**
The general-capability guard was run on MMLU (0-shot, N=1000 stratified across all 57
subjects, seed 20260715, deterministic answer-letter log-likelihood scoring; full
method and raw scores in [`RUN_H3.md`](./RUN_H3.md)). Accuracy vs Arm 0's 52.4%:

| arm | SA | SB | RA | RB | (P) |
|-----|-----|-----|-----|-----|-----|
| Δpp vs base | **−3.2** | −3.0 | −0.1 | +0.9 | (−14.4) |

**Verdict: FAIL by the pre-registered point-estimate rule, a *marginal* one** (Arm
SA is more than 3 points below base, PREREG §4, H3, the sole violation; but its Δ 95%
CI [−5.3, −1.1] spans the −3 line, see note (i)). The shape is the off-task echo of
§5.6: **both SFT arms regress ~3 pp of general capability; both GRPO arms retain it**
(RA flat, RB slightly up). The §5.6 result that the SFT arms broke more than they
fixed *on their trained task* now has a matching off-task signature: the SFT recipe's
damage is not confined to the task it was trained on, while GRPO's tiny KL leash
(β=0.04, measured KL ~0.006 to 0.016) held it near the base policy on both axes.

**This is plausibly one story with the interaction, not a separate second finding.**
The method split here (SFT damages, GRPO preserves) is the *same* split that
produces the negative interaction (§5.1). SFT's weak coverage recovery with heavy
breakage (Rec 0.284, Brk 0.552, §5.6) and its ~3-pt general-capability regression are
two faces of one behavior: a 123-example scaffold distillation broadly overwriting the
base model. GRPO's conservative, KL-tethered on-policy updates improve the target and
leave the rest intact on *both* the on-task and off-task axes. Read this way, the
reversal is not only "GRPO is unexpectedly good at coverage" but equally "SFT is
unexpectedly *destructive* at this scale," and H3 is the off-task corroboration of the
destructive half. (This is a post-hoc synthesis of two correlated observations, not a
causal claim that off-task regression produces the on-task interaction.)

Three honesty notes carried from RUN_H3.md:
(i) the guard is a **point-estimate** rule and SA's Δ 95% CI is [−5.3, −1.1], which
spans the −3 line; SA (−3.2) and SB (−3.0) are statistically indistinguishable, so
the robust claim is "both SFT arms sit at ≈ −3 pp," not "SA specifically failed and
SB specifically passed"; (ii) MMLU here is scored by log-likelihood over the answer
letters with no generation, so this is a **knowledge** regression, separate from and
additional to SA's decode-time repetition pathology (§5.2); (iii) Arm P's −14.4 is a
measurement artifact: the scaffold makes P's first assistant token a checklist step,
not an answer letter, on the base weights P shares with Arm 0, so it is not a
capability loss, and P is (correctly) not gated by H3.

**(h) Tier-1 vs Tier-2 (CC-75 transfer): the coverage effect does not generalize;
the SFT-destructive pattern does.** *(Added 2026-07-21 after the Tier-2 run,
[`RUN_TIER2.md`](./RUN_TIER2.md); exploratory/descriptive per §6 and underpowered for
the interaction by design, PREREG §5.)* **The lead finding is a non-transfer.** T01's
recovery estimand, re-measured on the *naturalistic* CC-75 corpus that E08 originally
diagnosed (75 prompts, opus/k=3 grading, recovery against a *local* Arm-0 reference
that cancels a measured ~20% stack+grader false-recovery floor; RUN_TIER2 §2),
collapses:

| Rec (arm on its trained cause) | Tier-1 (synthetic pool) | Tier-2 (CC-75 naturalistic) |
|---|---|---|
| SA · coverage | 0.284 | 0.052 |
| **RA · coverage** | **0.740** | **0.076** |
| SB · precision | 0.190 | 0.094 |
| RB · precision | 0.306 | 0.113 |
| **method×cause interaction** | **−0.339** [−0.41, −0.27] | **≈ −0.005** (CIs overlap 0) |

RA (the arm that recovered **74%** of Tier-1 coverage failures, the single cell that
drove the entire negative interaction) recovers **7.6%** on naturalistic coverage.
All four arms recover only 5 to 11% on CC-75, and the interaction is indistinguishable
from zero. **The honest read: T1.4's headline coverage-specific GRPO advantage appears
tied to T01's *synthetic* constraint distribution: the surface-format constraint types
RA mastered (start_phrase, casing, no_commas; §6(d)), rather than a capability that
generalizes to how coverage failures actually appear in the wild.** This is the
empirical corroboration of §6(d)'s confound: the effect lived in the constraint-type
family, and CC-75's naturalistic `constraint_unaddressed` failures do not share it.

What *does* transfer is the other half, now resting on **three independent
measurements** that SFT damages the model more than GRPO does: (i) Tier-1 on-task
breakage (§5.6): SFT arms net-negative, SA broke 55% of base-passed criteria;
(ii) H3 off-task (addition (g)): both SFT arms regress ~3 pp of general capability,
both GRPO arms retain; (iii) Tier-2 CC-75 breakage: SFT arms break **22.8% / 26.0%**
of base-passed criteria vs GRPO's **15.3% / 15.4%**. Three distributions (synthetic
holdout, MMLU, naturalistic CC-75), one consistent signal.

**So the more defensible, transfer-stable conclusion is not the pre-registered
interaction (real, but pool-specific) but this:** the method that looked *worse* on the
engineered coverage benchmark (SFT) is the one that reliably degrades the model across
every distribution tested, while GRPO's advantage is concentrated where the reward is a
clean verifier and does not obviously generalize past it. (Tier-2 is descriptive and
underpowered for the interaction (PREREG §5's 5 to 8-point floor) so this does **not**
re-test H1; the non-transfer is a statement about recovery *magnitude* corroborating
§6(d), and the CC-75 grades carry the §2 single-judge/stack caveat, held on the
absolute numbers not the qualitative reading.)

## 7 · Threats to validity and limitations

1. **Main effects are compute-confounded** (GRPO ~10× FLOPs; wall-clock minutes
   vs hours); only the interaction is compute-fair; all GRPO-vs-SFT *level*
   comparisons above are descriptive.
2. **H3 evaluated: FAILED; off-task regression confirmed for the SFT arms.** The
   Tier-3 MMLU guard (§6 post-finalization additions (g); [`RUN_H3.md`](./RUN_H3.md))
   ran 2026-07-20: both SFT arms fell ~3 pp below base (SA −3.2, the sole guard
   violation but a *marginal* one; CI [−5.3, −1.1] spans the −3 line; SB −3.0), both
   GRPO arms held (RA −0.1, RB +0.9). The off-task
   regression that §5.6's on-task breakage made a *concern* is now observed, for
   SFT specifically, not GRPO (though SA/SB are statistically indistinguishable at
   the 3-pt line; see (g)). **Tier-2** (CC-75 transfer) has now run (addition (h);
   [`RUN_TIER2.md`](./RUN_TIER2.md)): external validity *was* tested and the
   coverage-GRPO effect did **not** transfer (RA coverage recovery 74% → 7.6%,
   interaction ≈ 0), while the SFT-breakage pattern did. Tier-2 is descriptive and
   underpowered for the interaction (PREREG §5), and its single-judge/stack
   reproduction caveat (RUN_TIER2 §2) applies to its absolute numbers.
3. **Single model, single scale, single family**, one cell of the
   (family × scale) grid; a heavily post-trained subject (Meta's SFT+RS+DPO)
   whose headroom and method-response may be atypical.
4. **LLM-assigned cause labels.** The E08 Pareto is 95.1% single-classifier
   labeled; the ~40-row manual hand-check was deferred. The pre-registration
   recommends it as first follow-up on a null/ambiguous result; the result is
   neither, but mechanism (c) renews the case for auditing what
   `constraint_unaddressed` actually contained.
5. **Coverage was recalibrated mid-experiment** (before any training, under a
   pre-committed stopping rule, estimand untouched), but absolute coverage
   numbers are not comparable to pre-recal expectations, and see mechanism (c).
6. **SA's generation pathology** (16.5% degenerate-loop cap-hits) is itself a
   finding about this SFT recipe at eval temperature; the sensitivity analysis
   bounds its effect on H1 (§5.2) but its cause (123-example fine-tune?
   scaffold imitation?) is undiagnosed.
7. **Verifier-definable causes only**: comprehension-type causes
   (`constraint_misread`, `input_misread`) are out of scope by construction, and
   mechanism (b) suggests verifier-definability itself favors RL: the scope
   restriction and the finding may be entangled.
8. **Minor execution deviations, all logged:** eval via local HF `generate()`
   instead of vLLM (pre-declared); TRL 1.8.0 lacks `max_prompt_length` (inert;
   measured prompts ≪ the intended 512); RA's step-50/150 decision-point
   checkpoints rotated out after the fact (logs preserved; weights not
   re-instantiable).

## 8 · Conclusion

T01 set out to confirm that GRPO's edge over SFT concentrates on precision-type
failures. The pre-registered test returned the opposite, decisively: the
method×cause interaction is −0.34 (95% CI [−0.41, −0.27]), robust to every
treatment of the one identified confound, with GRPO's edge concentrating on
coverage-type failures, where it repaired 74% of base failures while breaking
5% of base passes, against SFT's 28% repaired / 55% broken. The honest summary
is not "H1 was wrong about GRPO" (GRPO still beat SFT on precision) but "H1 was
wrong about *coverage*": verifier-checkable coverage constraints turned out to be
excellent RL targets and poor imitation targets: the latter partly for a
train/eval grading reason specific to how the teacher data was accepted.
Diagnosis-driven training remains supported in the weak sense (failure type
modulated method effectiveness strongly); the specific mapping pre-registered
here does not. The promised Tier-3 guard has since run and **failed for the SFT arms** (SA −3.2 pp,
SB −3.0; both GRPO arms held); off-task confirmation of §5.6's on-task SFT breakage
(§6 additions (g)). Tier-2 has since tested transfer to the naturalistic CC-75 corpus
(§6 addition (h)): the coverage-GRPO advantage did **not** generalize (RA coverage
recovery 74% → 7.6%, interaction ≈ 0), locating the headline effect in T01's synthetic
constraint distribution, while the SFT-more-destructive pattern transferred, making it
the more defensible, transfer-stable conclusion. Recommended next steps, in order: the
~40-row E08 label hand-check (cheap, tests mechanism (c) at the source); a marker-reliability or
whole-text-graded SFT re-run to isolate mechanism (a); then T02 cross-family, with
the *same* constraint-type families instantiated under both causes so the archetype
effect is not confounded with type-family or pool-authoring epoch (§6 additions (d)).

## 9 · Provenance appendix

Pointers, not duplicates. Volume paths are gitignored by design; tracked records
are the committed provenance.

- **Pre-registration:** [`../PREREG.md`](../PREREG.md) (design §§3-10; append-only
  amendment log: coverage recal, SFT parity, GT3 bar, no-vLLM).
- **Frozen configs:** [`../config/t1_3_frozen.md`](../config/t1_3_frozen.md),
  [`../config/t1_3_grpo_probe.md`](../config/t1_3_grpo_probe.md),
  [`../config/recalibration_stopping_rule.md`](../config/recalibration_stopping_rule.md).
- **Run records:** [`RUN_SA.md`](./RUN_SA.md) · [`RUN_SB.md`](./RUN_SB.md) ·
  [`RUN_RA.md`](./RUN_RA.md) · [`RUN_RB.md`](./RUN_RB.md) ·
  [`RUN_T1.4.md`](./RUN_T1.4.md) (eval + sensitivity).
- **Analysis code (tracked):** `../analysis/tier1_analysis.py` (pre-registered
  estimand), `../analysis/tier1_truncation_sensitivity.py` (V0-V3),
  `../analysis/tier1_descriptives.py` (post-hoc §5.6 tables);
  eval pipeline `../eval/{run_tier1.sh, tier1_decode.py, grade_tier1.py}`.
- **Volume artifacts:** `results/eval_t1_4/{analysis.json,
  truncation_sensitivity.json, criteria.jsonl, decode_meta.jsonl, run.log,
  decodes/}`; adapters `results/adapters/T01-{SA,SB,RA,RB}/`; training logs
  `results/logs/` (incl. `RA_gate_step50.json`, `RA_step150_s9.json`,
  `{RA,RB}_full_run.json`).
- **Data:** `../data/{train,holdout,sft}/`, parity manifests
  `../data/sft_manifests/{SA,SB}.json` (123/123, 123/270; seed 20260715).

---

*Report finalized 2026-07-20 from committed artifacts; supersedes the 2026-07-18
scaffold version of this file. Post-finalization passes incorporated the H3 Tier-3
guard result ([`RUN_H3.md`](./RUN_H3.md)) and four reviewer questions on the confound
structure (2026-07-20; §6 additions (d)-(g); the `===FINAL===` marker count corrected
2300→4800 after verifying all four arms), then the Tier-2 CC-75 transfer result
(2026-07-21; [`RUN_TIER2.md`](./RUN_TIER2.md); §6 addition (h); abstract, STATUS, §2,
§7, §8 updated to match). PREREG.md remains the authority on the design; where this
report explains or interprets, §§5.6 and 6 are explicitly post-hoc.*
