# T01 — Pre-registration: method × cause interaction (SFT vs GRPO × coverage vs precision)

**Status:** PRE-REGISTERED · committed at Gate E→T · **append-only from this
commit** (amendments must be dated and added to the log at the bottom; nothing
above the log may be edited).
**Subject:** `meta-llama/Llama-3.2-3B-Instruct`.
**Bound at:** Gate E→T, 2026-07-15, from the E08 census
(`E08-llama3-2-3b-cc75`, post truncation-repair).
**Plan of record:** `docs/superpowers/plans/2026-07-15-e08-t01-lamma.md` (Part II).
**Gate E→T inputs:** `experiments/E08/pareto.json`,
`meta/2026-07-15-e08-llama3-2-3b-cc75.md`.

---

## 1. Bound causes (from E08, Gate E→T)

E08 diagnosed 882 consensus-FAIL criteria (one blinded root-cause label each,
3-judge panel verdicts, `claude-fable-5→opus-4-8` classifier chain). The top two
causes are high-prevalence and mechanistically distinct, and map cleanly onto
the plan's two intervention archetypes:

| binding | cause (E08 label) | count | share | archetype | E07 analog |
| --- | --- | --- | --- | --- | --- |
| **`CAUSE_A`** | `constraint_unaddressed` | 254 | **28.8%** | **coverage** — failures of noticing/tracking requirements | `constraint_dropped` |
| **`CAUSE_B`** | `execution_slip` | 321 | **36.4%** | **precision** — failures of exact execution | `execution_slip` |
| | **combined** | 575 | **65.2%** | | |

Both clear the prevalence bar (jointly ≥ ~35% of failures; each individually
well-populated for holdout-transferable measurement). They are mechanistically
distinct: coverage is *whether* a requirement is engaged at all; precision is
*whether* an engaged requirement is executed correctly. Under E08's reasoning-off
decode, `constraint_unaddressed` is the collapse of E07's
`never_surfaced`+`dropped` (no trace to separate them) — the coverage family.

## 2. Gate E→T decision

Per the plan's Step-3 decision table, the E08 outcome is **row 1: "Top-2 causes
map cleanly onto coverage vs precision archetypes" → keep H1 as primary, bind
and proceed.** No amendment to the provisional primary hypothesis is warranted by
the census. (Row 2, "same-archetype / misread-dominated," does not apply:
`constraint_misread` is #3 at 14.4%, well behind the two archetype-clean leaders.
Row 3, "no cause reaches measurable prevalence," does not apply: the model is far
from ceiling at 43.4% criterion pass.)

## 3. Design

A 2 (method) × 2 (data-cause) factorial of LoRA adapters, plus two eval-only
reference arms. **Arm ceiling: 6.**

```
                        TRAINED WITH →
                  ┌──────────────┬──────────────┐
                  │  SFT (copy)  │  GRPO (RL)   │
 ┌────────────────┼──────────────┼──────────────┤
 │ CAUSE_A data   │   Arm SA     │   Arm RA     │   CAUSE_A = coverage
 ├────────────────┼──────────────┼──────────────┤
 │ CAUSE_B data   │   Arm SB     │   Arm RB     │   CAUSE_B = precision
 └────────────────┴──────────────┴──────────────┘
 + Arm 0 : base model, untrained          (eval only; not in the primary calc)
 + Arm P : base + enumerate-then-verify system prompt (eval only, $0)
```

**Primary estimand (the interaction):**

```
Interaction = [ Rec(RB, CAUSE_B) − Rec(SB, CAUSE_B) ]
            − [ Rec(RA, CAUSE_A) − Rec(SA, CAUSE_A) ]
```

i.e. (RL−SFT advantage on precision) minus (RL−SFT advantage on coverage).

**`Rec(arm, cause)` (recovery):** the fraction of Arm-0's failed-criterion set
for that cause pool (majority over Arm-0's decodes) that the arm now passes
(majority over the arm's decodes). Each trained arm is scored on the cause pool
matching its training data.

**Structural fairness (reported in the writeup):** (1) **compute-fair** — GRPO's
~10× FLOP surplus applies to both causes equally and cancels in the interaction;
main effects are compute-confounded and reported as descriptive only;
(2) **family-fair** — RL-family quirks cancel across the two RL cells;
(3) **data-fair** — both methods in a row consume the same 300-prompt pool.

## 4. Hypotheses

**H1 (PRIMARY, confirmatory — the one and only confirmatory test):**
GRPO's advantage over SFT is **larger for `CAUSE_B` (precision) than for
`CAUSE_A` (coverage)**, i.e. **`Interaction > 0`**.
*Rationale:* coverage is a behavioral template — enumerate→draft→verify — that
imitation (SFT) installs cheaply; precision is correctness under the model's own
output distribution, which on-policy practice with verifier feedback (GRPO)
trains directly. Confirmed **iff** the 95% CI for `Interaction` excludes 0 **and**
`Interaction > 0`.

**H2 (SECONDARY, exploratory):** within-method (SFT) data-targeting crossover —
a cause-matched vs cross-cause difference-in-differences: does training on a
cause's own data help that cause more than the other cause's data does?
Reported with CIs, labeled exploratory; not a confirmatory test.

**H3 (REGRESSION GUARD, not a hypothesis to confirm):** no trained arm scores
**more than 3 points below Arm 0** on the Tier-3 general-capability battery.
Violations are reported prominently.

## 5. Sample sizes, decode & training config

**Data (per cause):** 300 train + 200 holdout. Holdout under `data/holdout/`;
training code must never read that path (enforced by a test). Contrast
requirement: A-prompts (6–12 heterogeneous, buried constraints) and B-prompts
(2–4 exactness-demanding constraints) must be sortable at a glance, or the
interaction collapses toward 0 by construction. Difficulty calibrated to a
30–70% base-model criterion-pass band (above IFEval difficulty).

**Contamination screen:** 13-gram overlap vs (a) T01 holdout, (b) CC-75,
(c) IFEval originals; hits discarded/regenerated/logged.

**SFT data:** teacher = strong open instruct model (OpenRouter), ≤4
attempts/prompt, keep only 100%-verifier-pass responses; archetype-specific
format (coverage: inventory→draft→checklist; precision: worked
computation→draft→reconcile).

**Training (frozen after one hyperparameter probe per *method*, no per-arm
tuning):**

| knob | SFT (SA, SB) | GRPO (RA, RB) |
| --- | --- | --- |
| adapter | LoRA r=16, α=32, attn+MLP (identical all arms) | same |
| data / epochs | 300 filtered pairs × 2 | 300 prompts × 2, k=6 rollouts, rollout temp 0.9 |
| LR | 1e-4 cosine | 5e-6–1e-5 |
| reward | — | verifier fraction − malformed penalty (length-capped) |

**Evaluation decode:** Tier-1 = 400 holdout prompts (200/cause) × **k=3 decodes**,
temperature **0.6**, seeds logged, via vLLM; verifier-graded. Per-criterion
record `{arm, prompt_id, decode_idx, cause_pool, criterion_id, pass}`.

## 6. Statistical analysis (fixed in advance)

- **Pairing:** criterion-level, base vs arm on identical criteria.
- **Clustering:** all uncertainty clustered at the **prompt** level (criteria
  within a prompt are not independent — clustering at the criterion level
  fabricates significance).
- **Primary CI:** prompt-level **cluster bootstrap** — resample prompts with
  replacement *within each cause pool*, **10,000 iterations**, report the
  2.5th/97.5th percentiles of the `Interaction` sampling distribution.
- **Decision rule:** H1 confirmed **iff the 95% CI for `Interaction` excludes 0
  and `Interaction > 0`.**
- **Multiple-comparisons stance:** **exactly one confirmatory test (H1).**
  Everything else — H2, H3, main effects, McNemar flip tables, overall/full-prompt
  rates, Tier-2 transfer, Arm-P comparison, cost-per-point, compute disclosure —
  is **exploratory / descriptive**; reported with CIs but not error-rate-corrected
  and not used to claim confirmation.
- **Power context (reported, not a gate):** 200 prompts/cause × k=3 ⇒ ~3–4-point
  detection floor for the interaction (vs CC-75's 5–8 points — why Tier-2 is
  descriptive only).

## 7. Seeds (fixed in advance)

- **Data-split seed:** `20260715` (train/holdout partition per cause pool).
- **Tier-1 eval decode seeds (k=3):** `20260715`, `20260716`, `20260717`
  (one per decode index; logged per record).
- **Bootstrap seed:** `20260715` (10,000 resamples).
- Training seeds logged per arm at run time.

## 8. Outcome interpretations (committed now, before any data)

| Outcome | Interpretation |
| --- | --- |
| `Interaction > 0`, CI excludes 0 | **H1 confirmed** — failure type determines method choice; diagnosis-driven training is real. |
| `Interaction ≈ 0` (CI includes 0) | Method choice is cause-independent at this scale — contradicts the premise; still ships as a null. |
| `Interaction < 0`, CI excludes 0 | Opposite direction; report honestly, speculate cautiously. |
| Arm P ≈ trained arms | A $0 system prompt matched fine-tuning — the baseline nobody runs. |
| Gains co-move with loop-rate ↓ or length ↑ | Decoding/verbosity artifact, not capability; say so. |

## 9. Kill-switches & fallbacks

- **GRPO stall** (mean reward flat by ~150 steps on either RL arm) → demote
  **both** RL cells to **RFT** (best-of-8 base-model rollouts, verifier-filtered,
  SFT on survivors); rename the contrast honestly ("on-policy verifier-filtered
  vs off-policy imitation"); **dated PREREG amendment required**.
- **SFT rejection yield < 30%** (Gate GT2) → simplify prompt difficulty, log it.
- **Budget ceiling** → cut Tier-2 to the 4 H1-critical arms (0, P, SB, RB); log
  the downgrade in run metadata (never applied silently).

## 10. Threats to validity (stated before a reviewer does)

1. **Compute mismatch on main effects** — GRPO has ~10× the FLOPs of SFT; the
   *main effects* are compute-confounded (descriptive only). The **interaction is
   immune** (the surplus cancels).
2. **Single family / scale** — one cell of the (family × scale) grid;
   cross-family is the natural T02.
3. **LLM-assigned cause labels** — the E08 Pareto is 95.1% labeled by a single
   classifier (`claude-fable-5`); the manual hand-check was **deferred** at E8.2
   in favor of the 3-judge panel protocol. **If T01's primary result is null or
   ambiguous, the ~40-row hand-check is the recommended first follow-up** — not
   yet performed.
4. **Verifier-definable causes only** — coverage and precision are both
   mechanically checkable; comprehension-type causes (`constraint_misread`,
   `input_misread`) are out of scope by construction.
5. **Heavily post-trained subject** — Llama-3.2-3B already had SFT+RS+DPO applied
   by Meta; headroom and method-response may differ from a base model.
6. **RFT fallback** — if triggered, the RL cells are no longer strictly on-policy;
   reported honestly.
7. **Provider/stack variance** — E08 ran the candidate via OpenRouter; T01 arms
   run locally. Arm 0 is re-run locally in Tier-2 for stack parity; the
   local-vs-OpenRouter delta is itself reported as a consistency check.

---

## Amendment log (append-only; date every entry)

### 2026-07-16 — T1.3 operational config, GT3 pass bar, and no-vLLM training backend

**Context.** T1.3 execution on the rented A100 surfaced items the frozen design
(§§3–7) and the Plan of Record (Part II — **local-only / gitignored under
`docs/superpowers/`, not present on the training host**) did not fully pin down,
plus one environment change forced at Gate GT0. Inlined here so this tracked
PREREG is self-sufficient on the training host. **None of this alters the
estimand, hypotheses, sample sizes, seeds, or analysis in §§3–7** — these are
execution details and one speed-only backend change.

**(a) GT3 pass bar (positive criterion; §9 gave only the failure kill-switch).**
GT3 passes when **all four arms show learning**: SFT arms (SA, SB) — training
loss clearly decreasing across the 2 epochs; GRPO arms (RA, RB) — mean batch
reward trending up (NOT flat by ~150 steps). The §9 GRPO-stall kill-switch
(→ both RL cells to RFT) is unchanged and remains the only sanctioned failure path.

**(b) GRPO probe sweep — resolves the k-vs-sweep tension.** `num_generations`
(k) is **frozen at 6** (§5 design table) and is **NOT** a sweep axis. The one
per-*method* probe (~50 prompts, 3-hr cap) sweeps **LR only**, within the
pre-registered GRPO range **5e-6–1e-5**; pick the LR with the healthiest reward
trend, then freeze for both RL arms (no per-arm tuning). **KL/β is not
pre-registered:** use the TRL GRPO default (β=0.04); change it only if the
default run is unstable (reward collapse / KL blow-up), and if so freeze + log
the value. (Supersedes an operator instruction that mistakenly listed
`num_generations`/KL as sweep axes.)

**(c) Implementation defaults absent from the §5 table (probe-validated, then
frozen).** Execution parameters, not design; the interaction estimand is immune
to them. Frozen once the probe confirms memory fit + stable training:
- **LoRA (both methods, identical):** target_modules = all attention+MLP linears
  (`q,k,v,o_proj`, `gate,up,down_proj`); lora_dropout 0.05; bias none.
- **SFT:** **completion-only loss** (mask the user prompt; train on scaffold +
  `===FINAL===` + answer per §5); max_seq_len 4096 (report token-length dist +
  truncation rate at probe; bump to 8192 if truncation >1%); effective batch 16
  (per-device 4 × grad-accum 4); LR 1e-4 cosine, warmup 0.03; bf16 + gradient
  checkpointing.
- **GRPO:** max_prompt_length 512; max_completion_length 1536 (must fit
  scaffold+answer before `===FINAL===`; report cap-hit rate at probe); rollout
  temp 0.9; β 0.04.
- **Reward wiring:** grade `extract_final(completion)` with
  `verifiers/reward.py::constraint_reward`. That fn is `(response, specs) ->
  float`; TRL wants `(prompts, completions, **kwargs) -> list[float]`, so the
  GRPO dataset carries a per-prompt `specs` column and the adapter returns
  `[constraint_reward(extract_final(c), specs_i, max_chars=M) for c in
  completions]`. `max_chars M` activates the pre-registered length cap; set from
  the teacher-answer length distribution (dock genuine padding only), probe-tunable.
- **Checkpoints/durability:** output_dir
  `/workspace/failure-mode-id/results/adapters/T01-{arm}/` (persistent volume);
  save every 50 steps + epoch end; save_total_limit 3; resume-safe state (per the
  T01-pilot durable-checkpoint pattern).

**(d) Training rollout backend — no vLLM (Gate GT0 environment freeze).** The
GT0 dependency resolution produced a working, frozen training env (torch
2.5.1+cu121, TRL 1.8.0, transformers 4.46.3, peft 0.19.1) in which **vLLM cannot
be installed without forcing an incompatible torch/CUDA downgrade**. GRPO
rollouts therefore run on the stock **`model.generate()`** backend
(`GRPOConfig(use_vllm=False)`), optionally with
`use_transformers_continuous_batching=True` (no vLLM). This **supersedes** the
vLLM-rollout assumption in Part II §T1.0 and any training-side note predating
GT0. **Speed-only — no effect on the sampling distribution, estimand, or any
result** (generate() and vLLM draw from the same policy at temp 0.9). The Tier-1
**eval** decode (§5, "via vLLM") is a **separate T1.4 concern**: if vLLM is used
there it runs in an **isolated venv** never mixed into the training env, and may
equally fall back to generate(). Decided at T1.4.

**Provenance.** The Plan of Record named in the header (Part II) is a local-only
planning artifact (gitignored under `docs/superpowers/`); its load-bearing T1.3
values are inlined above so this tracked PREREG stands alone on the training host.
