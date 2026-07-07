# Failure-mode taxonomy v1 + root-cause backfill · 2026-07-07 · `diagnose` stage over E03/E04/E05

**Not an experiment run** — this is the analysis layer's first deliverable: a
grounded root-cause taxonomy derived from E05's failures, applied (blinded)
to every failed criterion of E03, E04, and E05, with per-experiment iteration
syntheses. Spec: `docs/superpowers/specs/2026-07-06-failure-mode-analysis-design.md`.
Dashboard: each experiment's **Analysis** tab. Artifacts:
`failure_analysis` blocks in `outputs/experiments/E0{3,4,5}-*.json` (schema 3.1).

## TL;DR

- **The #1 root cause on the flagship sample (E05, reasoning-on, 20 prompts) is `constraint_dropped`** — 54 of 219 failed criteria (25%): the reasoning trace tracks or even verifies the requirement and the final answer loses it anyway, mostly during patch-style replanning of schedules. With `constraint_overridden` (27) and `constraint_never_surfaced` (16), **44% of E05's failures are reasoning-process failures** that are invisible without stored traces.
- **Reasoning mode fixes comprehension, not execution.** On the identical 3 prompts, E03 (reasoning-off) → E04 (reasoning-on) eliminated `input_misread` (9→0) and `degenerate_output` (8→0) and halved `constraint_misread` (13→7), while `execution_slip` persisted (31→20) and became 61% of what remains. Inference-time compute is not a substitute for execution-level training signal.
- **The diagnosis layer caught a probable benchmark bug**: all three models' (independent, blinded) diagnoses flagged the *same* criterion — CIF-018 #18 — as `judge_suspect`, each concluding the criterion's premise is contradicted by the roster it grades against (Marcus, SafeSERV, is available Wed/Thu mornings). One criterion, three independent flags → top of the human-validation queue.
- **Verifier-based RL can cover roughly half the problem**: 54% of E05's failed criteria are auto-verifiable, rising to 70% for `execution_slip` and 67% for `constraint_sacrificed` — the mechanical-check causes are exactly the RLVR-ready ones.
- **The iteration chain says: don't train yet.** E05's synthesis judges the Pareto unstable (top cause flipped between runs, cross-sample confound) and recommends one more robustness run (limit ≥60, fixed seed, identical config) plus human validation of ~40 diagnosed labels before committing to training data.
- Taxonomy fit is good: `other` rate **1.4%** on E05 (tripwire was 10%); 85% of labels carry `high` confidence.

## Method (two-pass, blinded)

**Pass 1 — open coding.** 70 failed criteria sampled from E05 (stratified by
model × instruction type), one blinded free-text root-cause description each
(Opus 4.8, Message Batches). The analyst payload contains the prompt, all
criteria, the unmet indices, the response, and the reasoning trace — never
judge reasons, candidate identity, or the second judge's verdicts, so the
taxonomy cannot inherit judge framings. 70/70 returned.

**Consolidation.** Clustered by *mechanism of model behavior*, not topic →
8 derived categories (below), plus a-priori `judge_suspect` (licensed
disagreement: the analyst must locate the failure itself before labeling, and
may refuse to), `other` (escape hatch), and `constraint_unaddressed` (the
no-trace collapse of never-surfaced/dropped, used for reasoning-off runs).
Two categories were kept below the ≥4-sample support rule, deliberately:
`constraint_never_surfaced` (n=3; the never-vs-dropped split is the
taxonomy's load-bearing axis) and `degenerate_output` (n=3 in the sample but
~16 in the E05 population — one degenerate response owns them all).
Knowledge fabrication (n=2) was left for `other`; Pass 2 confirmed exactly 3
`other` rows, all fabricated-recall cases, so it stays out of v1.

**Pass 2 — labeling.** The standing `diagnose` stage: one blinded batch call
per failed (model, prompt) cell labels all of that cell's failed criteria
against the frozen enum. Backfill ran in chain order E03 → E04 → E05 so each
experiment's synthesis could read its predecessor's published analysis.
Zero pipeline errors; 323 criteria diagnosed.

### Taxonomy v1 (sample support from Pass 1, n=70)

| key | mechanism | n | trace? |
| --- | --- | --- | --- |
| `constraint_dropped` | tracked in trace, lost from answer (replanning/transcription) | 11 | ✓ |
| `constraint_overridden` | noticed, then argued out of (habit/source/helpfulness wins) | 10 | ✓ |
| `constraint_never_surfaced` | never extracted into the plan at all | 3 | ✓ |
| `constraint_sacrificed` | traded away under (perceived) conflict, often with a false compliance claim | 11 | |
| `constraint_misread` | engaged but interpreted wrong (scope/boundary/conditional/format words) | 13 | |
| `input_misread` | source-data parsing (columns, false equivalences) | 5 | |
| `execution_slip` | right plan, wrong rendering (arithmetic/markup/templates/no self-check) | 12 | |
| `degenerate_output` | generation collapse (restart loops, no committed answer) | 3 | |

## E05 failure Pareto (219 failed criteria, Opus verdict basis)

| root cause | n | % | 9b / 35b / 397b | top instruction types | auto-verifiable |
| --- | --- | --- | --- | --- | --- |
| constraint_dropped | 54 | 25% | 17 / 24 / 13 | Implicit 27, Multistep 17 | 54% |
| constraint_misread | 35 | 16% | 19 / 10 / 6 | Negative 13, Conditional 8 | 46% |
| execution_slip | 30 | 14% | 6 / 14 / 10 | Implicit 11, Multistep 9 | 70% |
| constraint_overridden | 27 | 12% | 5 / 9 / 13 | Conditional 10, Multistep 8 | 48% |
| constraint_sacrificed | 21 | 10% | 4 / 7 / 10 | Multistep 15 | 67% |
| constraint_never_surfaced | 16 | 7% | 5 / 6 / 5 | Implicit 8, Negative 7 | 38% |
| degenerate_output | 15 | 7% | 15 / 0 / 0 | Implicit 15 | 67% |
| input_misread | 15 | 7% | 10 / 0 / 5 | Negative 12 | 47% |
| other (fabricated recall) | 3 | 1.4% | 2 / 1 / 0 | Multistep 3 | 100% |
| judge_suspect | 3 | 1.4% | 1 / 1 / 1 | Conditional 3 | 0% |

Reading the cuts:

1. **The Implicit weakness decomposes into retention, not comprehension.** E05's hardest category (Implicit, 63–68% pass) is dominated by `constraint_dropped` (27) and `degenerate_output` (15 — all one qwen-9b response), not `constraint_misread` (7). Models mostly *do* infer the implicit constraints; they fail to carry them into the final answer.
2. **Model size changes the failure flavor, not just the rate.** qwen-9b over-indexes on `constraint_misread` (19), `input_misread` (10), and owns every `degenerate_output`; qwen-397b's failures shift toward `constraint_overridden` (13) and `constraint_sacrificed` (10) — the flagship notices more and then trades off or overrides. Scale converts comprehension failures into prioritization failures rather than eliminating failure.
3. **`constraint_sacrificed` is a Multistep phenomenon** (15/21): long task lists with real conflicts (coverage vs ceilings, time budgets) — and the Pass-1 descriptions show it frequently pairs with a false compliance claim ("all staff have consecutive days off", "meets the requirement"), i.e., models overclaim rather than surface tradeoffs.
4. **Compound failures are common**: the most frequent secondary labels (overridden 25, execution_slip 25, dropped 22) show single criteria failing through two mechanisms at once.
5. Judge concurrence: 205 of 219 diagnosed rows are `both_fail` (Fable independently agrees), 5 `opus_only` (judge-noise candidates, de-emphasized in the dashboard Pareto), 9 `fable_refused` (the CIF-006 refusal block). Confidence: 186 high / 32 medium / 1 low.

### The `judge_suspect` catch

All three models were failed on CIF-018 #18 ("bring in a 7am SafeSERV FOH on
Wed/Thu"), and all three *independent, blinded* diagnoses refused the premise:
Marcus (SafeSERV) is available Mon–Fri mornings, so the rule's trigger
condition doesn't fire. Pass 1 had itself labeled one of these a
`constraint_misread` — the diagnosis layer disagreeing with both judges *and*
with Pass 1 on the same cell is exactly the signal the reserved label exists
to surface. These three rows head the human-validation queue; if the
criterion is confirmed defective, every model's CIF-018 score gains a point
and one of E05's seven ladder-order violations (CIF-018: 15/14/12) weakens.

## What inference-time compute fixes (E03 ↔ E04, same 3 prompts)

E03 (reasoning-off, temp 0, 71 failed criteria) vs E04 (reasoning-on,
temp 0.6, 33) — cross-config but same prompts, so composition shifts are
meaningful even though the treatment is confounded (reasoning+temp+routing,
inherited caveat):

| root cause | E03 | E04 | read |
| --- | --- | --- | --- |
| execution_slip | 31 | 20 | **persists** — 44% → 61% of remaining failures |
| constraint_misread | 13 | 7 | halved |
| input_misread | 9 | 0 | **eliminated** (all 9 were qwen-9b) |
| degenerate_output | 8 | 0 | **eliminated** (all 8 were qwen-35b, no-trace run) |
| constraint_sacrificed | 6 | 1 | mostly gone |
| constraint_unaddressed → dropped/overridden | 4 | 3+2 | the no-trace bucket resolves into its trace-visible parts |

Reasoning mode buys comprehension and stability: data parsing, degenerate
collapse, and most misreads improve or vanish. What it does **not** buy is
execution — arithmetic, formatting, per-item rendering — which is precisely
the most auto-verifiable category (70%). The intervention implication is
clean: **prompt-side/inference levers are the cheap fix for comprehension
failures; execution failures need training-side (verifier-reward) signal.**
Caveat: E05's degenerate case (qwen-9b, reasoning-*on*) shows reasoning mode
relocates rather than abolishes collapse risk — E03's collapses were 35b's.

## Ladder persistence (E05)

Of 121 distinct (prompt, criterion) failures: **33 failed by all three
models** (scale-resistant — the buy-data-for-this list), 56 by exactly one,
32 mixed. The all-model set is led by `constraint_overridden` and
`constraint_misread` (16 each) — instruction-priority and
instruction-semantics gaps that scale does not fix — while single-model
failures skew toward qwen-9b's comprehension/degeneration profile, which
scale (or a second draw) already handles.

## Training recommendations (in priority order)

1. **CoT→answer faithfulness rewards** — targets `constraint_dropped` (25% of E05). Penalize final answers that omit or contradict constraints their own trace satisfied; train long-horizon state tracking across plan revisions (the dominant aggravator is patch-style replanning). 54% of these criteria are auto-checkable, so a trace-vs-answer consistency verifier is buildable.
2. **Verifier-based RL on execution** — targets `execution_slip` (+ the mechanical halves of `sacrificed`/`dropped`). 70% auto-verifiable, reasoning-resistant (E03→E04), present at every scale. The cheapest high-volume signal in this list.
3. **Instruction-priority preference data** — targets `constraint_overridden` (grows with scale; co-leads the scale-resistant set). Pairs where explicit user constraints must beat priors, source-document conventions, and helpfulness instincts.
4. **Conflict-arbitration preference data** — targets `constraint_sacrificed` and its false-compliance-claim habit: prefer responses that surface infeasibility or flag the tradeoff over responses that silently sacrifice and overclaim. (The overclaiming itself is a self-verification failure worth its own reward term.)
5. **Instruction-semantics contrast sets** — targets `constraint_misread` (scope/boundary/conditional/format-word minimal pairs; Negative instructions over-represented).
6. **Structured-input robustness + decoding hygiene** — smaller, model-specific: column-aligned table parsing with re-derivation steps (qwen-9b), and anti-loop termination training (collapse observed in different models under different configs).

### Iteration trajectory (the synthesis chain)

Each experiment's analysis now carries machine-written next steps and a
review of its predecessor's. Backfill caveat: E03→E04→E05 ran historically,
*before* these recommendations existed, so the chain's "NOT addressed" review
lines describe what the historical sequence didn't do — the prospective value
starts with the next run. The chain's content is nonetheless consistent:
E03's synthesis asks for a bigger seeded sample and human validation; E04's
repeats both and adds the degenerate-output ablation; **E05's verdict: early-
to-mid iteration, NOT ready to train** — top cause flipped between runs
(execution_slip → constraint_dropped) across confounded samples, so run one
robustness pass (limit ≥60, fixed seed, identical config) and human-score
~40 diagnosed labels first. That matches this report's own reading.

## Limitations

- **Self-diagnosis residual**: Opus diagnoses failures Opus judged. Blinded inputs (no judge reasons, no model identity, no second-judge verdicts) mitigate anchoring; they cannot remove same-model bias. The 60-row judge-validation sample plus the ~40-row label-validation recommendation are the anchor — both still human-unscored.
- **Cross-sample comparisons**: E03/E04 (3 prompts) vs E05 (20 prompts, different set); E03↔E04 additionally confounds reasoning with temperature and routing. Composition shifts are directional.
- **Single-draw noise**: every per-prompt cell is one sample at temp 0.6 (E04/E05); the 7/20 ladder inversions from the E05 report remain unexplained pending a repeat-draw run.
- **Taxonomy provenance**: derived from one experiment's (E05's) failures under one treatment. The >10% `other` tripwire is the drift alarm; v1 sits at 1.4%.
- **E03's labels are coarser by design**: no traces → never-surfaced/dropped/overridden collapse into `constraint_unaddressed` (4 rows) and mechanism attribution leans on answer-observable categories.

## Reproducibility

```
# Code: branch feat/failure-analysis (spec + plan in docs/superpowers/).
# Taxonomy derivation (Pass 1): ~70 blinded open-coding calls on E05
uv run python scripts/opencode_failures.py E05-reasoning-rand20p --sample 70
#   → outputs/opencode/E05-reasoning-rand20p.jsonl (70/70)
# Consolidation → pipeline/_taxonomy.py DERIVED (TAXONOMY_VERSION = 1)

# Pass 2 backfill, chain order (each aggregate publishes before the next diagnose):
for slug in E03-judge-compare-3p E04-reasoning-smoke-3p E05-reasoning-rand20p; do
  uv run python main.py diagnose  --experiment "$slug"
  uv run python main.py aggregate --experiment "$slug" --run-report <its run report>
done
# 323 criteria diagnosed, 0 errors; E05 other-rate 1.4% (gate ≤10%).
# Diagnosis judge: claude-opus-4-8 (Fable refuses on CIF-006-class content).
# Verdict basis: the canonical first judge (Opus for all three experiments).
```

Future experiments run the stage automatically inside `all` (opt out:
`--diagnose off`; backfill later with `main.py diagnose --experiment <slug>`).

## Open questions / follow-ups

- Is CIF-018 #18 a defective criterion? (3/3 independent `judge_suspect` flags; human check is one criterion's worth of work.)
- Does `constraint_dropped`'s dominance survive a same-config repeat draw (E05's synthesis robustness rec) — or is it partly one sample's schedule-heavy composition?
- Do the 5 `opus_only` diagnosed rows (judge-noise candidates) fail human validation? If so, E05's honest failure count drops.
- Fable's refusal block (9 `fable_refused` rows) still awaits the refusal census before Fable can be trusted as a solo judge anywhere.
