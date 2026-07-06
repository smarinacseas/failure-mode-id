# Reasoning-on randomized 20-prompt run · 2026-07-06 · `all --experiment E05-reasoning-rand20p --limit 20 --sample-seed 20260706 --reasoning on --max-tokens 48000 --temperature 0.6 --timeout 600 --provider-sort throughput --judges claude-opus-4-8,claude-fable-5`

**Experiment slug:** `E05-reasoning-rand20p` — the identifier under which this
run appears in the dashboard dropdown and in `outputs/experiments/index.json`.
Results file: `outputs/experiments/E05-reasoning-rand20p.json`.

This is the randomized-sample scale-up that
[`2026-07-03-reasoning-smoke.md`](2026-07-03-reasoning-smoke.md) queued as its
Experiment 1 (scaled to 20 prompts rather than the full 75, pending
generation parallelism). E04 established that reasoning mode is a large,
judge-robust win — on 3 unrepresentative prompts. E05 keeps E04's treatment
frozen (reasoning on, temp 0.6, 48k shared thinking+answer budget,
throughput-sorted routing, dual judges) and swaps the sample: 20 prompts drawn
by a new seeded stratified sampler (`--sample-seed`, commit `85e2325`) that
spreads the draw across every use case, instruction type, and prompt style in
the 75-prompt set. The questions: do E04's findings survive a broader,
deliberately diverse sample — and what do the categories the first-3 smoke
prompts never touched look like? Success = clean end-to-end run on the sampled
20 plus category-level results the E01–E04 sample could not produce.

## TL;DR

- **E04's reasoning-on result broadly survives the 6.7× sample**: criterion pass rates land at 79.8 / 82.7 / 84.9 (9b / 35b / 397b, Opus view) with the scaling order intact in aggregate — but compressed: the 9b→397b gap is 5.1 pp here vs 13.9 pp on E04's three prompts.
- **Full-prompt pass jumps to 15% for every model under both judges** — and it's the *same three prompts* (CIF-015, CIF-028, CIF-070) for all three models. Full-prompt success on this benchmark looks prompt-driven, not model-driven: even qwen-9b sweeps a prompt when it's sweepable at all.
- **New judge failure mode — model-level refusal**: Fable 5 deterministically refused to grade CIF-006 (benign human-physiology flashcard notes) — 7/7 calls returned `stop_reason: refusal` with zero output, including an ablation probe with a dummy candidate response, so the trigger is the prompt/criteria content, not the responses. Opus graded the same cells without issue. The pipeline previously mislabeled this `judge_parse_error`; it now records `judge_refusal` (commit `d9c6333`) and excludes such rows from the human-validation pool.
- **Implicit instructions are the hardest category** (63–68% criterion pass, Opus) — a category E01–E04 never saw (the first-3 sample was Negative/Negative/Multistep). Conditional is the easiest (88.7–90.9%). This is exactly the kind of blind spot the stratified sampler was built to expose.
- **The judges agree**: 96.8% criterion-level verdict agreement (excluding the refused prompt), Fable ~2–3 pp stricter than Opus everywhere, aggregate gap ≤ 2.9 pp per model.
- **Caveats**: E05 vs E01–E04 comparisons are cross-sample (zero prompt overlap); balanced allocation over-represents tiny use cases (three are n=1); Fable's headline numbers include a 38-criteria refused prompt scored all-FAIL (corrected views below); judge agreement with humans still unvalidated (60 rows sampled, 0 scored).

## Scope

| | |
| --- | --- |
| Sample size | 20 of 75 prompts, seeded stratified draw (seed 20260706) |
| Total criteria graded | 416 criteria × 3 models × 2 judges → 2,496 criterion verdicts over 60 responses / 120 judge calls |
| Prompt diversity | 7 of 7 use cases · 4 of 4 instruction types · 3 of 3 prompt styles |
| Candidates | qwen-9b, qwen-35b, qwen-397b (Qwen3.5 ladder via OpenRouter) |
| Judge / classifier | `claude-opus-4-8` + `claude-fable-5` judges (adaptive thinking); Opus doubles as classifier |
| Decoding | temperature 0.6 · top_p provider default · max_tokens 48 000 · reasoning **enabled** · provider sort `throughput` |

Selection mechanics (new this run): `pipeline/_select.py` draws `--limit`
prompts by balanced round-robin over `use_case` strata, greedily preferring
unseen instruction types / prompt styles within each stratum; ties fall to a
seed-shuffled order. Selection is a pure function of (dataset, limit, seed),
so every stage independently re-derives the same subset. Balanced (not
proportional) allocation is deliberate — it maximizes category spread — but it
over-weights small use cases relative to the full 75 (see Flaws).

| id | use_case | instruction_type | prompt_style | n_criteria |
| --- | --- | --- | --- | --- |
| CIF-006 | Educational Planning | Negative | Context | 38 |
| CIF-010 | Professional & Workplace | Conditional | Context | 26 |
| CIF-011 | Data Processing & Math | Implicit | Direct | 18 |
| CIF-012 | Logistics & Scheduling | Implicit | Direct | 30 |
| CIF-015 | Logistics & Scheduling | Multistep | Direct | 15 |
| CIF-018 | Logistics & Scheduling | Conditional | Context | 18 |
| CIF-021 | Professional & Workplace | Multistep | Rambling | 30 |
| CIF-023 | Educational Planning | Implicit | Direct | 13 |
| CIF-024 | Data Processing & Math | Negative | Rambling | 25 |
| CIF-027 | Health & Dietary | Negative | Direct | 18 |
| CIF-028 | Data Processing & Math | Conditional | Context | 10 |
| CIF-031 | Professional & Workplace | Negative | Direct | 12 |
| CIF-032 | Professional & Workplace | Conditional | Rambling | 17 |
| CIF-034 | Technical Design & PM | Conditional | Context | 21 |
| CIF-038 | Logistics & Scheduling | Conditional | Direct | 32 |
| CIF-043 | Data Processing & Math | Multistep | Rambling | 23 |
| CIF-046 | Logistics & Scheduling | Multistep | Context | 23 |
| CIF-058 | Creative Writing | Implicit | Context | 12 |
| CIF-070 | Educational Planning | Conditional | Rambling | 18 |
| CIF-072 | Educational Planning | Multistep | Context | 17 |

Zero overlap with CIF-001/002/003 (the E01–E04 sample), so every comparison to
earlier experiments in this report is aggregate-level across different prompt
sets, not per-prompt.

## Initial configuration (as specified)

E04's frozen parameters carried over verbatim; the only new knob is the
sampler seed. Frozen on first invocation:

```python
# runs/E05-reasoning-rand20p/experiment.json
limit         = 20
sample_seed   = 20260706     # new: stratified draw instead of first-N
max_tokens    = 48000        # thinking + answer share one budget (E04 lesson)
temperature   = 0.6          # Qwen thinking mode must not run greedy (E04 lesson)
reasoning     = True
timeout_s     = 600.0
provider_sort = "throughput" # default routing is a ~10x tok/s lottery (E04 lesson)
judges        = ["claude-opus-4-8", "claude-fable-5"]
```

Code state at launch: `85e2325` (sampler feature, committed pre-launch per
template hygiene rule 6). `9810c23` (README link, no code) landed mid-run.

## Run timeline

### Attempt 1 (2026-07-06 11:59–16:35 PT) — clean end-to-end, one new failure mode

Single attempt, no relaunches — the first experiment in this series to run the
whole pipeline in one shot. E04's three hard-won lessons (temp 0.6, 48k
budget, throughput routing) plus `caffeinate -is` from the start meant none of
E04's failure modes recurred.

- **generate** 12:01–15:13 (3h 12m, 60 responses, ~3.2 min/response): 59/60 `finish_reason: stop`; 7 transient retries (6 × qwen-9b, 1 × qwen-35b, all `RuntimeError` empty-body transients that resolved on the next attempt — the fail-fast signature, not cap-outs). One true cap-out survived to storage: qwen-9b × CIF-012 hit the 48k budget mid-answer (`finish_reason: length`) after 35.8k chars of thinking + 87.7k chars of answer whose tail is a visible restart loop ("Okay, I will write the response." … followed by another outline). Stored and graded as-is: 14/30 criteria.
- **grade** 15:14–16:27 (1h 13m, 120 cells, ~37 s/cell): 117 cells clean. Three cells — Fable × CIF-006 × every candidate — failed with what the pipeline then called `judge_parse_error` after 2 attempts each.
- **classify** 16:28–16:35 (7m, 20 calls), **validate** (60-row sample drawn), **aggregate** instant. Total 4h 36m.

### Post-run: diagnosing the CIF-006 failure (16:40–17:10 PT)

The 6 failed grade calls all reported "no parseable JSON array found". A probe
reproducing the exact judge call with raw capture showed the real cause:
**`stop_reason: refusal`, zero output text** — a model-level refusal, which the
parse layer can't distinguish from noise once the empty string reaches it. A
second probe substituting a dummy candidate response ("Sorry, I cannot help
with this request.") still refused → the trigger is CIF-006's own content (a
student's homeostatic-regulation study notes and/or its 38 criteria), not
anything the candidates wrote. 7/7 refusals across pipeline + probes;
deterministic. Opus 4.8 graded all three CIF-006 cells without complaint.

Fix (commit `d9c6333`, test-first): `grade` now records
`judge_refusal: model declined to grade this cell (stop_reason=refusal)`
without burning a second attempt; `validate` excludes refusal rows from the
human-validation pool; `aggregate` counts them in `run_notes`. The three
artifact rows were then deleted (backup kept) and the grade step re-run —
resumability regraded exactly those 3 cells (10 s; each refused again, now
recorded honestly), followed by re-`validate` (fresh 60-row pool without
refusal rows) and re-`aggregate`.

## Configuration adjustments + justifications

None mid-run — all frozen parameters survived first contact. Two code changes
bracket the run:

1. **Pre-run** — `--sample-seed` / `pipeline/_select.py` (commit `85e2325`): the experiment's motivation. Includes a fix to `aggregate`, which re-sliced `records[:limit]` independently of the other stages and would have aggregated the wrong prompts for any sampled run.
2. **Post-run** — `judge_refusal` recording (commit `d9c6333`): honest labeling of the CIF-006 cells; numbers unchanged (refused criteria still count FAIL — see Flaws for why that choice is conservative).

## Models evaluated

Same ladder as E01–E04 via OpenRouter, throughput-sorted routing:
`qwen/qwen3.5-9b`, `qwen/qwen3.5-35b-a3b`, `qwen/qwen3.5-397b-a17b`.
Reasoning traces ran 8.6k–89k chars (median ~27k–40k by model); visible
answers 242–87,676 chars. qwen-397b thinks the least (median 27k chars) and
qwen-9b the most (median 40k) — the smaller model spends more tokens to get
less far, consistent with E04's observation.

## Headline results

**Reasoning-on performance holds at 20 prompts; scaling order survives in
aggregate but compresses; full-prompt pass is prompt-driven.**

### Aggregate — criterion pass rate % (all 20 prompts)

| judge | qwen-9b | qwen-35b | qwen-397b |
| --- | --- | --- | --- |
| claude-opus-4-8 | 79.8 | 82.7 | 84.9 |
| claude-fable-5 (raw, refusal-as-FAIL) | 69.2 | 71.9 | 74.8 |
| claude-fable-5 (excl CIF-006, n=378 crit) | 76.2 | 79.1 | 82.3 |
| claude-opus-4-8 (excl CIF-006, same basis) | 79.1 | 81.5 | 83.9 |

Fable's raw row is depressed ~8 pp per model by the refused prompt's 38
criteria counting FAIL; on the comparable 19-prompt basis the two judges sit
2.9 / 2.4 / 1.6 pp apart, Fable stricter. Criterion-level verdict agreement on
the shared basis: **96.8%** (1,098 of 1,134 (prompt, criterion, model) cells).

**Full-prompt pass rate: 15.0% for every model × judge cell** — CIF-015,
CIF-028, and CIF-070 pass 15/15, 10/10, and 18/18 respectively for *all three
models* under *both judges*. No other prompt is passed by any model. Versus
E04 (397b alone at 33% on 3 prompts), the broader sample reframes the story:
when a prompt is passable, the whole ladder passes it; when it isn't, scale
doesn't rescue it. The three passable prompts are mid-length (10–18 criteria),
Conditional/Multistep, and none are Implicit.

### Vs. earlier experiments (different prompt sets — directional only)

| | E03 (3p, reasoning off) | E04 (3p, reasoning on) | E05 (20p, reasoning on) |
| --- | --- | --- | --- |
| Opus: 9b / 35b / 397b | 61.1 / 65.3 / 75.0 | 76.4 / 87.5 / 90.3 | 79.8 / 82.7 / 84.9 |
| 9b→397b gap | 13.9 pp | 13.9 pp | 5.1 pp |

E04's three prompts flattered the larger models: on the broad sample the
ladder's spread shrinks to ~5 pp. Either the first-3 prompts happened to be
scale-sensitive, or 20 diverse prompts dilute the categories where scale pays.
Per-prompt scaling is noisy either way — the aggregate order 9b ≤ 35b ≤ 397b
holds, but 7 of 20 prompts individually violate it (see Anomalies).

### Per-prompt (criteria passed, Opus view; Fable tracks within 0–2 criteria everywhere except CIF-006)

| | qwen-9b | qwen-35b | qwen-397b |
| --- | --- | --- | --- |
| CIF-006 (Edu · Negative · Context, n=38) | 33 | 36 | 36 |
| CIF-010 (Prof · Conditional · Context, n=26) | 25 | 25 | 25 |
| CIF-011 (Data · Implicit · Direct, n=18) | 13 | 14 | 14 |
| CIF-012 (Log · Implicit · Direct, n=30) | 14 | 14 | 17 |
| CIF-015 (Log · Multistep · Direct, n=15) | **15** | **15** | **15** |
| CIF-018 (Log · Conditional · Context, n=18) | 15 | 14 | 12 |
| CIF-021 (Prof · Multistep · Rambling, n=30) | 24 | 25 | 28 |
| CIF-023 (Edu · Implicit · Direct, n=13) | 9 | 10 | 10 |
| CIF-024 (Data · Negative · Rambling, n=25) | 16 | 22 | 21 |
| CIF-027 (Health · Negative · Direct, n=18) | 15 | 14 | 15 |
| CIF-028 (Data · Conditional · Context, n=10) | **10** | **10** | **10** |
| CIF-031 (Prof · Negative · Direct, n=12) | 10 | 11 | 11 |
| CIF-032 (Prof · Conditional · Rambling, n=17) | 16 | 15 | 16 |
| CIF-034 (Tech · Conditional · Context, n=21) | 20 | 19 | 19 |
| CIF-038 (Log · Conditional · Direct, n=32) | 22 | 27 | 29 |
| CIF-043 (Data · Multistep · Rambling, n=23) | 19 | 20 | 17 |
| CIF-046 (Log · Multistep · Context, n=23) | 13 | 11 | 16 |
| CIF-058 (Creative · Implicit · Context, n=12) | 10 | 9 | 9 |
| CIF-070 (Edu · Conditional · Rambling, n=18) | **18** | **18** | **18** |
| CIF-072 (Edu · Multistep · Context, n=17) | 15 | 15 | 15 |

Leads:

1. **Implicit is the hard category**: the four Implicit prompts (CIF-011/012/023/058) are the worst block on the table — 63.0 / 64.4 / 68.5% (9b/35b/397b, Opus). E01–E04 contained zero Implicit prompts; this is new coverage, and it's where all three models leave the most criteria unmet. Constraints the user implies rather than states appear to be the benchmark's real teeth.
2. **Conditional is the easy category** (88.7 / 90.1 / 90.9%) — and supplies 2 of the 3 full-pass prompts. Explicit if-then structure seems to play to reasoning mode's strengths.
3. **By use case, Logistics is hardest** (66.9 / 68.6 / 75.4%) — consistent with E01–E04, whose Logistics-heavy sample now looks like part of why absolute rates were lower there.
4. **The strongest single-model result is qwen-35b on Negative prompts** (89.3%, vs 9b's 79.6) — the mid-model matches the flagship on Negative and Data-Math (86.8 vs 81.6, where it *beats* 397b).

### Verifiability split (criterion pass %, Opus view)

| | auto | judge-tagged, subjective |
| --- | --- | --- |
| qwen-9b | 76.9 | 82.5 |
| qwen-35b | 79.4 | 85.7 |
| qwen-397b | 83.9 | 85.7 |

Subjective criteria pass at *higher* rates than mechanically-checkable ones
for every model — E04's "larger models sweep the subjective criteria"
direction persists (though no longer at 100%), and the E01 anomaly (397b
collapsing on judge-graded criteria) stays dead. The gap suggests models fail
the countable constraints (word limits, format, arithmetic) more than the
judgment calls — or that judges grade judgment calls leniently; the
still-unscored human validation sample is the arbiter.

## Experimental flaws, biases, and limitations

### Sample-size + selection effects

- 20 prompts is 27% of the set; per-category cells remain small (Implicit: 4 prompts; three use cases are n=1 — Health, Technical Design, Creative Writing). By-category numbers are directional.
- Balanced allocation deliberately over-represents tiny use cases relative to the full 75 (a singleton use case is 5% of this sample but 1.3% of the set). Aggregate rates are therefore *not* an unbiased estimate of full-75 rates; they weight use-case diversity over prevalence.
- Zero overlap with E01–E04's prompts: all cross-experiment deltas here are between different prompt sets. The E04→E05 "compression" of the scaling gap could partly be sample composition, not model behavior.

### Confounded treatment (inherited from E04)

Versus E03 (reasoning off), the treatment still folds together reasoning mode,
temp 0.0→0.6, and default→throughput routing. E05 changes nothing here — it
inherits the confound and adds a different prompt set on top. Only the
temp-only ablation (E04's queued Experiment 3) can unpick it.

### Judge biases

- **Refusal as all-FAIL is a conservative-but-wrong scoring rule**: Fable's CIF-006 refusal contributes 114 FAILs that say nothing about the candidates. The dashboard's Fable view carries a `run_notes` warning; this report's corrected (excl-CIF-006) rows are the honest Fable numbers. The deeper risk is silent: a judge with content-triggered refusals will bias any benchmark slice whose prompts trip its safety layer — and 1 in 20 sampled prompts did. Fraction of the full 75 that would refuse: unknown (see Next experiments).
- Fable is systematically ~2–3 pp stricter than Opus on identical responses. Direction is consistent across all models, so model *rankings* are judge-robust, absolute rates are not.
- Both judges are Anthropic models grading Qwen outputs with an Anthropic classifier's tags; no cross-vendor judge exists in the series yet.
- Judge agreement with humans remains unvalidated (60-row sample drawn post-regrade, 0 scored).

### Classifier biases

Opus doubles as the criterion classifier (auto vs judge-tagged verifiability).
Its tags gate the verifiability split; no second opinion exists. The
judge-tagged > auto pass-rate gap could partly be classifier miscategorization
(e.g., tagging easy-to-satisfy criteria as subjective).

### Decoding-mode caveats

- Temp 0.6 means every response is a single draw; per-prompt cell values carry sampling noise of unknown width. The 7 per-prompt scaling inversions could be draw luck. (A repeat-seed run — same everything, new draws — would bound this; queued.)
- qwen-9b × CIF-012's runaway answer (87.7k chars, capped at 48k total) shows the budget can still bind — not via long thinking this time, but via a degenerate restart-looping *answer*. Its 14/30 is partly a truncation artifact. 1 of 60 responses; the guard (`finish_reason` stored per response) makes these auditable.

### Anomaly hypotheses (not yet ruled in or out)

- **7 of 20 prompts violate the ladder order** (e.g., CIF-018: 15/14/12 — inverse; CIF-043: 19/20/17; CIF-046: 13/11/16). Candidates: single-draw noise (most likely), genuine per-prompt scale non-monotonicity, provider quantization differences across the ladder.
- **qwen-35b beats 397b on Data-Math** (86.8 vs 81.6, driven by CIF-024/043). Same candidate explanations; also plausible that 397b's shorter thinking (median 27k chars vs 38k) under-serves arithmetic-heavy prompts.
- **Fable's CIF-006 refusal**: hypothesis — the combination of detailed physiology content plus "grade this" framing trips Fable 5's dual-use safety layer (Opus 4.8 lacks the additional measures). Untested against the other 55 unsampled prompts.

### Provider-side uncontrolled variance

Throughput-sorted routing pins the fast end of the provider pool, but the pool
itself is opaque: no per-call provider capture exists yet (carried from E01,
still the prerequisite for any quantization investigation). The 7 transient
empty-body retries are consistent with provider-side flakiness at this
routing preference.

### Missing validation

`judge_validation.json`: 60 rows sampled (seed 20260101, drawn *after* the
regrade so refusal rows are excluded), 0 scored. Every number above is
judge-conditional.

### Benchmark-internal caveats (inherited)

Criteria vary 10–38 per prompt, so criterion pass rates weight long-criteria
prompts more; CIF-006 alone is 9.1% of all criteria. Some criteria are
compound ("X and Y"), graded as one unit. Unchanged from E01's observations.

## Cost & timing

| stage | wall clock | items | per-item |
| --- | --- | --- | --- |
| generate | 3h 12m | 60 responses | ~3.2 min |
| grade | 1h 13m + 10s regrade | 120 cells | ~37 s |
| classify | 7m | 20 calls | ~21 s |
| validate + aggregate | seconds | — | — |
| **total** | **4h 36m** (+ ~35m post-run diagnosis/regrade) | | |

Generation remains the bottleneck and is still sequential — 3 candidates
interleave nothing, so wall clock ≈ sum of all 60 calls. qwen-9b alone took
~2h of the 3h12m (it thinks longest and drew the most retries). Costs not
precisely tracked (per-call usage logging still missing — carried); judge-side
volume was 120 streamed calls with adaptive thinking over 32k budgets, roughly
6.7× E04's grading volume. No sleep wedge: `caffeinate -is` from launch.

## Output schema (what gets produced)

Unchanged from E04 (`meta/RESULTS_SCHEMA.md`), plus two additions this run:
`meta.config` now records `limit` and `sample_seed` (so a results file fully
determines its prompt subset), and grade rows can carry
`judge_refusal: …` reasons alongside the existing `judge_parse_error` /
`judge_truncated` vocabulary. Refusal rows count FAIL in summaries and are
flagged in `meta.run_notes`; they are excluded from the validation pool.

## Lessons / notes for the next experimenter

1. **Judges can refuse, and it looks like a parse error until you capture the raw stream.** `stop_reason` is the tell: `refusal` + empty text ≠ malformed JSON. The pipeline now separates them; any new judge model should be smoke-tested against content-sensitive prompts (health, security, chemistry) before a full run.
2. **A refusal is sticky — don't retry it.** 7/7 identical outcomes across pipeline retries and probes. The grade step now takes one attempt on refusal instead of two.
3. **The stratified sampler works and already paid for itself**: it surfaced the Implicit-instruction weakness and the Fable refusal, both invisible to four experiments of first-3 sampling. Wiring lesson: selection must be a *pure function of frozen params* — the aggregate stage had its own `records[:limit]` slice that would have silently aggregated the wrong prompts (caught by test before launch).
4. **E04's frozen lessons transferred cleanly** — first one-attempt end-to-end run in the series. The failure-mode frontier has moved from infrastructure (hangs, cap-outs, sleep wedges) to measurement (judge refusals, single-draw noise, sample composition).
5. **Degenerate answers exist at temp 0.6**: one runaway restart-looping answer (qwen-9b × CIF-012) burned the 48k budget in *answer* text. `finish_reason: length` on a response with huge `response` chars is the signature (vs E04's huge-`reasoning`, zero-content signature).

## Next experiments

### Experiment 1 — *Refusal census over the full 75*

- **What**: for each of the 75 prompts, one Fable judge call with a dummy response (the ablation template) recording `stop_reason` only. ~75 cheap calls.
- **Why**: 1 of 20 sampled prompts deterministically refuses; if several more do, Fable is structurally unusable as a solo judge on this benchmark and every Fable number needs an exclusion basis. Must precede any full-75 run that leans on Fable.
- **Cost**: ~1h wall clock, small.

### Experiment 2 — *Full-75 reasoning run* (carried from E04, still blocked)

- **What**: this run's exact treatment, `--limit 75` (no sampling needed — it's the whole set).
- **Blocker**: generation is sequential; extrapolating E05's 3.2 min/response → ~12h generate alone. Parallelize `generate.run()` first (E04 next-step #1, unchanged).

### Experiment 3 — *Repeat-draw variance bound*

- **What**: re-run E05's exact config with a different `--sample-seed`-independent axis: same 20 prompts, fresh generations (new slug, same seed, temp 0.6 does the rest). Compare per-prompt cells.
- **Why**: 7/20 scaling inversions and the 35b>397b Data-Math cell are uninterpretable without knowing single-draw noise width. Two draws give a crude bound.
- **Cost**: ≈ this run (~4.5h, ~same spend).

### Experiment 4 — *Temp-only ablation* (carried from E04)

Unchanged: reasoning off + temp 0.6 + throughput routing on a fixed sample, to
unpick the E03→E04/E05 confound.

### Experiment 5 — *CoT-vs-answer constraint coverage* (carried from E04, free)

60 stored reasoning traces now exist for diverse prompts. Check whether
constraints verified in-CoT get dropped from terse answers — qwen-9b produced
a 242-char minimum answer here; E04's 427-char observation generalizes.

## Suggested next steps

1. **Run the refusal census (Experiment 1) before anything else that uses Fable** (~1 sitting). It changes how every subsequent dual-judge experiment is read.
2. **Parallelize `generate.run()`** (~1 sitting) — carried from E04, now measured at 3h12m/20 prompts; the full-75 run stays infeasible without it.
3. **Log per-call usage + provider to a sidecar JSONL** (~1 sitting) — carried; prerequisite for cost figures and the quantization question.
4. **Hand-grade the 60-row validation sample** (~1h) — carried; now drawn from a refusal-clean pool.
5. **Commit run artifacts + this report** and sync the dashboard (done as part of this delivery).

## Judge validation

Not scored for this experiment (`status: "sampled"`, 60 rows, seed 20260101,
drawn post-regrade so no refusal rows). See Suggested next steps #4.

## Reproducibility

Commands run during this session, in order:

```
# Pre-launch: sampler feature (commit 85e2325), suite 85/85 green.
uv run pytest -q

# Launch (single attempt, ran clean end-to-end, 4h36m):
caffeinate -is uv run python main.py all --experiment E05-reasoning-rand20p \
  --limit 20 --sample-seed 20260706 --reasoning on --max-tokens 48000 \
  --temperature 0.6 --timeout 600 --provider-sort throughput \
  --judges claude-opus-4-8,claude-fable-5 --description "…"

# Post-run diagnosis of Fable × CIF-006 (raw-capture probe, then dummy-response
# ablation — both returned stop_reason=refusal, 0 chars):
#   scratchpad/cif006_probe.py, scratchpad/cif006_ablate.py

# judge_refusal handling (commit d9c6333), then surgical regrade of the 3 cells:
#   backup grades/ → delete CIF-006 rows from grades/claude-fable-5/*.jsonl
uv run python main.py grade     --experiment E05-reasoning-rand20p   # 3 cells, 10s
uv run python main.py validate  --experiment E05-reasoning-rand20p --mode sample
uv run python main.py aggregate --experiment E05-reasoning-rand20p \
  --run-report meta/2026-07-06-reasoning-rand20p.md
```

Code state: launch at `85e2325` + `9810c23` (docs-only) mid-run; analysis
artifacts produced at `d9c6333`. The prompt subset is fully determined by
`(data/complexconstraints.jsonl, limit=20, sample_seed=20260706)` via
`pipeline/_select.py`; grades backup from before the regrade surgery is
outside the repo (scratchpad) — the deleted rows differed only in `reason`
strings (`judge_parse_error` → `judge_refusal`).

## Open questions / follow-ups

- How many of the 75 prompts does Fable refuse to grade? (Experiment 1 — the answer decides whether Fable stays a standing judge.)
- Is the refusal triggered by the prompt text, the criteria list, or the judge-role framing? A 3-way ablation on CIF-006 (prompt-only / criteria-only / reworded system prompt) would localize it.
- Are the 7 per-prompt scaling inversions draw noise or real? (Experiment 3.)
- Why do all three models pass exactly the same 3 prompts and no others? Hand-inspect what separates CIF-015/028/070 from near-misses like CIF-010 (25/26 for all three models) — one shared stumble criterion on CIF-010 would be a benchmark-side finding.
- Does the Implicit-instruction weakness (63–68%) reproduce on the remaining 10 Implicit prompts in the set?
- Is qwen-35b's Negative/Data-Math strength (matching or beating 397b) stable, or an artifact of n≤4 prompts per cell?
