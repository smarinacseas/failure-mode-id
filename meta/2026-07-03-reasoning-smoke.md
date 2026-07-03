# Reasoning-on smoke · 2026-07-03 · `all --experiment E04-reasoning-smoke-3p --limit 3 --reasoning on --max-tokens 48000 --temperature 0.6 --timeout 600 --provider-sort throughput`

**Experiment slug:** `E04-reasoning-smoke-3p` — the identifier under which this
run appears in the dashboard dropdown and in `outputs/experiments/index.json`.
Results file: `outputs/experiments/E04-reasoning-smoke-3p.json`.

This run is the reasoning-mode ablation that
[`2026-06-30-smoke-test.md`](2026-06-30-smoke-test.md) deferred when it disabled
Qwen3.5's thinking mode to stop the token budget vanishing into invisible
chain-of-thought. E03 (`E03-judge-compare-3p`) established the dual-judge,
reasoning-off baseline on 3 prompts; E04 re-runs the identical prompts,
candidates, and judges with candidate reasoning enabled. The question: does
letting the models think before answering close any of the
constraint-satisfaction gap — and does the answer survive two independent
judges? Success = non-empty reasoning-mode generations end-to-end plus an
apples-to-apples pass-rate comparison against E03. Getting there took three
configurations and surfaced four distinct operational failure modes, all
documented below.

## TL;DR

- **Reasoning mode is a large, judge-robust win**: criterion pass rate +13.9 to +29.2 points on every (model × judge) cell vs E03. qwen-35b gains most (65.3→87.5 Opus; 56.9→86.1 Fable).
- **First full-prompt pass in the benchmark's history**: qwen-397b satisfied all 19/19 criteria on CIF-002 — confirmed independently by both judges. Every prior run (E01–E03) was 0% everywhere.
- **Greedy decoding + thinking mode is a trap, not a slowdown**: at `temperature=0.0` every generation looped in its chain-of-thought until it burned the full budget and emitted zero visible content. Four launch attempts on 2026-07-02 appeared "hung" for exactly this reason. Qwen's model card explicitly forbids greedy decoding in thinking mode; temp 0.6 fixed it.
- **Two more infrastructure lessons**: OpenRouter's default routing is a ~10× throughput lottery (20 vs 164 tok/s measured), fixed with a new frozen `provider_sort=throughput` knob; and a MacBook sleeping mid-run wedges a streamed judge call on a dead socket forever — the run lost 10.3 hours overnight and now runs under `caffeinate -is`.
- **Caveats**: n=3 prompts; E04 vs E03 folds together three changes (reasoning on, temp 0.0→0.6, default→throughput routing), so deltas attribute to "reasoning mode" loosely, not strictly; judge agreement is unvalidated against humans (60-row sample drawn, not yet scored).

## Scope

| | |
| --- | --- |
| Sample size | 3 of 75 prompts (same first-3 as E01–E03) |
| Total criteria graded | 72 per judge · 2 judges → 216 grade cells over 9 responses |
| Prompt diversity | 2 use cases · 2 instruction types · 3 prompt styles |
| Candidates | qwen-9b, qwen-35b, qwen-397b (Qwen3.5 ladder via OpenRouter) |
| Judge / classifier | `claude-opus-4-8` + `claude-fable-5` judges (adaptive thinking); Opus doubles as classifier |
| Decoding | temperature 0.6 · top_p provider default (1.0) · max_tokens 48 000 · reasoning **enabled** · provider sort `throughput` |

Non-representative sample, inherited from E01: first-3 rows of the xlsx, 2 of 3
prompts share (Logistics · Negative). All by-category cells are n=1.

| id | use_case | instruction_type | prompt_style | n_criteria |
| --- | --- | --- | --- | --- |
| CIF-001 | Logistics, Scheduling & Event Planning | Negative | Context prompting | 19 |
| CIF-002 | Logistics, Scheduling & Event Planning | Negative | Direct prompting | 19 |
| CIF-003 | Data Processing, Formatting & Math | Multistep | Rambling/Stream-of-Consciousness | 34 |

## Initial configuration (as specified)

Inherits E03's frozen parameters (see `runs/E03-judge-compare-3p/experiment.json`)
with reasoning switched on and the budget raised — the first launch attempts froze:

```python
# runs/E04-*/experiment.json (attempts 1–4, all failed)
max_tokens   = 32000        # raised from E03's 8000: thinking + answer share one budget
temperature  = 0.0          # E03 default, inherited *by omission* — the bug
reasoning    = True         # the experimental variable
timeout_s    = 600.0
judges       = ["claude-opus-4-8", "claude-fable-5"]
limit        = 3
# generate.py at the time returned message.content only and raised
# "empty completion content" (retriable, 5 attempts) on blank output.
```

## Run timeline

### Attempts 1–4 (2026-07-02 22:04–22:44) — every generation "hangs"; four slugs abandoned

Four launches under three slugs (`E04-reasoning-3p`, `E04-dual-judge-thinking-3p`
×2, `E04-reasoning-smoke-3p` v1), all with the config above. Symptom: the
generate stage never completed a single item. Logs show `retry
openrouter:qwen/qwen3.5-9b (RuntimeError)` at 3–20-minute intervals — each
attempt burned the full 32k budget on chain-of-thought and returned empty
`content`, tripping the empty-completion guard, which retried… for up to 5 ×
~15 min per item. Two abandoned launches kept running as orphaned processes for
~3 hours (see Lessons), one of which eventually produced a single CIF-001
response on its 4th attempt after ~45 minutes — proof that greedy+thinking
occasionally converges, at unusable latency.

Diagnosis: Qwen's model card states thinking mode must **not** use greedy
decoding ("can lead to endless repetitions"); recommended thinking-mode sampling
is temp 0.6 / top_p 0.95. The 2026-06-30 smoke had already recorded the
signature (59 920 reasoning chars, 0 content chars, `finish_reason: length`)
without connecting it to temperature. `temperature=0.0` came along silently as
the registry default.

### Attempt 5 (22:55–23:52) — temp 0.6: CoT healthy, but 32k caps out twice

Relaunched frozen at temp 0.6 / 32k / 600s. qwen-9b × CIF-001: attempt 1 ran
24.8 min → empty at cap; attempt 2 ran 28.3 min → empty at cap. Three streamed
diagnostic probes established the facts:

1. **CoT is healthy at 0.6** — live tail showed genuine rota-solving (staff-hour arithmetic, constraint checks), zero repetition loops.
2. **Default routing drew a ~20–25 tok/s provider** (~9 500 CoT chars in 120 s); a full 32k-token generation at that rate is ~25 min — matching the observed attempt durations exactly. A second probe drew a ~110 tok/s provider (20 002 chars in 45 s): the provider lottery spans ~10×, consistent with the orphan that burned 32k in 3.2 min (~164 tok/s).
3. **`reasoning.max_tokens` is ignored for Qwen on OpenRouter** — a 2 000-token thinking bound blew past 5 000 tokens without wrapping. The total `max_tokens` is the only cap lever. A trivial-prompt call confirmed providers accept `max_tokens=48000` (`finish_reason: stop`, 3.4 s).

The 600-second request timeout never fired on the 25-minute calls: chunks
trickle in and reset httpx's read timer, so `timeout_s` only catches dead
connections, not slow generations.

### Attempt 6 (23:54–00:48, wedged; resumed 11:07–11:17) — 48k + throughput routing: clean end-to-end

Two changes, both frozen: `max_tokens` 32 000 → 48 000, and a new
`provider_sort=throughput` parameter (plumbed through `RunConfig.extra_body` as
OpenRouter's `provider: {"sort": "throughput"}`).

All 9 generations completed in 44.4 min with `finish_reason: stop`, zero
truncations. Thinking traces ran 26 894–63 408 chars (~7k–16k tokens); visible
answers 427–6 292 chars. One transient empty-body retry (qwen-35b CIF-001)
resolved in a minute — distinct signature from a cap-out (fails fast, tiny
`reasoning_chars`).

Grading proceeded at ~75 s/cell for 7 of 18 cells, then the MacBook slept at
~00:48. On wake the in-flight streamed Opus call sat on a half-open TCP socket
that delivers neither bytes nor EOF — no timeout, no exception, heartbeat
frozen for 10.3 hours. Killed and re-ran the same slug under `caffeinate -is`:
resumability skipped all completed work, the remaining 11 cells graded in 8.8
min (~48 s/cell), classify took 46 s, validate + aggregate were instant.
Complete at 11:17:19.

## Configuration adjustments + justifications

| Change | From | To | Why |
| --- | --- | --- | --- |
| `temperature` | 0.0 | 0.6 | Vendor-mandated for thinking mode: greedy decoding degenerates into endless CoT repetition. Observed: 4+ full-budget zero-content generations at 0.0; healthy convergent CoT at 0.6. |
| `max_tokens` | 8 000 (E03) → 32 000 (attempts 1–5) | 48 000 | Thinking + answer share one budget and `reasoning.max_tokens` is ignored for Qwen. At 32k, qwen-9b × CIF-001 capped out mid-thought twice consecutively; successful traces later measured up to ~16k thinking tokens, i.e. the thinking-length distribution is heavy-tailed. 48k clears the observed tail. |
| `provider_sort` (new frozen param) | — (OpenRouter default routing) | `throughput` | Default routing repeatedly drew ~20 tok/s providers → 25–40-min full-budget calls; measured fast providers run 110–164 tok/s. New knob in `run_config.py`/`main.py`, recorded in the freeze and manifest; back-compat default `None` for E01–E03 freezes. |
| response record schema | `{id, response}` | `{id, response, finish_reason, reasoning?}` | `generate.py` now stores the chain-of-thought and finish_reason per response — post-hoc failure-mode analysis without re-running. Judges still see only the visible answer (`grade.py` reads `response` alone). Empty-completion errors now report `finish_reason` + `reasoning_chars` so a budget cap-out is distinguishable from a transient blank body in one log line. |

**Not** changed:

- **`top_p`** — Qwen recommends 0.95 for thinking mode; we ran the provider default (1.0). Temperature was the load-bearing fix; adding a second sampling deviation mid-firefight would have muddied attribution further. Candidate for a controlled ablation.
- **Judge configuration** — identical to E03 (both judges, adaptive thinking, 32k judge budget, same prompt SHAs `c3ed3a2cefcb` / `88d130c223aa`), deliberately, so the judge axis stays comparable.
- **Prompt sample** — same first-3 rows as E01–E03, non-random, kept for direct comparability.
- **`timeout_s`** — left at 600 despite being demonstrably toothless against slow-trickling generations (see Lessons #4); fixing timeout semantics is infrastructure work, not an experiment knob.

## Models evaluated

| key | provider ID | role |
| --- | --- | --- |
| qwen-9b | `qwen/qwen3.5-9b` | candidate (small, dense) |
| qwen-35b | `qwen/qwen3.5-35b-a3b` | candidate (medium, MoE 3B active) |
| qwen-397b | `qwen/qwen3.5-397b-a17b` | candidate (large, MoE 17B active) |
| judge 1 | `claude-opus-4-8` | grader + classifier |
| judge 2 | `claude-fable-5` | grader |

Same lineup as E03; no additions or removals.

## Headline results

**Reasoning mode lifts criterion pass rate by double digits on every cell, under
both judges independently** (n=3 prompts — directional, not conclusive).

### Aggregate — criterion pass rate % (E03 reasoning-off → E04 reasoning-on)

| judge | qwen-9b | qwen-35b | qwen-397b |
| --- | --- | --- | --- |
| claude-opus-4-8 | 61.1 → **76.4** (+15.3) | 65.3 → **87.5** (+22.2) | 75.0 → **90.3** (+15.3) |
| claude-fable-5 | 59.7 → **75.0** (+15.3) | 56.9 → **86.1** (+29.2) | 75.0 → **88.9** (+13.9) |

**Full-prompt pass rate**: qwen-397b 0% → **33%** (CIF-002 at 19/19, both
judges — the first full-prompt pass across E01–E04). qwen-9b and qwen-35b
remain 0%, so the smoke-test's headline "high criterion pass, zero full-prompt
pass" survives reasoning mode everywhere except at the largest scale on the
easiest prompt.

Judge agreement tightened: E03's Opus−Fable gap reached 8.4 pp (qwen-35b);
E04's largest gap is 1.5 pp. Cleaner answers appear to leave less room for
grader interpretation.

### Per-prompt (criteria passed, E03 → E04)

Opus view (Fable tracks within 0–2 criteria on every cell):

| | qwen-9b | qwen-35b | qwen-397b |
| --- | --- | --- | --- |
| CIF-001 (Negative · Context · Logistics) | 14 → 12 ↓ | 9 → 16 | 12 → 16 |
| CIF-002 (Negative · Direct · Logistics) | 13 → 15 | 17 → 18 | 16 → **19/19** |
| CIF-003 (Multistep · Rambling · Data-Math) | 17 → 28 | 21 → 29 | 26 → 30 |

Leads:

1. **The scaling order restores under reasoning** — 9b ≤ 35b ≤ 397b on every prompt. E01's CIF-001 inverse-scaling anomaly (13 → 7 → 4) is gone; its anomaly table listed "(b) reasoning-disabled mode hurts 397b more than smaller siblings" as an alternative explanation, and E04 is evidence for exactly that.
2. **qwen-9b × CIF-001 is the sole decline** (−2 under both judges). Single sample at temp 0.6 — see Anomaly hypotheses.
3. **Largest gains concentrate on CIF-003 (Multistep · Data-Math)** — +9 to +11 criteria for the two smaller models. Consistent with thinking helping arithmetic/multistep work most.
4. **qwen-9b's CIF-002 answer is 427 chars after 58 904 chars of thinking** — reasoning mode shifted answer style sharply terse, and the terse answer still gained (+2). Worth reading the stored CoT before the full run to check whether constraint verification is happening in-CoT and being summarized away.

### Verifiability split (criterion pass %, E04)

| | auto (Opus / Fable) | judge-tagged, subjective (Opus / Fable) |
| --- | --- | --- |
| qwen-9b | 79.4 / 77.8 | 55.6 / 55.6 |
| qwen-35b | 85.7 / 84.1 | **100.0 / 100.0** |
| qwen-397b | 88.9 / 87.3 | **100.0 / 100.0** |

E01's most interesting cell — qwen-397b dropping to 50% on judge-graded
criteria — is fully reversed: both larger models now sweep the subjective
criteria under both judges. Only 9 criteria per response are judge-tagged, so
these are small-n cells; but the double-judge unanimity makes "judge quirk" a
weaker explanation than it was in E01.

## Experimental flaws, biases, and limitations

### Sample-size + selection effects

Unchanged from E01–E03 and inherited deliberately: n=3 first-rows sample,
effective diversity ~2 (CIF-001/002 share use_case × instruction_type), every
by-category cell n=1, one generation per (model, prompt) — **and E04's
temperature 0.6 makes single generations noisier than E03's greedy ones**: some
per-cell deltas (notably qwen-9b CIF-001's −2) are within plausible resampling
noise. E03 vs E04 deltas are also cross-sample comparisons of *different*
random draws, not paired responses.

### Confounded treatment

The headline E03→E04 delta folds together **three** changes: reasoning on,
temperature 0.0→0.6, and default→throughput provider routing (fast providers
may serve different quantizations). The +14 to +29 pp magnitude dwarfs
plausible sampling/quantization effects, but strict attribution to "reasoning
mode" requires the ablations in Next experiments. This is the single biggest
interpretive caveat in this report.

### Judge biases

Self-preference: structurally controlled (Anthropic judges, non-Anthropic
candidates), unchanged. Verbosity/format/position bias: unmeasured, unchanged —
but note reasoning mode changed candidate answer *style* (terser, more
structured), so any judge style-bias now interacts with the treatment.
Judge-agreement tightening (max gap 8.4→1.5 pp) is measured and reassuring but
is inter-judge agreement, not accuracy.

### Classifier biases

Unchanged from E03: Opus classifies and grades; tags unvalidated by hand;
`auto` verifiability remains descriptive (no deterministic checker runs).
Criteria tags were re-generated for this run (fresh classify pass over the same
3 prompts).

### Decoding-mode caveats

This configuration (thinking on, temp 0.6, top_p 1.0, 48k cap, throughput-sorted
providers) generalizes to *reasoning-enabled sampled deployments*, not to E03's
greedy no-CoT mode — the two runs bracket the deployment space rather than
either being "the" number. No E04 response hit the token cap (all
`finish_reason: stop`), but two 32k attempts did during config search;
thinking-length is heavy-tailed and 48k clears only the *observed* tail.

### Anomaly hypotheses (not yet ruled in or out)

| Observation | Real? | Alternative explanations |
| --- | --- | --- |
| qwen-9b CIF-001 decline (14→12 / 13→12) | Uncertain | (a) sampling noise at temp 0.6, single draw; (b) reasoning genuinely hurts the small model on hard combinatorial scheduling (overthinking); (c) judge penalizing its new answer format |
| qwen-35b/397b sweep judge-tagged criteria at 100/100 | Plausibly real | (a) only 9 subjective criteria per response; (b) reasoning-mode answers are more structured → judge reads them more charitably (style bias, not substance) |
| CIF-002 terse-answer effect (427 chars, +2 criteria) | Style shift is real | Whether terseness *helps* (less surface area to violate constraints) vs *risks* under-specification is unresolved — needs the stored CoT read and more prompts |
| Judge-agreement tightening (8.4→1.5 pp max gap) | Plausibly real | (a) n=216 cells; (b) E03's disagreement may have concentrated in response-quality ranges reasoning mode happens to vacate |

### Provider-side uncontrolled variance

Larger than in prior runs and now partially *chosen*: `provider_sort=throughput`
pins routing to the fast end of the pool, which may correlate with quantization
choices. Which concrete provider served each call is still not captured
(request IDs / provider names not logged — carried forward as a suggested next
step). Observed reliability events this run: one transient empty completion
(retried clean), two 32k cap-outs during config search, one 10.3-hour wedge
caused by client-side sleep, and the 600s timeout demonstrably not bounding
slow-trickle calls.

### Missing validation

Phase-6 human scoring not run: `runs/E04-reasoning-smoke-3p/judge_validation.json`
holds the fixed-seed 60-row sample, unscored (`validation.status: "sampled"`,
`agreement_pct: null`). Every pass-rate above is judge-conditional with unknown
error. No control prompts, no deterministic verifier for auto-tagged criteria.

### Benchmark-internal caveats (inherited)

Unchanged from `2026-06-30-smoke-test.md`: domain skew (Logistics-heavy),
English-only, first-3-rows ordering confound.

## Cost & timing

Wall-clock measured; dollar figures are estimates (per-call usage/cost logging
is still an open next-step — carried since E01).

| | per call (this run avg) | this run | full-75 extrapolation |
| --- | --- | --- | --- |
| generate (OpenRouter, reasoning on) | ~4.9 min (range 1.4–11.3) · ~$0.01–0.05 | 9 calls · 44 min · <$1 | × 225 ≈ **18 hr sequential** · ~$3–10 |
| grade (2 judges, thinking on) | ~48–75 s · ~$0.25–0.50 | 18 cells · ~18 min · ~$5–9 | × 450 ≈ 7.5 hr · ~$110–225 |
| classify (Opus) | ~15 s · ~$0.05 | 3 calls · 46 s | × 75 ≈ 20 min · ~$4 |
| **total** | | **30 LLM calls · ~$6–12** | **~26 hr sequential · ~$120–240** |

Failed attempts (4 greedy launches incl. ~3 orphan-hours, two 32k cap-outs,
three diagnostic probes) burned an additional ~1M candidate-side tokens ≈
$1–3 — the cost of the config search, not the experiment.

The full-75 extrapolation says **sequential generation is now the wall-clock
bottleneck** (reasoning inflates per-call time ~10×). Parallelizing generate
(suggested since E01) graduates from nice-to-have to prerequisite before the
full reasoning run.

## Output schema (what gets produced)

Deliverable matches [`meta/RESULTS_SCHEMA.md`](RESULTS_SCHEMA.md) at
`schema_version 3.0` — no schema changes. Two run-specific notes:

| path | content |
| --- | --- |
| `runs/E04-reasoning-smoke-3p/responses/{model}.jsonl` | **extended records**: `{id, response, finish_reason, reasoning}` — chain-of-thought (27k–63k chars each) stored for post-hoc analysis; all downstream readers consume `response` only |
| `runs/E04-reasoning-smoke-3p/experiment.json` | freeze now includes `provider_sort` (older freezes load with implicit `null`) |
| `outputs/experiments/E04-reasoning-smoke-3p.json` + `index.json` | deliverable + dashboard index entry (synced to `dashboard/` by `scripts/dashboard_sync.py` at aggregate time) |
| `runs/E04-reasoning-smoke-3p/run_manifest.json` | config snapshot incl. `extra_body` with reasoning + provider blocks |

## Lessons / notes for the next experimenter

1. **Thinking mode changes the *decoding contract*, not just the output length.** Enabling reasoning while inheriting greedy temp=0.0 produced a 100%-failure mode that presents as a hang (long call → empty content → retriable error → another long call). Check the vendor's sampling guidance whenever toggling reasoning on any model family.
2. **"Empty completion" now tells you which failure it was.** `finish_reason=length` + large `reasoning_chars` = budget cap-out (deterministic-ish, consider raising the cap); `finish_reason=stop`-ish + zero chars = transient provider blank (retry will fix). Before this run the two were indistinguishable in logs.
3. **The provider lottery is a first-order effect for reasoning workloads.** 20 vs 164 tok/s is the difference between a 25-minute and a 3-minute full-budget call. `--provider-sort throughput` is now a frozen, manifest-recorded knob — but nobody logs *which* provider actually served a call yet.
4. **`timeout_s` does not bound slow generations** — httpx read timeouts reset on every trickled chunk. It fires on dead sockets only… and not even those when the client machine slept mid-stream (half-open socket, blocked read, no EOF): that wedge cost 10.3 hours. **Run anything long under `caffeinate -is`** and treat a stale `progress.json` heartbeat (older than ~timeout+60s) as wedged: kill and re-run the slug — resumability makes this free.
5. **Kill orphaned launches before starting a new one** (`ps aux | grep main.py`). Abandoned attempts keep grinding for hours, recreate deleted `runs/<slug>/` dirs, clobber the shared `outputs/progress.json`, and would eventually overwrite `outputs/results.json` and append junk to the dashboard index.
6. **Store the CoT.** It costs nothing (fields are additive; readers unaffected) and converts "why did the terse answer pass?" from speculation into a grep. The 9 stored traces are the first raw material for the failure-mode-ID objective of this whole project.
7. **Freeze-on-first-run cuts both ways.** It reproducibly captured every bad configuration (useful!) but silently inherits registry defaults you didn't think about (`temperature` here). When adding a mode-changing flag, review the *whole* default set it composes with.

## Next experiments

### Experiment 1 — *Full-75 reasoning run*

- **Type**: confirmatory (of E04's effect at scale)
- **Hypothesis**: Reasoning-on criterion pass rate exceeds the reasoning-off full-run baseline by ≥8 pp overall, with the largest per-category gain on Multistep.
- **Prior**: E04 smoke: +14 to +29 pp, Multistep (CIF-003) gains largest for small/medium models. Expect shrinkage at n=75 but same sign everywhere; 397b full-prompt pass rate 5–15% (vs 0% baseline).
- **What would change my mind**: overall delta <4 pp or negative on any model → the smoke's effect was prompt-selection or single-sample artifact.
- **Operationalization**: needs the reasoning-off full-75 baseline run first (still not done), then `all --experiment E05-reasoning-75p --limit 75 --reasoning on --max-tokens 48000 --temperature 0.6 --provider-sort throughput`. **Blocked on parallelizing generate** (~18 hr sequential otherwise) and should run under `caffeinate`.
- **Cost / wall-clock**: ~$120–240, ~26 hr sequential / ~8 hr with 4-way generate parallelism.
- **Priority**: H — this is the deliverable the smoke exists to de-risk.
- **Depends on**: reasoning-off 75p baseline; generate parallelism; ideally per-call usage logging for real cost numbers.

### Experiment 2 — *qwen-9b CIF-001 decline: noise or real?*

- **Type**: exploratory (variance measurement)
- **Hypothesis**: The −2-criteria decline is within single-sample noise at temp 0.6.
- **Prior**: k=5 resamples of (qwen-9b, CIF-001) will span ≥4 criteria of range; E03's greedy 14/19 will fall inside the E04 resample distribution.
- **What would change my mind**: all 5 resamples ≤12/19 → reasoning genuinely hurts the small model here (overthinking a hard combinatorial problem) — a publishable failure mode.
- **Operationalization**: small script over the existing pipeline pieces (generate k samples into a sidecar dir, grade with one judge); no framework changes needed.
- **Cost / wall-clock**: 5 generations (~30 min) + 5 Opus grade calls (~$1.50), same evening.
- **Priority**: M — cheap, and it calibrates how much to trust *every* single-sample cell in E04.
- **Depends on**: nothing; can run today.

### Experiment 3 — *Disentangle the treatment: temp-only ablation*

- **Type**: ablation
- **Hypothesis**: Temperature 0.0→0.6 alone (reasoning off) moves criterion pass rate by <3 pp on these 3 prompts — i.e., E04's gains belong to reasoning, not sampling.
- **Prior**: near-zero mean shift with ±2-criteria per-cell noise; E03's greedy results sit inside the temp-0.6 resample band.
- **What would change my mind**: a consistent ≥5 pp shift from temperature alone → E04's headline overstates the reasoning effect and the report's confound caveat becomes its headline.
- **Operationalization**: `all --experiment E06-temp06-3p --limit 3 --reasoning off --max-tokens 8000 --temperature 0.6` (default routing, matching E03 elsewhere). Compare three-way: E03 vs E06 (temp effect) vs E04 (temp+reasoning).
- **Cost / wall-clock**: one E03-sized run: ~30 min, ~$5–8.
- **Priority**: H — directly de-confounds the headline claim of this report.
- **Depends on**: nothing; can run today.

### Experiment 4 — *Provider-pin quantization check*

- **Type**: ablation
- **Hypothesis**: Pinning a single named provider vs `sort=throughput` changes per-model criterion pass rate by <3 pp — routing is a latency choice, not a capability treatment.
- **What would change my mind**: ≥5 pp shift on any model → provider/quantization variance is result-relevant and every cross-run comparison needs provider capture.
- **Operationalization**: requires logging which provider served each call first (OpenRouter returns it in the response; add to the stored record), then re-run E04's config with `provider: {"order": [<named>]}` via a new experiment slug.
- **Cost / wall-clock**: one E04-sized run (~$6–12, ~1.5 hr).
- **Priority**: M — becomes H before any cross-time-period comparison is published.
- **Depends on**: provider-name capture in `generate.py`.

### Experiment 5 — *CoT-vs-answer constraint coverage (uses stored reasoning, no new calls)*

- **Type**: exploratory
- **Hypothesis**: Failed criteria are disproportionately ones never mentioned in the CoT (dropped constraints), rather than ones considered and then violated.
- **Prior**: ≥60% of failed criteria have no CoT mention; qwen-9b's terse CIF-002 answer discusses more constraints in-CoT than it states in-answer.
- **What would change my mind**: most failures ARE discussed in-CoT → the failure mode is execution/summarization, not attention — a different fix (answer-format prompting) than "think more."
- **Operationalization**: offline analysis over `responses/*.jsonl` reasoning fields + grade verdicts; string/embedding match criteria against CoT.
- **Cost / wall-clock**: zero API cost; an afternoon of analysis.
- **Priority**: M — this is the "failure-mode ID" thesis of the project made concrete.
- **Depends on**: nothing; data is on disk now.

## Suggested next steps

1. **Parallelize `generate.run()`** (small thread pool over candidates, ~1 sitting). **Why:** reasoning made generation the bottleneck (~18 hr sequential for 75p); 3–4× parallelism makes Experiment 1 feasible in one overnight run. [carried from E01 list, now blocking]
2. **Log per-call usage + provider name to a sidecar JSONL** (~1 sitting). **Why:** cost figures in this report are estimates; provider capture is a hard prerequisite for Experiment 4 and for investigating any future anomaly that smells provider-shaped. [carried from E01, upgraded in urgency]
3. **Hand-grade `judge_validation.json` for E04** (60 rows ≈ 1 hr) and run `validate --mode score`. **Why:** every number above is judge-conditional; E04's cleaner answers may also be easier for humans to grade, making this the cheapest agreement number yet. [carried from E01]
4. **Add a wall-clock deadline to candidate calls** (wrap the OpenRouter call in a hard timer or switch to streaming with an elapsed-time guard, ~1 sitting). **Why:** `timeout_s` provably does not bound slow-trickle calls; the 25-min cap-out attempts and the 10.3-hour sleep wedge were both invisible to it.
5. **Document the `caffeinate -is` requirement in the README run instructions** (5 min). **Why:** the sleep wedge will recur on every laptop-hosted long run; one sentence prevents 10-hour losses.
6. **Commit this state before the next experiment** (code changes: `pipeline/generate.py`, `pipeline/run_config.py`, `main.py`, test mocks; plus this report and the E04 dashboard sync). **Why:** template hygiene rule 6 — the E04 result must stay reproducible from a tagged commit; the run manifest currently records `f92f858 (dirty)`.

## Judge validation

Not run for this experiment (`status: "sampled"`, 60 rows drawn with seed
20260101, 0 scored). See Suggested next steps #3.

## Reproducibility

Commands run during this session, in order (failures included — they are the story):

```
# Attempts 1–4 (2026-07-02 evening): reasoning on, temp 0.0 inherited — all "hung", slugs abandoned:
uv run python main.py all --experiment E04-reasoning-3p --limit 3 --reasoning on --max-tokens 32000 --timeout 600 --description "…"
uv run python main.py all --experiment E04-dual-judge-thinking-3p --limit 3 --max-tokens 32000 --reasoning on --judges claude-opus-4-8,claude-fable-5 --description "…"   # ×2
uv run python main.py all --experiment E04-reasoning-smoke-3p --limit 3 --reasoning on --max-tokens 32000 --timeout 600 --description "…"   # v1, Ctrl-C'd
# Orphaned processes from the above killed ~23:10; their recreated runs/ dirs removed.

# Attempt 5: temp fixed, 32k — two cap-outs on qwen-9b × CIF-001; killed; dir wiped:
rm -rf runs/E04-reasoning-smoke-3p
uv run python main.py all --experiment E04-reasoning-smoke-3p --limit 3 --reasoning on --max-tokens 32000 --temperature 0.6 --timeout 600 --description "…"

# Diagnostic probes (ad-hoc, streamed): CoT health + tok/s at default routing;
# reasoning.max_tokens honored? (no); max_tokens=48000 accepted? (yes).

# Attempt 6 (final): 48k + throughput routing; wedged by system sleep mid-grade at 00:48;
rm -rf runs/E04-reasoning-smoke-3p
uv run python main.py all --experiment E04-reasoning-smoke-3p --limit 3 --reasoning on --max-tokens 48000 --temperature 0.6 --timeout 600 --provider-sort throughput --description "…"
# resumed 11:07 under caffeinate — completed 11:17:
caffeinate -is uv run python main.py all --experiment E04-reasoning-smoke-3p
uv run python main.py aggregate --experiment E04-reasoning-smoke-3p --run-report meta/2026-07-03-reasoning-smoke.md
```

Code state: `f92f858` ("Fixed overlap") **plus uncommitted changes** made for
this run — `pipeline/generate.py` (CoT/finish_reason capture, enriched
empty-completion error), `pipeline/run_config.py` + `main.py`
(`provider_sort` frozen param + `--provider-sort` flag), and
`tests/test_generate_monitor.py` (mock updates; 71/71 passing). The run
manifest records `dirty: true`; Suggested next steps #6 is the fix.

## Open questions / follow-ups

- Does reasoning mode's terse-answer style (427-char CIF-002 answer) generalize, and does it help (smaller violation surface) or hurt (under-specification) on prompts whose criteria demand completeness?
- Why does CIF-001 stay hard for everyone (best: 16/19)? Its 19 criteria include a genuinely over-constrained-feeling rota; is there a satisfying assignment at all? A hand-solve (or SAT-style check) of CIF-001 would tell us whether 19/19 is achievable.
- Is thinking length predictive of pass rate per prompt (within-model), or is it pure difficulty signal? Free with stored CoT.
- The 32k cap-outs: were those two draws unusually long thinkers, or does qwen-9b's CIF-001 thinking distribution mostly exceed 32k? (Five failed/capped traces exist across attempts if OpenRouter logs are consulted; locally we only know durations.)
- OpenRouter non-streaming semantics: is there any server-side keep-alive on non-streamed completions that our client-side wall-clock guard (next steps #4) should account for?
