# Smoke run · 2026-06-30 · `all --limit 3`

First end-to-end exercise of the v1 evaluation pipeline. Three prompts of
the Complex Constraints benchmark, three Qwen3.5 candidates via OpenRouter,
graded by `claude-opus-4-8` via Anthropic. Goal: prove the pipeline runs
cleanly on a small slice and shake out any contract / API surprises before
spending hours and dollars on the full 75-prompt run.

## TL;DR

- **Pipeline works end-to-end** — generate → grade → classify → validate(sample) → aggregate produced a `results.json` matching the FIXED CONTRACT 4 shape on the first clean run.
- **Two real surprises** surfaced and were patched in-flight: Qwen3.5 reasoning mode burns the entire token budget on internal CoT, and OpenRouter occasionally returns mid-stream-truncated JSON bodies.
- **Resumability worked correctly** — when an OpenRouter parse error crashed the run mid-generate, restart skipped completed work and only re-attempted the failing candidate.
- **Headline gap visible at n=3**: criterion pass rate 61–69 %, full-prompt pass rate **0 %**. The constraint-satisfaction gap the runbook predicted is already detectable on three prompts.

## Scope

| | |
| --- | --- |
| Sample size | 3 of 75 prompts |
| Total criteria graded | 72 (across 3 × 3 = 9 model-prompt responses) |
| Prompt diversity | 2 use cases · 2 instruction types · 3 prompt styles |
| Candidates | qwen-9b, qwen-35b, qwen-397b (Qwen3.5 size ladder, all via OpenRouter) |
| Judge / classifier | `claude-opus-4-8` (Anthropic, non-candidate family → no self-preference bias) |
| Decoding | `temperature=0` (greedy) for both candidates and judge |

The 3 prompts cover:

| id | use_case | instruction_type | prompt_style | n_criteria |
| --- | --- | --- | --- | --- |
| CIF-001 | Logistics, Scheduling & Event Planning | Negative | Context prompting | 19 |
| CIF-002 | Logistics, Scheduling & Event Planning | Negative | Direct prompting | 19 |
| CIF-003 | Data Processing, Formatting & Math | Multistep | Rambling/Stream-of-Consciousness | 34 |

This is **not a balanced cross-section** — the first three rows of the xlsx skew toward Negative instruction types. With n=3, by-category breakdowns are directional at best.

## Initial configuration (as specified)

```python
CANDIDATES = {
    "qwen-9b":   "qwen/qwen3.5-9b",
    "qwen-35b":  "qwen/qwen3.5-35b-a3b",
    "qwen-397b": "qwen/qwen3.5-397b-a17b",
}
JUDGE = "claude-opus-4-8"

# generate.py (initial)
router.chat.completions.create(
    model=model_id, temperature=0, max_tokens=4000,
    messages=[{"role": "user", "content": prompt}],
)
```

Retry predicate matched HTTP-error patterns only (`429`, `5xx`, `rate`, `timeout`, `overloaded`, `connection`).

## Run timeline

### Attempt 1 — bad first contact

Connectivity check passed for all four model IDs. The smoke kicked off `generate · qwen-9b`. After ~5 minutes with no `gen ✓` event from the monitor, the on-disk response file was checked — it contained three rows like `{"id": "CIF-001", "response": ""}`. **Empty content across all three Qwen-9b generations.**

A single diagnostic call with `max_tokens=200` returned `content="4"` for `What is 2+2?` plus a 600-char `reasoning` field in `model_extra`. A second diagnostic with the actual CIF-001 prompt at `max_tokens=16000` returned:

- elapsed 97.7 s
- `finish_reason: length`
- `content` length **0 chars**
- `reasoning` length 59,920 chars
- `usage.completion_tokens=16000`, of which `reasoning_tokens=15,423`

→ Qwen3.5 is a thinking-mode family. The full budget went to chain-of-thought; the visible answer was never emitted.

### Attempt 2 — reasoning disabled

Set `extra_body={"reasoning": {"enabled": False}}`. Quick A/B at three `max_tokens` values on CIF-001:

| max_tokens | elapsed | finish_reason | content chars | completion_tokens |
| --- | --- | --- | --- | --- |
| 4000 | 26 s | length | 9 748 | 4 000 |
| 8000 | 39 s | **stop** | 8 682 | 3 869 |
| 12000 | 133 s | stop | 20 341 | 8 892 |

At 4000 the model truncated mid-answer (unfair to the judge). At 8000 it finished naturally with a substantive response. At 12000 it rambled more (different decoding path — interesting, but not the eval condition we want).

Settled on `max_tokens=8000` with reasoning disabled.

The pipeline restarted. qwen-9b and qwen-35b completed cleanly (~30 s per call, three calls each). Mid-way through qwen-397b the run crashed with a `JSONDecodeError` raised from inside `httpx.Response.json()` — OpenRouter returned a multi-line body and the decoder failed at line 1231 char 6765. The retry predicate did not match this exception (it's a parse error, not an HTTP error), so the run aborted instead of retrying.

### Attempt 3 — broadened retry predicate

Added `JSONDecodeError`, `RemoteProtocolError`, `ReadTimeout`, and similar SDK-level transient patterns to the retry predicate in `pipeline/_io.py`. Restarted. Resumability picked up qwen-9b and qwen-35b as already-done (read existing JSONL → skip-set on id) and only re-ran qwen-397b, which completed without re-tripping the JSON error.

Downstream steps ran cleanly to completion:

```
grade · qwen-9b   : 13/19 · 17/19 · 15/34
grade · qwen-35b  :  7/19 · 13/19 · 24/34
grade · qwen-397b :  4/19 · 18/19 · 28/34
classify CIF-001 : auto=17/19 gameable=4
classify CIF-002 : auto=17/19 gameable=4
classify CIF-003 : auto=32/34 gameable=0
validate sample  : 60 rows
aggregate        : 3 prompts × 72 criteria → results.json
```

## Configuration adjustments + justifications

| Change | From | To | Why |
| --- | --- | --- | --- |
| Reasoning mode | enabled (default) | `extra_body={"reasoning": {"enabled": False}}` | At spec settings, qwen-9b emitted zero visible content because reasoning ate the budget. The verbatim spec would produce uniform empty-response failures on every complex prompt — no signal. Disabling reasoning yields the model's no-CoT instruction-following behavior, which is also closer to what users see in production deployments. |
| `max_tokens` | 4000 | 8000 | At 4000 the candidate truncated mid-answer (`finish_reason: length`). At 8000 the same call completed naturally with `finish_reason: stop`. Going higher (12k+) caused the model to ramble in a different decoding path — undesirable for reproducibility. |
| Retry predicate | HTTP-error keywords only | + `JSONDecodeError`, `RemoteProtocolError`, `ReadTimeout`, `APIConnectionError`, `APIError` | OpenRouter returned a mid-stream-truncated body that bubbled up as a parse error. Without retry, one transient bad body crashed an entire generate batch. With retry, the same condition is recoverable. |
| Request timeout | (default) | `timeout=300.0` on the OpenRouter call | The 397B model can take 90+ s per call. The default httpx timeout was too tight and risked spurious failures. |

**Not** changed:
- Judge model (`claude-opus-4-8`) — works as specified.
- Judge / classifier prompts — verbatim from FIXED CONTRACTS 2 / 3, with one obvious typo fix (`-rifiability` → `"verifiability"`).
- Anthropic-side defaults — judge with greedy decoding, system prompt from `prompts/judge.txt`.

## Models evaluated

| key | OpenRouter ID | role |
| --- | --- | --- |
| qwen-9b   | `qwen/qwen3.5-9b`        | candidate (small)  |
| qwen-35b  | `qwen/qwen3.5-35b-a3b`   | candidate (medium) |
| qwen-397b | `qwen/qwen3.5-397b-a17b` | candidate (large)  |
| judge     | `claude-opus-4-8`        | grader + classifier |

DeepSeek is wired but commented out — adds cross-family robustness later, kept off v1's first lap.

## Headline results (n=3, directional only)

### Aggregate

| metric | qwen-9b | qwen-35b | qwen-397b |
| --- | --- | --- | --- |
| criterion pass rate | 62.5 % | 61.1 % | 69.4 % |
| **full-prompt pass rate** | **0 %** | **0 %** | **0 %** |

The headline finding the runbook predicted — *high criterion-level pass, zero full-prompt pass* — is visible immediately. Models satisfy most constraints individually; **none** satisfy them all simultaneously on any of the three prompts.

### Per-prompt (criteria passed)

| | qwen-9b | qwen-35b | qwen-397b |
| --- | --- | --- | --- |
| CIF-001 (Negative · Context · Logistics) | 13/19 | 7/19 | 4/19 |
| CIF-002 (Negative · Direct · Logistics)  | 17/19 | 13/19 | 18/19 |
| CIF-003 (Multistep · Rambling · Data)    | 15/34 | 24/34 | 28/34 |

Two anti-patterns worth flagging for the full-run analysis to confirm or refute:

1. **CIF-001 inverse scaling.** On the rota problem (Context-prompted Negative), pass rate *drops* with scale (13 → 7 → 4). The 397B's failure mode in this prompt was "no final coherent rota; multiple conflicting versions, incomplete and cut off" per the judge — looks like the larger model wandered into a longer scratch-work response and never converged. May be an n=1 artifact OR may be a real interaction with `max_tokens=8000` capping a model that wants more room when given a hard problem.
2. **Data-math scaling.** On CIF-003, scaling helps cleanly (15 → 24 → 28). Multistep arithmetic seems to benefit from parameter count in the way conventional wisdom predicts.

### Verifiability split

| | auto (deterministic) | judge (subjective) |
| --- | --- | --- |
| qwen-9b   | 62 % | 67 % |
| qwen-35b  | 61 % | 67 % |
| qwen-397b | 71 % | 50 % |

qwen-397b's drop on judge-graded criteria (50 % vs 71 % on auto) is the most interesting cell here. Two non-exclusive readings: (a) the 397B's answers are correct but stated in ways the judge reads less favorably; (b) the larger model is overconfident on the subjective criteria where the judge is the only check. Phase 6 judge-validation will distinguish these once the full run completes.

**All single-cell findings here are n≤3 — they are leads to verify on the full 75, not conclusions.**

## Experimental flaws, biases, and limitations

The headline numbers above are not error bars. They are point estimates from
n=3 prompts with no human ground truth yet attached. Reading them as
conclusions would be a category error — the right framing is "leads, not
findings." This section enumerates the specific reasons.

### Sample-size + selection effects

- **n=3 means by-category cells are n=1.** Of the six summary breakdowns, every per-category column is computed from a single prompt. `by_instruction_type` has 2 Negative + 1 Multistep; `by_use_case` has 2 Logistics + 1 Data-Math; `by_prompt_style` has 1 of each of three styles. No statistical statement is licensed from these.
- **Non-random selection.** The 3 prompts are the *first three rows* of the xlsx, not a random sample. The xlsx's row order is unknown — possibly chronological, possibly grouped by difficulty, possibly ordered by author. Whatever the ordering is, the smoke inherits it as a confound.
- **Effective diversity is 2, not 3.** CIF-001 and CIF-002 are both Logistics · Negative; only the prompt_style differs. Half the smoke's signal is from one configuration, half from a second, and effectively zero from any third axis.
- **No within-run variance.** Each candidate was queried exactly once per prompt at temperature=0. There is no estimate of how stable a single generation is — even greedy decoding is sensitive to upstream provider-side conditions (load balancer routing, version drift, caching state).

### Judge biases

- **Self-preference: structurally controlled.** Opus judges Qwen candidates; Anthropic models grade non-Anthropic ones. Family-stake bias is by-design absent for v1. **Caveat:** if Claude is ever added as a candidate, this control disappears and the eval needs a different judge.
- **Verbosity bias: unmeasured.** Opus may systematically reward longer or more structured responses. CIF-001 qwen-397b's verdict ("no final coherent rota; multiple conflicting versions, incomplete and cut off") suggests the judge may have penalized output truncation per se, in addition to the underlying correctness — those two failure modes are entangled at the smoke's `max_tokens=8000` cap.
- **Position bias: unmeasured.** Criteria are presented to the judge in their xlsx-row order, every time. The judge may attend more to earlier criteria. A randomization treatment (shuffle criterion order per call with a fixed seed) would detect this and is cheap to add.
- **Format bias: unmeasured.** Markdown structure, emoji, table layouts may shift judge attention. Qwen-9b's response uses headers + emoji; qwen-35b uses headers + bold; qwen-397b's structure is less consistent across prompts.
- **Reason-quality drift.** Some judge reasons are precise ("Maria Sunday 18:00-01:00 finishes after midnight"); others are vague ("no total hours stated; response cut off"). The latter may indicate the judge punting on cases where the model didn't produce gradeable output rather than substantive judgment.

### Classifier biases

- **Same family as judge.** Opus runs both grading and classification. Any systematic Opus-side bias (verbosity preference, format preference) is correlated across both layers — a criterion the judge would grade ambiguously is likelier to also be tagged ambiguous, inflating apparent internal consistency.
- **Tags are unvalidated.** The runbook recommends hand-correcting ~20 classifier outputs ("this layer is where your judgment is the product"). Smoke did not do this. The `verifiability` and `gameable` columns surfaced in `by_verifiability` are therefore Opus's opinion, not a verified label.
- **"auto" verifiability is descriptive, not operational.** The pipeline currently grades all criteria via Opus regardless of tag. "auto"-tagged criteria are merely *labeled* as auto-checkable; they are not actually run through a deterministic verifier that could disagree with the judge.

### Decoding-mode caveats

- **Greedy + reasoning-disabled** tests one specific deployment configuration. Production users routinely invoke these models with sampling (top-p, temperature 0.7) and with reasoning enabled. Smoke results do not generalize to those modes.
- **Token-budget interactions.** The `max_tokens=8000` cap interacts with response style. Models that prefer longer scratch-work (qwen-397b on CIF-001) may exhaust the budget before producing a final answer; smaller models that converge faster may *appear* more capable on this benchmark specifically because the cap suits them.

### Anomaly hypotheses (not yet ruled in or out)

The two interesting per-category cells in Headline Results each have multiple plausible explanations. Listing them here so future runs can distinguish:

| Observation | Real? | Or: alternative explanations |
| --- | --- | --- |
| CIF-001 inverse scaling (13 → 7 → 4) | Plausibly real on Negative · Context | (a) n=1 noise; (b) reasoning-disabled mode hurts 397b more than smaller siblings; (c) 8000-token cap truncates 397b's verbose scratch-work; (d) judge format-bias against 397b's less-structured response on this prompt |
| qwen-397b verifiability drop on `judge` criteria (71 → 50 %) | Plausibly real | (a) only 6 judge-graded criteria in the smoke; (b) overlap with the CIF-001 failure mode (subjective criteria over-represented on the prompt 397b failed on); (c) judge-side overconfidence on the larger model |

### Provider-side uncontrolled variance

- **No control for time-of-day or API load.** OpenRouter routes to different upstream providers based on availability; the judge call sees Anthropic load.
- **No control for provider-side caching state.** A repeat of CIF-001 may return a cached response with different latency, with no observable signal that caching occurred.
- **The JSON corruption observed mid-smoke** (qwen-397b call) is a provider-side reliability event with no known reproduction. Future runs should log full response headers (Cloudflare ray IDs, OpenRouter request IDs) to make these investigatable.

### Missing validation

- **Phase 6 (judge validation against 60 human-graded rows) has not been run.** Without it, every aggregate number is judge-conditional with unknown agreement. This is the single largest credibility gap at the current state.
- **No control prompts.** No sanity-check criterion that should pass for any non-empty response (validates the judge isn't broken). No deliberately-failed response (validates the judge isn't lenient).
- **No second-judge cross-check.** The eval is single-judge by design for v1, but the bias-audit deliverable will want one ablation against a non-Anthropic judge.

### Benchmark-internal caveats (inherited, not specific to this smoke)

- Domain skew is benchmark-wide: 34 Logistics / 22 Data-Math / 0 Sales prompts in the full xlsx. Findings extrapolate to constraint-heavy planning, not all instruction-following.
- English-only.
- Author / creation timing of the benchmark not currently captured in run metadata.

## Cost & timing (extrapolated from smoke)

| | per call (smoke avg) | full run (×N) | total |
| --- | --- | --- | --- |
| generate (OpenRouter) | ~$0.001 / ~35 s | × 225 (75 × 3) | ~$0.20 · ~2.2 hr |
| grade (Opus judge) | ~$0.22 / ~25 s | × 225 | ~$50 · ~1.6 hr |
| classify (Opus) | ~$0.05 / ~15 s | × 75 | ~$4 · ~20 min |
| **total** | | **525 LLM calls** | **~$55 · ~4.5 hr** |

Opus dominates cost. Prompt caching (the system + per-prompt criteria are reused across the 3 candidates) could roughly halve the Opus bill; not implemented in v1.

## Output schema (what gets produced)

| path | content |
| --- | --- |
| `data/complexconstraints.jsonl` | one JSON line per benchmark prompt (id, prompt, use_case, instruction_type, prompt_style, criteria[]). Regenerated by `load`. |
| `responses/{model}.jsonl` | candidate responses (id, response). Appended by `generate`. Resumable. |
| `grades/{model}.jsonl` | judge verdicts (id, verdicts[{index, verdict, reason}]). Appended by `grade`. Resumable. |
| `outputs/criteria_tags.jsonl` | classifier tags (id, tags[{index, verifiability, gameable, reward_hack, ambiguous}]). Appended by `classify`. Resumable. |
| `outputs/judge_validation.json` | 60 fixed-seed sampled rows for human grading. Written by `validate --mode sample`. |
| `outputs/results.json` | dashboard handoff (FIXED CONTRACT 4 shape). Written by `aggregate`. |
| `outputs/run_manifest.json` | models, counts, run_date, and `judge_agreement` block (the latter merged by `validate --mode score`). |

All of `responses/`, `grades/`, `outputs/`, and `data/complexconstraints.jsonl` are gitignored — they are derivable artifacts, not source.

## Lessons / notes for the next experimenter

1. **The reasoning-mode trap is the highest-impact gotcha.** If you swap candidates to a different thinking family (DeepSeek-R1, o-series, Anthropic extended-thinking), re-confirm by inspecting `model.model_extra.reasoning` on a single call before launching a batch. The connectivity check uses `max_tokens=4` — too tight to surface the failure mode.
2. **Always run a small smoke before the full set.** Cheap to do (~10 min, ~$0.50), cripplingly expensive not to. The runbook calls this out explicitly; it earned its keep here.
3. **Resumability is load-bearing.** Two crashes mid-smoke (one diagnostic kill, one OpenRouter parse error) were free to recover from because every step skips ids already in its output file. If you change the on-disk schemas, preserve this property.
4. **Time per call varies wildly by candidate.** qwen-9b averaged ~25 s; qwen-397b averaged ~50 s and spiked to 90+ s on the hardest prompt. Plan wall-clock budgets accordingly. Sequential execution; no parallelism is built in for v1.
5. **The judge is blind by construction.** `pipeline/grade.py` builds the user message from prompt + response + numbered criteria only — the candidate's name lives in the filename, never in the message body. Don't break this when refactoring.
6. **`response_excerpt` in `judge_validation.json` is capped at 800 chars.** For longer responses, grep the matching id in `responses/{model}.jsonl` to see the full text the judge actually graded.
7. **n=3 by-category breakdowns are nearly noise.** The interesting per-category cells (CIF-001 inverse scaling, qwen-397b judge-criteria drop) are leads to verify on the full run, not findings to act on.

## Suggested next steps

A flat list of concrete actions that would advance the v1 program from where
the smoke leaves it. Each entry is scoped small enough to do in a single
sitting and has an explicit rationale + what it improves for downstream
experiments. Items are unordered by priority — pick the one whose blocking
relationship best matches your current question.

1. **Hand-grade the existing `outputs/judge_validation.json`** (~60 rows, ~1 hr of focused work) and run `validate --mode score`. **Why:** Without an agreement number, every pass-rate cell in `results.json` is judge-conditional with unknown error. This is the single highest-information-per-hour action between now and the writeup; nothing else in the bias-audit section can be quantified until this exists.
2. **Run the full 75-prompt set** with `uv run python main.py all --limit 75`. **Why:** Smoke proved the pipeline is correct; full run produces the actual deliverable. Resumable, so a mid-run crash never wastes prior work.
3. **Hand-spot-check 20 classifier outputs** in `outputs/criteria_tags.jsonl`. **Why:** The runbook explicitly calls this out as "the layer where your judgment is the product." The `verifiability` and `gameable` splits in the dashboard inherit Opus's classifier accuracy, which is currently unverified.
4. **Add a controlled randomized prompt sampler for smokes.** Replace "first N rows" with `random.Random(SEED).sample(records, N)` so a `--limit 3` smoke draws across instruction_types and use_cases. **Why:** The current smoke had 2 of 3 prompts in the same (use_case, instruction_type) cell. Future smokes will catch category-specific breakages earlier.
5. **Log per-call latency and token usage to a sidecar JSONL.** Append `{id, model, step, t_start, t_end, prompt_tokens, completion_tokens, reasoning_tokens, cost}` for every LLM call. **Why:** Cost extrapolation is currently rough; concrete timing data makes "should we parallelize generate?" decisions evidence-based instead of vibes-based, and feeds the bias-audit "decoding choice" subsection.
6. **Add Anthropic prompt caching to `pipeline/grade.py`.** Cache the judge system prompt and the (prompt + criteria) prefix that is re-used across 3 candidates per benchmark item. **Why:** Judge cost dominates the full-run bill (~$50 of ~$55). Caching the shared prefix cuts judge cost ~60% with zero behavioral change.
7. **Parallelize candidate generation.** Use a small thread pool inside `generate.run()` so the 3 Qwen candidates run concurrently per prompt. **Why:** Generation is currently sequential; the 3 candidates have no shared state and OpenRouter rate limits should easily handle parallel-3. Estimated wall-clock saving on the full run: ~1.5 hr.
8. **Capture run-config in `outputs/run_manifest.json`.** Add fields for the git commit hash, the exact `MAX_TOKENS` / `temperature` / `EXTRA_BODY` used, judge model id, and OpenRouter response-time stats. **Why:** Currently a future reader cannot tell from `results.json` whether a result came from the reasoning-disabled config or a different one. Reproducibility requires unambiguously specifying the run.
9. **Add a deterministic verifier for "auto"-tagged criteria.** A small Python check that confirms presence/absence/exact-string criteria against the response text, then compares against Opus's verdict. **Why:** Turns the `verifiability` tag from descriptive into operational. Disagreement counts between the deterministic check and the judge become a hard upper-bound on judge error for the auto-verifiable half of the eval.
10. **Save raw OpenRouter response JSON alongside the extracted content.** Write `responses/{model}.raw.jsonl` with `{id, raw_response_dict}`. **Why:** Enables post-hoc analysis of `finish_reason`, `usage`, `reasoning` fields, and provider IDs (Cloudflare ray, OpenRouter request-id) without re-running. The CIF-001 inverse-scaling anomaly is hard to diagnose without this data.
11. **Add a `--budget-dollars N` guard to `main.py`.** Before running, extrapolate from the smoke's per-call cost; abort if the projected total exceeds the budget unless overridden. **Why:** Cheap insurance against typos like `--limit 750` silently kicking off a 10× run.
12. **Add a per-prompt criterion-order randomization treatment.** For 10 prompts, shuffle criterion order in the judge user message with a fixed seed, run grading twice (original + shuffled), and report the verdict-flip rate. **Why:** Quantifies position bias in the judge — a measurement the bias-audit deliverable currently has no number for.
13. **Run a single-prompt regression smoke as a pre-commit hook.** A 30-second test using a fixed fixture-prompt that exercises load → generate (cached response) → grade → classify → aggregate. **Why:** Catches refactors that break the judge user-message format, the verdict-normalization logic, or the aggregation schema before they corrupt a real batch run.
14. **Add a known-strong reference model as a control candidate.** A frontier non-Anthropic model (DeepSeek-V4, Llama-4-405B if accessible) as a fourth candidate. **Why:** Without a reference, "qwen-397b passes 69 %" has no calibration — is that strong or weak? A control anchors the readout.
15. **Cluster judge `reason` texts to surface systematic patterns.** Group reasons by string similarity; flag clusters that contain >20 % of all FAILs. **Why:** Surfaces judge biases quantitatively — e.g., "judge fails 31 % of all rows on 'response cut off', concentrated in qwen-397b" tells you the truncation issue is grading the cap, not the model.

When an item gets done, link from its bullet to the resulting commit, run
report, or follow-up issue. Items only leave the list when finished; mark
them `[done → <hash> | <report>]` rather than deleting them, so the lineage
survives.

## Reproducibility

Commands run during this smoke session, in order:

```
# 1. Pre-flight (model IDs)
uv run python main.py connectivity

# 2. Initial smoke (failed: empty Qwen content)
uv run python main.py all --limit 3

# 3. Diagnostic: confirm the reasoning-budget trap
#    (ad-hoc — see attempt-1 narrative above)

# 4. Restarted smoke after generate.py fix (crashed: JSONDecodeError mid-qwen-397b)
uv run python main.py all --limit 3

# 5. Resumed smoke after retry-predicate broadening — succeeded end-to-end
uv run python main.py all --limit 3
```

Commit pointer for this smoke (in order applied):

```
6a7bd7e  Scaffold v1 eval pipeline structure
bc5a781  Implement six pipeline steps
eee1ccc  Replace stub main.py with CLI orchestrator
3b48c3e  Generation fixes discovered during smoke
```
