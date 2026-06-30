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
