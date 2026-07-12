# `results.json` — deliverable schema

Every experiment produces one JSON file (`outputs/experiments/<slug>.json`)
with the shape defined here. The **ConstraintLens dashboard**
(`dashboard/index.html`, unpacked from the Claude Design bundle) binds
against this schema once and renders any run — its Logic reads
`./runs.json` on load, then each run's JSON on selection.
`scripts/dashboard_sync.py` mirrors every experiment into `dashboard/`
and rebuilds `dashboard/runs.json`; `pipeline/aggregate.py` calls it
automatically for tagged runs.

The compact registry at `outputs/experiments/index.json` is a legacy /
back-compat index for consumers other than the dashboard.

Current schema version: **`3.3`**.

### Changes in `3.3` (judge panel via OpenRouter)

- `by_judge` is keyed by judge **key** (path-safe handle; equals the model id
  for all pre-3.3 runs). `judge_details` per judge grew to
  `{id, model_id, provider, transport, reasoning, family, family_overlap,
  family_stake_note}`.
- New top-level `panel` block when the run has ≥ 2 judges:
  `{judges, summary, prompts, agreement}`. Panel `prompts` entries carry
  consensus verdicts; each criterion's `results[model]` is
  `{pass, reason, votes: {pass, fail, abstain}}`. Artifact FAILs
  (judge_refusal / judge_parse_error / judge_truncated /
  missing_in_judge_output) abstain; `loop_failure` FAILs vote. Quorum
  min(2, n_judges); ties → FAIL `panel_tie`; below quorum → FAIL
  `panel_no_quorum`.
- `panel.agreement`: pairwise percent-agreement matrix, `fleiss_kappa`
  (variable-rater generalization over cells with ≥ 2 votes), per-judge
  `with_consensus` agreement, per-judge `abstentions`.
- Top-level `summary`/`prompts` mirror the **panel** when ≥ 2 judges
  (single-judge runs keep the first-judge view). New `meta.verdict_basis`
  records which (`"panel"` or a judge key). NOTE: re-aggregating an old
  2-judge freeze switches its default view to a 2-member panel where any
  disagreement is a tie → FAIL (strict both-must-pass); `panel_tie` reasons
  make the shift auditable.
- `meta.config` adds `classifier_chain` (judge keys), `judge_workers`,
  `judge_deadline_s`. `criteria_tags.jsonl` records add `model` (which
  chain member classified). Diagnosis records add `analyst`;
  `failure_analysis.diagnose_judge` is now the first entry of
  `failure_analysis.diagnose_chain`, and `judge_concurrence` generalizes
  from second-judge strings to a `{pass, fail, abstain}` vote split.

### Changes in `3.2` (additive)

Top-level `decode_health` block — mechanical repetition-loop census over
stored responses and captured rejected generation attempts, per candidate
model. Always present (pure computation, no judge cost). Records the
standing ruling (2026-07-09) that ANY loop — reasoning or content, escaped
or not — counts as a failure and is training signal. Documented in the
`decode_health` section below.

### Changes in `3.1` (additive)

Optional top-level `failure_analysis` block — root-cause diagnosis rows,
`by_root_cause` rollups, taxonomy echo, and post-hoc judge-concurrence
labels produced by the `diagnose` stage. Absent when the diagnose stage
has not run for the experiment. Documented in the `failure_analysis`
section below.

### Changes in `3.0` (multi-judge)

Top-level `by_judge` map added — one self-contained `{judge, judge_details,
validation, summary, prompts}` view per grader, all over the SAME candidate
responses. The top-level `summary` / `prompts` became the DEFAULT (first)
judge's view, kept for single-judge readers. `meta.judges` (ordered grader
array; first entry is the default) and `meta.counts.n_judges` added.

### Changes in `2.1` (additive)

Three top-level dashboard-facing fields promoted from `meta.experiment` /
`meta.config` so the ConstraintLens design's Logic can read them without
knowing our nesting. Nested versions remain authoritative; these are
display-only aliases.

- `meta.run_date` — mirrors `meta.experiment.run_date`.
- `meta.max_tokens` — mirrors `meta.config.candidate_max_tokens`.
- `meta.reasoning_enabled` — extracted from `meta.config.candidate_extra_body.reasoning.enabled`; `null` if the pipeline didn't set it explicitly.

### Breaking changes since `1.0`

- `meta.models` used to be `[{key, id, role}, …]`; is now an array of the
  key strings (`["qwen-9b", "qwen-35b", "qwen-397b"]`), matching how the
  dashboard indexes into `summary.criterion_pass_rate[key]`,
  `prompt.responses[key]`, `criterion.results[key]`. The rich variant
  moved to `meta.model_details`.
- `meta.judge` used to be `{id, provider, role, family_stake_note}`; is
  now the id string. The rich variant moved to `meta.judge_details`.

## Design principles

- **One file per experiment.** No cross-run joins in the deliverable — every field the dashboard needs to render a run is in that run's file.
- **Extensive but not exhaustive.** Include what the dashboard displays, what the limitations panel needs to explain the run, and what a future reader needs to reproduce it. Do not include per-call latency logs, provider request IDs, raw response bodies, or debug traces — those live in sidecar files.
- **Stable shape across runs.** Every experiment carries every key in this schema even when the value is null. Optional data is `null` / `[]` / `""`, never absent — with two documented exceptions where the *key itself* is absent: `failure_analysis` (diagnose stage hasn't run) and `panel` (fewer than 2 judges). See their rows in the table below.
- **Human-readable JSON.** UTF-8, 2-space indent, no trailing whitespace. File size is expected to be 100 KB – 5 MB depending on `n_prompts × n_models × response length`.

## Top-level shape

```json
{
  "schema_version": "3.3",
  "meta":    { … },
  "summary": { … },
  "prompts": [ … ],
  "by_judge": { … },
  "panel":   { … },
  "failure_analysis": { … },
  "decode_health": { … }
}
```

| key | type | notes |
| --- | --- | --- |
| `schema_version` | str | Matches this document. Dashboards refuse to render mismatched majors. |
| `meta` | object | Identity + configuration + validation status. |
| `summary` | object | Six pre-computed aggregate breakdowns — the **default view**: the `panel` consensus when ≥ 2 judges form one, else the first judge's view (`meta.verdict_basis` records which). |
| `prompts` | array | One entry per included prompt, in benchmark_id order — same default-view rule as `summary`. |
| `by_judge` | object | One self-contained `{judge, judge_details, validation, summary, prompts}` view per grader (schema 3.0). |
| `panel` | object | **Optional** (schema 3.3) — present when the run has ≥ 2 judges; consensus verdicts + agreement stats over the same eligible prompts. One of two exceptions to the never-absent principle: absent on single-judge runs (nothing to form a panel out of). See its section below. |
| `failure_analysis` | object | **Optional** (schema 3.1) — root-cause diagnosis; see its section below. The other exception to the never-absent principle: a missing key means the diagnose stage has not run. |
| `decode_health` | object | Mechanical loop census (schema 3.2) — always present; see its section below. |

---

## `meta`

### `meta.experiment`

Identity of this run.

| key | type | notes |
| --- | --- | --- |
| `slug` | str \| null | `E<NN>-<kebab-label>` (e.g. `E01-smoke-3p`). As of the parameterized CLI, every aggregate run carries a slug; `null` remains schema-legal for backward compatibility with pre-existing consumers, and those runs still write `outputs/results.json` but do not appear in `index.json`. |
| `number` | int \| null | Two-digit prefix parsed from `slug`. Used for dropdown ordering. |
| `label` | str \| null | Kebab-case portion after the prefix. |
| `description` | str | Free-text one-liner passed via `--description`. |
| `run_report` | str | Relative path to the `meta/<date>-<slug>.md` narrative report. |
| `run_date` | str | ISO-8601 UTC timestamp of when `aggregate` ran. |

### `meta.dataset`

Benchmark identity + license. Constants for v1 (only changes if the benchmark itself is swapped).

| key | type | notes |
| --- | --- | --- |
| `name` | str | Human-readable benchmark name. |
| `source` | str | Canonical URL. |
| `license` | str | SPDX-style license identifier. |
| `publisher` | str | Original author / organization. |

### `meta.models`

Ordered array of candidate **key strings** — the same short mnemonics used
everywhere else in the file (`prompt.responses[key]`,
`summary.criterion_pass_rate[key]`, `criterion.results[key]`). Array
order is the order the dashboard renders model tabs / columns.

```json
"models": ["qwen-9b", "qwen-35b", "qwen-397b"]
```

### `meta.model_details`

Optional companion to `meta.models` — richer per-model info. The
ConstraintLens dashboard doesn't read this; kept for future consumers
that need the provider-side IDs.

| key | type | notes |
| --- | --- | --- |
| `key` | str | Same short mnemonic as the corresponding entry in `meta.models`. |
| `id` | str | Provider-side model ID. |
| `role` | str | Always `candidate` in v1; kept for forward compat. |

### `meta.judge`

String — the **default grader's KEY** (e.g. `"claude-opus-4-8"`), i.e. the
first entry of `meta.judges`. Kept for back-compat with single-judge
dashboards; the dashboard renders this verbatim in the run-details panel and
footer. Grading itself may involve every member of `meta.judges` (a panel,
schema 3.0+); classification follows the separate, frozen
`meta.config.classifier_chain` (schema 3.3) — `meta.judge` is no longer a
"grader/classifier" model ID, just the default judge's handle.

### `meta.judge_details`

Per-judge provenance for the default judge (same shape as each entry under
`by_judge[*].judge_details`, schema 3.3).

| key | type | notes |
| --- | --- | --- |
| `id` | str | Same key string as `meta.judge`. |
| `model_id` | str | Provider-side model ID (e.g. `"claude-opus-4-8"`, `"openai/gpt-5.2"`). Equals `id` for Anthropic judges. |
| `provider` | str | `"anthropic"` or `"openrouter"`. |
| `role` | str | Always `"grader"`. |
| `transport` | str | `"batch"` (Anthropic Message Batches) / `"sequential"` (Anthropic, one streamed call per cell) / `"pooled_stream"` (OpenRouter judges — no batch API there). |
| `reasoning` | object | `{"type": "adaptive"}` for Anthropic judges; `{"enabled": true}` for OpenRouter judges. |
| `family` | str | Model family (e.g. `"anthropic"`, `"openai"`, `"google"`, `"deepseek"`). |
| `family_overlap` | bool | `true` if this judge shares a family with any candidate in `meta.models` — self-preference-bias risk flag. |
| `family_stake_note` | str | Human-readable statement of the self-preference-bias control. |

### `meta.counts`

Cross-referencing sums the dashboard uses as headers.

| key | type | notes |
| --- | --- | --- |
| `n_prompts` | int | Prompts fully graded across all candidates. |
| `n_criteria` | int | Sum of criteria across included prompts. |
| `n_models` | int | Candidate count. |
| `n_judges` | int | Judge count (schema 3.0) — length of `meta.judges`. |
| `n_grade_cells` | int | `n_criteria × n_models` — the grid the pass-rate metrics aggregate over (per judge). |

### `meta.categories`

Distinct values + counts per category, for the dashboard's filter UI.

```json
{
  "instruction_type": {"Negative": 2, "Multistep": 1},
  "prompt_style":     {"Direct prompting": 1, "Context prompting": 1, "Rambling/Stream-of-Consciousness": 1},
  "use_case":         {"Logistics, Scheduling & Event Planning": 2, "Data Processing, Formatting & Math": 1}
}
```

Empty categories (no prompts of that type in this run) are omitted from
their inner map. Keys inside each inner map are sorted alphabetically.

### `meta.config`

Every runtime knob that could differ between experiments. `candidates`
and `judge.id` live in `meta.models` / `meta.judge` respectively —
`config` is what a dashboard would surface in a "what was different
about this run" panel.

| key | type | notes |
| --- | --- | --- |
| `candidate_temperature` | float | Passed to OpenRouter. |
| `candidate_max_tokens` | int | Passed to OpenRouter. |
| `candidate_extra_body` | object | Provider extras (e.g. `{"reasoning": {"enabled": false}}`). |
| `candidate_timeout_s` | float | Per-call timeout. |
| `judge_max_tokens` | int | Judge + classifier call cap. |
| `judge_prompt_sha256_12` | str | First 12 hex chars of the judge system-prompt SHA256. Flags edits. |
| `classifier_prompt_sha256_12` | str | Same for the classifier system prompt. |
| `validate_seed` | int | Fixed seed for the human-validation sampler. |
| `validate_sample_target` | int | Number of rows sampled (capped by available). |
| `validate_response_excerpt_chars` | int | Char cap on `response_excerpt` in the validation file. |

### `meta.git`

| key | type | notes |
| --- | --- | --- |
| `commit` | str | Short SHA at the time `aggregate` ran. `""` if the repo isn't git-tracked. |
| `dirty` | bool \| null | `true` if the working tree had uncommitted changes; `null` if git wasn't available. |

### `meta.validation`

Judge-validation status. Populated by reading `outputs/judge_validation.json`
and `outputs/run_manifest.json`'s `judge_agreement` block at aggregation
time. **Re-run `aggregate` after `validate --mode score` to refresh.**

| key | type | notes |
| --- | --- | --- |
| `status` | str | `"not_run"`, `"sampled"`, or `"scored"`. Monotonic. |
| `n_sampled` | int | Row count in the sampled file. |
| `n_scored` | int | Row count with a filled `human` field. |
| `agreement_pct` | float \| null | Judge-vs-human agreement %; `null` until scored. |
| `scored_at` | str \| null | ISO-8601 UTC timestamp of `validate --mode score`; `null` until scored. |

---

## `summary`

All six values are model-keyed floats in the range `[0, 100]`. Every key
is present for every model in `meta.models`.

| key | shape | definition |
| --- | --- | --- |
| `criterion_pass_rate` | `{model_key → pct}` | passing criteria / all criteria, per model. |
| `full_prompt_pass_rate` | `{model_key → pct}` | prompts where every criterion passed / all prompts, per model. |
| `by_instruction_type` | `{type → {model_key → pct}}` | criterion-level pass rate scoped to that instruction type. |
| `by_prompt_style` | `{style → {model_key → pct}}` | criterion-level pass rate scoped to that prompt style. |
| `by_use_case` | `{use_case → {model_key → pct}}` | criterion-level pass rate scoped to that use case. |
| `by_verifiability` | `{"auto"\|"judge" → {model_key → pct}}` | criterion-level pass rate split by verifiability tag. |

The constraint-satisfaction gap surfaces as
`criterion_pass_rate − full_prompt_pass_rate`; compute it in the UI.

---

## `prompts`

Ordered array. One entry per included prompt (skipped prompts — those
where any candidate was missing a response or grade — are logged during
aggregation and excluded here so cross-model comparison stays
apples-to-apples).

Each prompt entry:

| key | type | notes |
| --- | --- | --- |
| `id` | str | Benchmark ID. |
| `use_case` | str | Verbatim from the benchmark. |
| `instruction_type` | str | Verbatim. |
| `prompt_style` | str | Verbatim. |
| `prompt_text` | str | Full original prompt. |
| `responses` | `{model_key → str}` | Full response text per candidate. |
| `criteria_passed` | `{model_key → "x/y"}` | Human-readable per-model tally. |
| `full_pass` | `{model_key → bool}` | `true` iff every criterion passed. |
| `criteria` | array | See below. |

Each `criteria[]` entry:

| key | type | notes |
| --- | --- | --- |
| `text` | str | Verbatim criterion text. |
| `verifiability` | str | `"auto"` (deterministically checkable) or `"judge"` (requires subjective judgment). |
| `gameable` | bool | `true` if a response could satisfy the literal wording while violating intent. |
| `reward_hack` | str | Short description of the shortcut if `gameable`; `""` otherwise. |
| `results` | `{model_key → {"pass": bool, "reason": str}}` | Judge verdict + reason per candidate. |

---

## `panel` (schema 3.3, optional)

Present when the run has ≥ 2 judges (`meta.judges` has 2+ entries) — a
provider-diverse panel of graders scoring the SAME candidate responses.
Absent for single-judge runs (nothing to form a panel out of); this is the
other exception to the never-absent shape principle, alongside
`failure_analysis`.

```json
"panel": {
  "judges": ["claude-opus-4-8", "claude-fable-5", "gpt-5", "gemini-3-pro", "deepseek-v4"],
  "summary": { /* same six keys as top-level summary, computed over consensus verdicts */ },
  "prompts": [ /* same shape as top-level prompts, but criteria[].results[model]
                  carries the consensus verdict + vote split (see below) */ ],
  "agreement": { /* pairwise + aggregate agreement stats, see below */ }
}
```

| key | type | notes |
| --- | --- | --- |
| `judges` | array | Judge keys that fed the panel, same order as `meta.judges`. |
| `summary` | object | Same six breakdowns as top-level `summary` (`criterion_pass_rate`, `full_prompt_pass_rate`, `by_instruction_type`, `by_prompt_style`, `by_use_case`, `by_verifiability`), computed over the panel's consensus verdicts. |
| `prompts` | array | Same shape as top-level `prompts` — same keys, same `criteria[]` structure — except each criterion's `results[model_key]` is `{pass, reason, votes}` (below) instead of `{pass, reason}`. |
| `agreement` | object | Pairwise + aggregate inter-judge agreement stats, see below. |

Top-level `summary`/`prompts` mirror this block whenever it's present (see
`meta.verdict_basis`); single-judge runs keep mirroring the first (only)
judge's `by_judge` view instead.

### `panel.prompts[*].criteria[*].results[model_key]`

| key | type | notes |
| --- | --- | --- |
| `pass` | bool | Consensus verdict — majority PASS among the judges that cast a vote. |
| `reason` | str | `""` on PASS. On FAIL: the most common reason text among the judges that voted FAIL, or `"panel_tie"` (equal PASS/FAIL split) / `"panel_no_quorum"` (fewer than quorum cast a vote). |
| `votes` | `{"pass": int, "fail": int, "abstain": int}` | Vote split over all panel judges. Artifact FAILs (`judge_refusal` / `judge_parse_error` / `judge_truncated` / `missing_in_judge_output`) abstain rather than cast a vote; `loop_failure` FAILs are genuine FAIL votes. Quorum is `min(2, n_judges)`; fewer cast votes than quorum → FAIL `panel_no_quorum`; an even PASS/FAIL split among cast votes → FAIL `panel_tie`. Missing evidence always resolves to FAIL, never a silent drop. |

### `panel.agreement`

| key | type | notes |
| --- | --- | --- |
| `pairwise` | `{judge_key → {judge_key → pct}}` | Percent agreement between each judge pair (0–100 scale), over cells where both judges cast a PASS/FAIL vote. Symmetric; a judge is never compared against itself. |
| `fleiss_kappa` | float | Variable-rater generalization of Fleiss' kappa, over cells with ≥ 2 cast votes — a chance-corrected agreement statistic, NOT on the 0–100 scale (typical range roughly -1 to 1; higher is more agreement than chance). |
| `with_consensus` | `{judge_key → pct}` | Each judge's percent agreement with the panel's own consensus verdict (0–100 scale). |
| `abstentions` | `{judge_key → int}` | Count of cells that judge abstained on (artifact FAILs or a missing grade record for that judge). |

---

## `failure_analysis` (schema 3.1, optional)

Absent key means the diagnose stage has not run for this experiment. `rows[*]` join `prompts[]` by `id` for criterion text and category metadata.

```jsonc
"failure_analysis": {
  "taxonomy_version": 1,          // provenance: stamped per artifact row at diagnose time;
                                  // "taxonomy_versions_seen": [..] appears (only) when rows
                                  // from mixed taxonomy versions are aggregated together
  "taxonomy": [ { "key": "constraint_dropped", "label": "…",
                  "description": "…", "training_implication": "…" } ],
  "diagnose_chain": ["claude-fable-5", "claude-opus-4-8"],  // analyst fallback chain
  "diagnose_judge": "claude-fable-5",       // chain[0] — scalar echo for old readers
  "verdict_basis": "panel",                 // whose FAILs were diagnosed: panel
                                            // consensus (>=2 judges) or a judge key
  "diagnosed_at": "…",                       // ISO timestamp
  "counts": { "failed_criteria": 219, "diagnosed": 219, "cells": 54 },
  "synthesis": { /* §4b — predecessor, comparison[], prior_recommendations_review[],
                    recommendations[1–3 of {category, action, rationale,
                    expected_signal}], iteration_note. Optional: absent when
                    synthesis was skipped or failed. */ },
  "rows": [ {
    "id": "CIF-012", "model": "qwen-9b", "criterion_index": 7,
    "root_cause": "constraint_dropped",
    "secondary": null,                       // optional second label (compound failures)
    "confidence": "high|medium|low",
    "evidence": "shortest quote from trace/answer that shows it",
    "rationale": "1–2 sentences",
    "trace_status": "present|absent|truncated",
    "analyst": "claude-opus-4-8",            // chain member that produced this row
                                             // (null for pre-3.3 artifacts)
    "judge_concurrence": { "pass": 0, "fail": 2, "abstain": 0 }  // panel vote split
  } ],
  "by_root_cause": { /* rollups: root_cause × model, × instruction_type, × use_case */ }
}
```

---

## `decode_health` (schema 3.2)

Mechanical repetition-loop census computed by `pipeline/_decode_health.py`
at aggregate time — pure computation over `runs/<slug>/responses/*.jsonl`
and the `runs/<slug>/rejected/*.jsonl` sidecars that `generate` writes for
doomed attempts (empty completion, deadline abort). No judge involvement,
so it is auto-verifiable and always present.

Standing ruling (2026-07-09): any loop counts as a failure, including loops
the model escaped before finishing (`n_loop_escaped`) — escaped loops waste
tokens and escaping is unreliable; looping is trainable behavior.

```json
{
  "detector": { "window": 4000, "ngram": 40, "min_repeats": 20 },
  "verdict_note": "any repetition loop counts as a failure …",
  "by_model": {
    "qwen-9b": {
      "n_responses": 75,
      "n_loop_reasoning": 6,          // loop detected in the thinking channel
      "n_loop_content": 1,            // loop detected in the visible answer
      "n_loop_any": 6,
      "n_loop_escaped": 5,            // looped but finish_reason == "stop" — still failures
      "n_loop_died": 1,               // looped and never finished cleanly
      "n_nonstop_finish": 4,          // finish_reason != "stop" (length/error/null)
      "rejected_attempts": { "n_captured": 12, "n_looping": 9 }
    }
  },
  "rows": [ {                          // FLAGGED responses only (loop or non-stop finish)
    "model": "qwen-9b", "id": "CIF-055", "finish_reason": "stop",
    "reasoning_chars": 149330, "content_chars": 4703,
    "reasoning_loop": { "looping": true, "period": 113, "onset": 31330 },
    "content_loop":   { "looping": false, "period": null, "onset": null }
  } ]
}
```

Caveat: `rejected_attempts` counts only attempts made after the capture
hook landed (2026-07-09, mid-E07) — earlier retries were discarded before
storage, so historical runs show `n_captured: 0`.

---

## Dashboard dropdown files

### `dashboard/runs.json` — the ConstraintLens design's dropdown source

Written by `scripts/dashboard_sync.py` (which is also invoked by
`pipeline/aggregate.py` at the end of every tagged run). The design's
Logic fetches `./runs.json` on load and, on selection, fetches each
run's own JSON at `./<path>`.

```json
{
  "runs": [
    {
      "id": "E01-smoke-3p",
      "label": "E01-smoke-3p — First end-to-end pipeline exercise",
      "date": "2026-07-01",
      "path": "./E01-smoke-3p.json",
      "n_prompts": 3,
      "n_models": 3
    }
  ],
  "synced_at": "2026-07-01T01:41:52…"
}
```

The design only reads `id`, `label`, `date`, `path`; the extra fields
are metadata for external tools.

### `outputs/experiments/index.json` — legacy dashboard registry

Carries one compact entry per tagged experiment. Kept because non-dashboard
consumers (analysis notebooks, batch scripts) may still index off it.

```json
{
  "schema_version": "2.0",
  "experiments": [
    {
      "slug": "E01-smoke-3p",
      "number": 1,
      "label": "smoke-3p",
      "description": "…",
      "run_date": "2026-07-01T00:31:47…",
      "run_report": "meta/2026-06-30-smoke-test.md",
      "n_prompts": 3,
      "n_criteria": 72,
      "n_models": 3,
      "models": ["qwen-9b", "qwen-35b", "qwen-397b"],
      "judge": "claude-opus-4-8",
      "validation_status": "sampled",
      "agreement_pct": null,
      "git_commit": "49a488f",
      "results_path": "experiments/E01-smoke-3p.json"
    }
  ]
}
```

Entries are sorted by `number` ascending. Re-tagging replaces the existing
entry with the same slug rather than duplicating.

---

## Versioning policy

- **Patch bumps (`2.0` → `2.0.1`)**: fixes to this document only, no shape change.
- **Minor bumps (`2.0` → `2.1`)**: additive-only. New optional key with a documented default. Existing dashboards keep rendering.
- **Major bumps (`2.0` → `3.0`)**: breaking change — key removed, renamed, or has an incompatible type change. Dashboards must gate rendering on `schema_version` prefix.

When bumping, update `SCHEMA_VERSION` in `pipeline/_experiment.py`, add
the change to this file (including a "Breaking changes since" section
under the header), and note the bump in the run report of the first
experiment produced under the new version.
