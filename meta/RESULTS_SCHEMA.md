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

Current schema version: **`2.0`**.

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
- **Stable shape across runs.** Every experiment carries every key in this schema even when the value is null. Optional data is `null` / `[]` / `""`, never absent.
- **Human-readable JSON.** UTF-8, 2-space indent, no trailing whitespace. File size is expected to be 100 KB – 5 MB depending on `n_prompts × n_models × response length`.

## Top-level shape

```json
{
  "schema_version": "2.0",
  "meta":    { … },
  "summary": { … },
  "prompts": [ … ]
}
```

| key | type | notes |
| --- | --- | --- |
| `schema_version` | str | Matches this document. Dashboards refuse to render mismatched majors. |
| `meta` | object | Identity + configuration + validation status. |
| `summary` | object | Six pre-computed aggregate breakdowns. |
| `prompts` | array | One entry per included prompt, in benchmark_id order. |

---

## `meta`

### `meta.experiment`

Identity of this run.

| key | type | notes |
| --- | --- | --- |
| `slug` | str \| null | `E<NN>-<kebab-label>` (e.g. `E01-smoke-3p`). `null` for untagged runs — those still write `outputs/results.json` but do not appear in `index.json`. |
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

String — the grader/classifier model ID (e.g. `"claude-opus-4-8"`). The
dashboard renders this verbatim in the run-details panel and footer.

### `meta.judge_details`

Optional richer variant of `meta.judge`.

| key | type | notes |
| --- | --- | --- |
| `id` | str | Same string as `meta.judge`. |
| `provider` | str | e.g. `Anthropic`. |
| `role` | str | e.g. `grader + classifier`. |
| `family_stake_note` | str | Human-readable statement of the self-preference-bias control. |

### `meta.counts`

Cross-referencing sums the dashboard uses as headers.

| key | type | notes |
| --- | --- | --- |
| `n_prompts` | int | Prompts fully graded across all candidates. |
| `n_criteria` | int | Sum of criteria across included prompts. |
| `n_models` | int | Candidate count. |
| `n_grade_cells` | int | `n_criteria × n_models` — the grid the pass-rate metrics aggregate over. |

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
