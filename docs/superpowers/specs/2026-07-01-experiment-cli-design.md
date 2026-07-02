# Parameterized Experiment CLI — Design

**Date:** 2026-07-01
**Status:** Approved (pending spec review)

## Goal

Make running a new experiment a single command with arguments — no editing of
hard-coded values in `config.py` — and document the workflow in the README so
anyone can clone the repo, add their own API keys, and run experiments.

## Motivation

Today every run-time knob (`CANDIDATE_MAX_TOKENS`, `CANDIDATE_TEMPERATURE`,
`CANDIDATE_EXTRA_BODY` reasoning toggle, `CANDIDATE_TIMEOUT_S`, `JUDGE`,
`CANDIDATES`) is a module-level constant in `config.py`, imported by name
across the pipeline. Running `E03-reasoning-on` means editing source. Worse,
`responses/` and `grades/` are keyed only by candidate, and every stage resumes
by skipping already-done IDs — so changing a parameter and re-running silently
mixes data generated under different settings.

## Decisions (made during brainstorming)

1. **Run isolation: per-experiment directories.** Each experiment's
   intermediate data lives under `runs/<slug>/`. Different parameter sets can
   never contaminate each other; per-id resume works within a run.
2. **CLI surface:** `--limit`, `--max-tokens`, `--reasoning`, `--temperature`,
   `--timeout`, `--candidates`, `--judge`, `--description`. Validation knobs
   (`VALIDATE_SEED`, `VALIDATE_SAMPLE_TARGET`, excerpt chars) stay fixed in
   `config.py` — the fixed seed is a deliberate reproducibility commitment.
3. **Param memory: freeze on first run.** The first invocation of a slug
   writes `runs/<slug>/experiment.json`; later invocations need only
   `--experiment <slug>` and reuse the frozen params. A conflicting explicit
   flag is an error.
4. **Architecture: explicit `RunConfig` dataclass** (approach A). Built once
   in `main.py`, passed to every stage. No module-level knob mutation, no
   env-var indirection.
5. **Judge flag (user amendment):** `--judge` selects the Anthropic model used
   for both grading and classification. Default changes to `claude-fable-5`.
   Anthropic-only for now (validated); non-Anthropic judges remain E04/v1.5.

## CLI interface

Every data-producing step (`generate`, `grade`, `classify`, `validate`,
`aggregate`, `all`) **requires `--experiment`**. `load`, `connectivity`, and
`status` do not take one (`load` output is parameter-independent and shared).

First invocation defines the experiment:

```bash
uv run python main.py all --experiment E03-reasoning-on --limit 75 \
    --reasoning on --description "reasoning-enabled ablation"
```

Later invocations resume with frozen params:

```bash
uv run python main.py all --experiment E03-reasoning-on      # crash resume
uv run python main.py grade --experiment E03-reasoning-on    # re-run one step
```

### Flags

| flag | type / values | default | frozen? |
|---|---|---|---|
| `--experiment` | `E<NN>-<kebab-label>` slug | *(required for data steps)* | — (identity) |
| `--limit` | int | required for new experiments via `all` | yes |
| `--max-tokens` | int | 8000 | yes |
| `--reasoning` | `on` \| `off` | `off` | yes |
| `--temperature` | float | 0.0 | yes |
| `--timeout` | float seconds | 300 | yes |
| `--candidates` | comma list, see below | full registry | yes |
| `--judge` | Anthropic model id | `claude-fable-5` | yes |
| `--description` | string | `""` | yes |
| `--run-report` | path | `""` | no — supplied at `aggregate` time |
| `--mode` | `sample` \| `score` | `sample` | no — `validate` subcommand selector |

`--candidates` entries are either a key from the default registry in
`config.py` (`qwen-9b`, `qwen-35b`, `qwen-397b`) or a `key=provider/model-id`
pair to add a new model, e.g.:

```bash
--candidates qwen-9b,qwen-397b,deepseek=deepseek/deepseek-v4
```

An unknown bare key errors, listing the registry keys.

`--judge` must look like an Anthropic model id (`claude-` prefix). The judge
client is Anthropic-only; a non-Anthropic value errors with a pointer to the
E04-judge-swap roadmap item. Note the Fable 5 default: ~2× Opus 4.8 per-token
pricing, and it can return `stop_reason: "refusal"` — the existing
parse-failure path (retry → all-FAIL with recorded reason) contains that case.

### Freeze semantics

- First invocation of a slug: resolve params = defaults ⊕ explicit flags;
  write `runs/<slug>/experiment.json` (params + `created_at` + schema tag).
- Later invocations: load the frozen params. Any *explicitly passed* flag that
  conflicts → exit 2 with a field-by-field diff (frozen vs passed) and a hint
  to pick a new slug. Re-passing an identical value is fine.
- `--run-report` and `--mode` are per-invocation, never frozen.

## RunConfig

New module `pipeline/run_config.py`:

```python
@dataclass(frozen=True)
class RunConfig:
    slug: str
    candidates: dict[str, str]   # key -> provider model id
    judge: str
    max_tokens: int
    temperature: float
    reasoning: bool              # serialized into extra_body at call site
    timeout_s: float
    limit: int | None
    description: str
```

- `to_json_dict()` / `from_json_dict()` for the experiment.json round-trip.
- `resolve(slug, cli_args) -> RunConfig` implements the freeze semantics.
- Path helpers derive per-run locations from the slug (see layout below).

Stages change signature to `run(cfg: RunConfig, *, monitor=None)` (validate
keeps `mode`, aggregate keeps `run_report`) and stop importing knob constants.
`config.py` keeps: API clients, the default candidate registry, default knob
values, validation constants, and shared paths.

`pipeline/_experiment.py`'s `config_block()` / `models_block()` /
`judge_block()` etc. take `cfg` and serialize from it — the `meta.config`
snapshot is built from the same object that ran, so it cannot drift.

## Directory layout

```
runs/
  E03-reasoning-on/
    experiment.json          # frozen RunConfig + created_at
    responses/<key>.jsonl
    grades/<key>.jsonl
    criteria_tags.jsonl
    judge_validation.json
    run_manifest.json
```

Unchanged:

- `data/complexconstraints.jsonl` — shared, written by `load`.
- `outputs/progress.json` + `outputs/logs/` — one heartbeat per machine;
  `status` command untouched.
- `outputs/experiments/<slug>.json` + `index.json` — the dashboard contract.
  No dashboard changes.

Migration: existing `responses/`, `grades/`, `outputs/criteria_tags.jsonl`,
`outputs/judge_validation.json`, `outputs/run_manifest.json` move into
`runs/E01-smoke-3p/` (one-time, part of implementation). A matching
`experiment.json` is written for E01 recording the parameters it actually used
(judge `claude-opus-4-8`, max_tokens 8000, reasoning off, temperature 0,
timeout 300, limit 3, full Qwen registry). `outputs/results.json` keeps being
written by `aggregate` as the "latest run" convenience copy — the dashboard's
legacy fallback reads it — but the per-experiment file is the deliverable.

## Error handling

- Resume with conflicting flags → exit 2, field-by-field diff, hint to use a
  new slug.
- Data step without `--experiment` → argparse error.
- Invalid slug → existing `InvalidSlugError` message.
- Unknown bare candidate key → error listing registry keys.
- Non-`claude-` judge id → error citing Anthropic-only judge client.
- New experiment via `all` without `--limit` → exit 2 (existing behavior,
  message updated).

## README changes

Replace the Quickstart with a **Running experiments** section:

1. Setup: clone, `uv sync`, `cp .env.example .env`, add `OPENROUTER_API_KEY`
   + `ANTHROPIC_API_KEY`.
2. The flag table above, with defaults.
3. Freeze/resume semantics in two sentences.
4. Where data lands (`runs/<slug>/`, `outputs/experiments/`).
5. Worked examples, each one copy-pasteable command:
   - smoke test (`E06-smoke-3p`-style, `--limit 3`)
   - full run (`--limit 75`)
   - `E03-reasoning-on` (`--reasoning on`)
   - `E05-cross-family` (`--candidates ...,deepseek=deepseek/deepseek-v4`)
   - judge swap within Anthropic (`--judge claude-opus-4-8`)

Also update the "Design choices" / "Project layout" sections to reflect
`runs/` and the RunConfig flow, and the experiments table statuses.

## Testing

- `RunConfig` round-trip (to/from JSON), freeze-on-first-run, conflict
  detection (explicit-flag diff), identical-repass acceptance.
- `--candidates` parsing: bare keys, `key=id` pairs, unknown-key error.
- Judge prefix validation.
- Path derivation from slug.
- Update existing wiring/stage/monitor tests for the new signatures; per-id
  resume tests now run against a `runs/<slug>/` tree.

## Out of scope

- Non-Anthropic judge support (E04, v1.5).
- Parallel generation (v1.5).
- Exposing validation sampler knobs on the CLI.
- Dashboard changes.
