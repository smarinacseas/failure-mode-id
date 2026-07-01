# Failure Mode ID

Results: <https://smarinacseas.github.io/failure-mode-id/>

A criterion-level instruction-following loss analysis for open language
models. The repo contains a resumable eval pipeline that runs Surge
AI's **Complex Constraints** benchmark through the Qwen3.5 size ladder,
grades every criterion in every response with a blind Opus 4.8 judge,
classifies criteria for verifiability and reward-hackability, samples a
60-row human-grading file for judge validation, and emits a
schema-versioned `outputs/results.json` for the ConstraintLens
dashboard to consume.

**v1 is eval-only.** No training, no fine-tuning. The intended
deliverable is a clear picture of _where_ current open models fail on
constraint-dense prompts, with explicit bias controls and an honest
limitations audit attached. Training-time interventions are v2.

## Current status (as of 2026-06-30)

- **Pipeline: complete.** Every step (`load` → `generate` → `grade` →
  `classify` → `validate` → `aggregate`) is implemented, resumable,
  and end-to-end validated by the smoke.
- **Evaluated so far: 3 prompts of 75.** `E01-smoke-3p` exercised the
  whole pipeline on the first three benchmark prompts across the
  Qwen3.5 ladder (9B / 35B / 397B), producing 216 (prompt × model ×
  criterion) grade cells. The full `E02-v1-75p` run has not been
  kicked off yet.
- **Judge validation: sampled, not scored.** `validate --mode sample`
  drew a fixed-seed 60-row human-grading file at
  `outputs/judge_validation.json`; hand-grading and `validate --mode
  score` are pending, so `judge_agreement` is `null` in
  `run_manifest.json` and in every dashboard run detail. Read every
  headline number with that caveat in mind.
- **Dashboard: live** at
  <https://smarinacseas.github.io/failure-mode-id/>. Currently
  surfaces `E01-smoke-3p` as the sole entry in the run dropdown.

## What this project is trying to learn

Open models satisfy individual instructions at high rates but struggle
when prompts pile constraints on top of each other. The headline
metric this project is designed to surface is the
**constraint-satisfaction gap**: high per-criterion pass rate
alongside low full-prompt pass rate. Concretely: a model passes most
constraints on most prompts, but rarely all constraints simultaneously
on any one prompt.

v1 is designed to measure three things about that gap (see §Current
status for what has actually been evaluated):

1. **How it behaves across a size ladder.** Qwen3.5 9B → 35B → 397B, same family, no architectural confounds within the ladder.
2. **What failure modes it concentrates in.** Broken out by `instruction_type` (Negative / Multistep), `prompt_style` (Direct / Context / Rambling), and `use_case` (Logistics / Data-Math).
3. **How much of the measurement is judge artifact vs. real signal.** Quantified by hand-grading 60 sampled criteria and reporting judge-vs-human agreement, plus splitting pass rates by criterion verifiability (deterministically checkable vs. judge-dependent).

## The benchmark — Complex Constraints

[Surge AI's Complex Constraints Benchmark Set](https://huggingface.co/datasets/surgeai/ComplexConstraints)
is 75 prompts × up to 40 criteria each (~1,560 total criteria across
the set), released under CC-BY-4.0. Each prompt encodes a realistic
constraint-heavy request — a week-long staffing rota with overlapping
union, age, and shift-length constraints; a structured data-formatting
task with arithmetic that must reconcile; a multi-step logistics plan
with explicit prohibitions. Each criterion expresses one narrow,
literally-judgeable condition the response must satisfy.

The prompt mix varies along three axes that v1's analysis breaks out.
The vocabulary is deliberately narrow — see §Glossary for exact values
and descriptions:

| axis               | # values | source                                            |
| ------------------ | -------- | ------------------------------------------------- |
| `instruction_type` | 2        | dataset column                                    |
| `prompt_style`     | 3        | dataset column                                    |
| `use_case`         | 2        | dataset column                                    |
| `verifiability`    | 2        | Opus classifier — auto vs. judge                  |
| `gameable`         | bool     | Opus classifier — reward-hackability + shortcut   |

The benchmark skews toward constraint-heavy planning tasks, not all
instruction-following. Findings extrapolate to this class of task, not
to chat or open-ended QA.

## High-level architecture

A linear, resumable pipeline driven by a single CLI:

```
   xlsx                                          dashboard
    │                                                ▲
    │                                                │
 ┌──▼──┐  ┌──────────┐  ┌───────┐  ┌────────┐  ┌────┴────┐
 │load │→ │ generate │→ │ grade │→ │classify│→ │aggregate│
 └─────┘  └────┬─────┘  └───┬───┘  └────┬───┘  └────┬────┘
               │            │           │           │
        responses/    grades/    criteria_tags  results.json
        {model}.jsonl {model}.jsonl  .jsonl       + run_manifest.json
                                                     ▲
                                                     │
                                              ┌──────┴──────┐
                                              │  validate   │
                                              │ (sample &   │
                                              │   score)    │
                                              └─────────────┘
                                                     ▲
                                              judge_validation.json
                                              (60-row human-grading file)
```

- **Candidates run via [OpenRouter](https://openrouter.ai/)** — one provider, three Qwen variants, swappable model IDs.
- **Judge + classifier run via Anthropic directly** — Opus 4.8 as `JUDGE`. The judge is structurally non-candidate-family, so self-preference bias is by-design absent (see §Known limitations for the biases that *do* remain).
- **Every step is resumable per-id.** Re-running skips IDs already in the output file. A mid-run crash never wastes prior work.
- **Every step honors `--limit N`.** `all` requires an explicit `--limit`, so a full 75-prompt run is always an opt-in.
- **Pre-flight `connectivity` check** runs one tiny call to every candidate and the judge before any batch step. Bad model IDs fail loudly and immediately.
- **The judge is structurally blind.** The judge user message contains task prompt + response + numbered criteria only — never the candidate's name. The candidate's identity lives in the filename (`grades/qwen-9b.jsonl`); the judge never sees it.

### Per-step contracts

| step                     | input                                                  | output                                                                                                                                       | LLM calls                 |
| ------------------------ | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| `load`                   | `data/ComplexConstraints.xlsx`                         | `data/complexconstraints.jsonl` (one JSON line per prompt with `id`, `prompt`, `use_case`, `instruction_type`, `prompt_style`, `criteria[]`) | 0                         |
| `generate`               | `data/complexconstraints.jsonl`, `CANDIDATES` registry | `responses/{model}.jsonl` (one line per prompt: `{id, response}`)                                                                            | 1 per (prompt, candidate) |
| `grade`                  | prompts + responses + `prompts/judge.txt`              | `grades/{model}.jsonl` (one line per prompt: `{id, verdicts[{index, verdict, reason}]}`)                                                     | 1 per (prompt, candidate) |
| `classify`               | prompts + `prompts/classifier.txt`                     | `outputs/criteria_tags.jsonl` (one line per prompt: `{id, tags[{index, verifiability, gameable, reward_hack, ambiguous}]}`)                  | 1 per prompt              |
| `validate --mode sample` | grades + responses                                     | `outputs/judge_validation.json` (60 fixed-seed rows for human grading: model, id, criterion, judge_verdict, human="")                        | 0                         |
| `validate --mode score`  | filled `judge_validation.json`                         | merges `judge_agreement` block into `outputs/run_manifest.json`                                                                              | 0                         |
| `aggregate`              | everything above                                       | `outputs/results.json` (dashboard handoff) + `outputs/run_manifest.json` (run metadata) + tagged copy in `outputs/experiments/`              | 0                         |

## Experimental process

The intended workflow for any new run:

1. **Verify model IDs** — `connectivity` catches stale or mistyped IDs against the providers.
2. **Smoke first** — `all --limit 3` exercises every step end-to-end on the cost of a coffee. Catches schema bugs, provider quirks (Qwen reasoning-mode is _the_ gotcha; see `meta/2026-06-30-smoke-test.md`), JSON parse failures, and timing surprises before they cost real hours. **This is the only run that has been executed to date.**
3. **Full run** — `all --limit 75` runs sequentially; resumable. Roughly $50–60 and 4–6 hours wall-clock on a fresh run. Not yet executed.
4. **Human-validate the judge** — hand-grade the 60 rows in `outputs/judge_validation.json`, run `validate --mode score`. The agreement % is the credibility keystone; the bias-audit deliverable depends on it. Not yet executed.
5. **Re-aggregate** — `aggregate --limit 75` to fold the agreement number into `run_manifest.json` (results.json doesn't change).
6. **Write a run report** — copy `meta/TEMPLATE.md` to `meta/YYYY-MM-DD-<run-name>.md` and fill it. Sections include scope, attempt narrative, configuration deltas, results with explicit n caveats, an experimental-flaws audit, and a suggested-next-steps list.

Each step is one module under `pipeline/`. Add a candidate by
appending to `CANDIDATES` in `config.py`. Swap the judge by editing
the `JUDGE` constant and re-running `grade` (responses don't need
regeneration; the pipeline is decoupled at the JSONL boundary).

## Design choices worth knowing

- **Greedy decoding** (`temperature=0`) for both candidates and judge. Reproducibility over realism — a re-run gives the same outputs. The decoding-choice caveat is on every run report.
- **Reasoning disabled** on Qwen3.5 via OpenRouter's `reasoning.enabled: false`. Forced by the empirical discovery in the v1 smoke that thinking-mode burned the entire token budget on internal CoT and emitted zero visible content on every complex prompt. The pipeline now tests these models' no-CoT instruction-following, which is the realistic production-deployment condition for many users. The diagnostic is in `pipeline/generate.py` and the full account in `meta/2026-06-30-smoke-test.md`.
- **`max_tokens = 8000`** for candidate calls — empirically the smallest value at which complex responses finish naturally (`finish_reason: stop`) rather than truncating.
- **Defensive JSON extraction** for judge and classifier outputs — strip code fences, parse, then fall back to balanced-bracket scanning; retry once on parse failure; fall back to all-FAIL with a clear `judge_parse_error` reason on persistent failure. A flaky judge call never crashes the batch.
- **Broad retry predicate** — 429s, 5xx, network errors, _and_ SDK-level parse errors (OpenRouter has been observed returning mid-stream-truncated JSON bodies). Exponential backoff with jitter.
- **Sequential by design for v1.** No parallelism between candidates or prompts. The simple model is correct; parallelism is on the suggested-next-steps list.

## Experiment tagging + deliverables

Every `aggregate` step can be tagged with an **experiment slug** so the
dashboard's dropdown has a stable identifier for each run. The
convention is:

```
E<NN>-<kebab-case-label>
```

The two-digit number gives dropdown ordering. The label is 1–3
kebab-case tokens that hint at what makes the experiment different:

| slug               | meaning                                 | status                       |
| ------------------ | --------------------------------------- | ---------------------------- |
| `E01-smoke-3p`     | first smoke, 3 prompts                  | **done, live in dashboard** |
| `E02-v1-75p`       | v1 full run, 75 prompts                 | planned                      |
| `E03-reasoning-on` | reasoning-enabled ablation              | planned (v1.5)               |
| `E04-judge-swap`   | judge replaced by a non-Anthropic model | planned (v1.5)               |
| `E05-cross-family` | DeepSeek added as a fourth candidate    | planned (v1.5)               |

Passing `--experiment SLUG --description "..." --run-report meta/..."`
does four things:

1. Assembles the standardized deliverable — schema version `2.1`, `meta` block (experiment identity, dataset provenance, model keys, judge id, prompt/criterion/model/grade-cell counts, per-category populations, config snapshot, git state, judge-validation status, plus promoted `run_date` / `max_tokens` / `reasoning_enabled` fields the dashboard reads directly), the six aggregate summaries, and the per-prompt array with per-model verdicts + reasons. Full field-by-field contract in [`meta/RESULTS_SCHEMA.md`](meta/RESULTS_SCHEMA.md).
2. Writes a per-experiment copy under `outputs/experiments/<slug>.json`.
3. Updates `outputs/experiments/index.json` — a compact registry of every tagged experiment with slug, run_date, counts, models, judge, validation status, agreement %, git commit, and results path.
4. **Syncs the ConstraintLens dashboard**: copies each `<slug>.json` into `dashboard/` and rebuilds `dashboard/runs.json` (the design's dropdown source). Fires automatically at the end of every tagged `aggregate`; also runnable standalone via `uv run python scripts/dashboard_sync.py`. On failure the sync is skipped (aggregate never blocks).

Untagged runs still write `outputs/results.json` (same shape) but do
not appear in the dashboard dropdown.

### What each deliverable contains

- **`outputs/experiments/<slug>.json`** — the per-experiment
  deliverable. Top level: `schema_version`, `meta`, `summary`,
  `prompts[]`. Contract in [`meta/RESULTS_SCHEMA.md`](meta/RESULTS_SCHEMA.md).
- **`outputs/experiments/index.json`** — flat registry of every
  tagged experiment (one entry per slug). Powers the dashboard's
  run dropdown. Contains slug, run_date, counts, models, judge,
  validation status, agreement %, git commit, path.
- **`outputs/run_manifest.json`** — mutable run metadata for the
  most recent aggregate. Kept in sync with `results.json`'s `meta`
  block. `validate --mode score` merges `judge_agreement` here
  without touching `results.json`.
- **`outputs/results.json`** — canonical output of the current
  aggregate. Same shape as `<slug>.json`.
- **`outputs/criteria_tags.jsonl`** — per-criterion classifier
  tags (verifiability, gameable, reward_hack, ambiguous).
- **`outputs/judge_validation.json`** — 60 fixed-seed rows for
  human grading. Rows carry model / prompt id / criterion index /
  criterion text / response excerpt / judge verdict / an empty
  `human` field to fill in.
- **`meta/<date>-<slug>.md`** — one narrative run report per
  experiment. Uses `meta/TEMPLATE.md`.

### What the dashboard surfaces

The dashboard is a static Claude Design bundle unpacked to
`dashboard/` by `scripts/unpack_design.py`. GitHub Pages serves
`dashboard/` as-is (no build step); the
`.github/workflows/dashboard-deploy.yml` workflow uploads on every
push to `main` that touches `dashboard/**`.

Views:

- **Overview** — the Run Details panel (11 fields: models tested,
  judge, prompts evaluated, criteria graded, run date, token
  limits, benchmark identity, coverage, reasoning mode,
  judge-validation status, git commit), per-criterion charts by
  instruction type / prompt style / use case, and a per-model
  comparison card.
- **Prompts** — a drilldown table with one row per prompt. Columns:
  id, use case, instruction-type pill, prompt style, pass ratio,
  full-pass status. Filter dropdowns above the table narrow by any
  of the three axes.
- **Prompt detail** — modal opened by clicking a drilldown row.
  Shows the full prompt text, the response from the currently
  selected model, and a per-criterion table with PASS/FAIL badges
  + the judge's stated reason for each verdict. Filterable by
  `verifiability` (all / auto / judge).

Chrome:

- **Model tabs** in the header switch which candidate's verdicts
  drive the shown numbers. Every ratio, chart, and status pill
  reflows.
- **Run dropdown** in the header lists every experiment slug the
  pipeline has tagged (currently just `E01-smoke-3p`).
- **Tag pills** on the drilldown and detail modal have `title="…"`
  tooltips populated from a project-specific glossary — see
  §Glossary.
- **Footer** shows a compact `run / judge / counts / models`
  summary and a `Glossary ↗` link back to this README.

Layout:

- **Responsive** at 640 / 1024 / 1600 breakpoints. Mobile stacks
  every grid to single column and lets tables scroll horizontally
  inside their containers. Tablet compresses 4/5-column grids to
  2-column. Ultra-wide displays get a raised content cap
  (`min(1600px, 96vw)`) so long lines don't drift into whitespace.
- **Text-selection highlight** overridden to a saturated warm
  brown with cream text, so highlighted text stays legible on
  every text tone in the design's palette.

### Concrete example

The current smoke, generated by:

```bash
uv run python main.py aggregate --limit 3 \
    --experiment E01-smoke-3p \
    --description "First end-to-end pipeline exercise (3 prompts, 3 Qwen candidates)." \
    --run-report meta/2026-06-30-smoke-test.md
```

For a full run:

```bash
uv run python main.py all --limit 75 \
    --experiment E02-v1-75p \
    --description "v1 full run, reasoning disabled, max_tokens=8000." \
    --run-report meta/2026-07-02-v1-full-run.md
```

## Quickstart

```bash
# 0. Dependencies (already in pyproject.toml + uv.lock)
uv sync

# 1. Provide API keys
cp .env.example .env       # then edit, add OPENROUTER_API_KEY and ANTHROPIC_API_KEY

# 2. Pre-flight — fails fast on bad model IDs
uv run python main.py connectivity

# 3. Smoke test — 3 prompts × 3 candidates, ~10–20 min
uv run python main.py all --limit 3

# 4. Full run — 75 prompts × 3 candidates, ~4–6 hr, ~$50–60
uv run python main.py all --limit 75

# 5. Human-grade outputs/judge_validation.json (60 rows), then:
uv run python main.py validate --mode score
uv run python main.py aggregate --limit 75
```

CLI surface:

```
uv run python main.py <step> [--limit N] [--mode sample|score]

steps:  load | generate | grade | classify | validate | aggregate | all | connectivity
```

Every step honors `--limit`. `all` _requires_ `--limit` — there is no default-to-full-set behavior, so production runs are always explicit.

## Project layout

```
config.py                 # API clients + CANDIDATES registry + JUDGE id + paths
main.py                   # CLI orchestrator

pipeline/
  load.py                 # xlsx → JSONL
  generate.py             # OpenRouter candidate calls (greedy, no-CoT, resumable)
  grade.py                # blind Opus judge, one call per (prompt, candidate)
  classify.py             # verifiability / gameable tags per criterion
  validate.py             # human-grading sampler + agreement scorer
  aggregate.py            # join all artifacts → results.json (+ tagged copy)
  connectivity.py         # pre-flight model-ID check
  _experiment.py          # meta-block assembly + per-experiment writer
  _io.py                  # JSONL helpers + retry-with-backoff
  _json_extract.py        # defensive JSON-array extraction from LLM output

prompts/
  judge.txt               # judge system prompt (verbatim contract)
  classifier.txt          # classifier system prompt (verbatim contract)

data/
  ComplexConstraints.xlsx        # source benchmark (Surge AI, CC-BY-4.0)
  complexconstraints.jsonl       # regenerated by `load`; gitignored

responses/                # candidate outputs, per model, JSONL; gitignored
grades/                   # judge verdicts, per model, JSONL; gitignored
outputs/                  # dashboard handoff + manifest + validation file; gitignored except experiments/
  results.json
  run_manifest.json
  criteria_tags.jsonl
  judge_validation.json
  experiments/            # tagged per-experiment JSONs + index.json (committed)
    E01-smoke-3p.json
    index.json

meta/                     # run reports (committed)
  RESULTS_SCHEMA.md
  TEMPLATE.md
  YYYY-MM-DD-<run>.md

design/
  ConstraintLens Dashboard.html  # design-tool bundle (source of truth for dashboard)

dashboard/                # unpacked static bundle deployed to GH Pages (committed)
  index.html              # regenerated by scripts/unpack_design.py
  support.js
  fonts/
  runs.json               # dashboard dropdown source (regenerated by dashboard_sync.py)
  <slug>.json             # per-experiment copies (regenerated by dashboard_sync.py)

scripts/
  unpack_design.py        # design bundle → deployable static files
  dashboard_sync.py       # outputs/experiments/*.json → dashboard/
```

## Documentation

- **[`meta/RESULTS_SCHEMA.md`](meta/RESULTS_SCHEMA.md)** — the dashboard input contract. Field-by-field spec for `outputs/experiments/<slug>.json` and `outputs/experiments/index.json`, plus the versioning policy for shape changes. The dashboard reads this file and nothing else to know what it's binding against. Currently at schema version `2.1`.
- **[`meta/TEMPLATE.md`](meta/TEMPLATE.md)** — canonical run-report shape. Every section is required, including the experimental-flaws audit and the suggested-next-steps list. Sections: TL;DR, scope, initial configuration, attempt-by-attempt run timeline, configuration adjustments + justifications, models evaluated, headline results (with explicit n caveats), experimental flaws and biases, cost & timing, output schema, lessons, next experiments (structured hypothesis tests), suggested next steps (procedural improvements), judge validation, reproducibility, open questions.
- **[`meta/2026-06-30-smoke-test.md`](meta/2026-06-30-smoke-test.md)** — first end-to-end exercise of the pipeline (3 prompts, slug `E01-smoke-3p`). Documents the two reliability surprises that surfaced (Qwen reasoning-mode trap, OpenRouter mid-stream JSON truncation), the configuration adjustments made in response, and a 15-item next-steps list. Read this before extending the pipeline or kicking off a new full run.
- The pre-implementation runbook (`ComplexConstraints-v1-master-runbook.md`) is the design spec and is gitignored — present locally as the source-of-truth for what v1 should be, intentionally not shipped.

## Glossary

Project-specific vocabulary. Every tag pill in the dashboard traces
back to one of these definitions.

### Prompt taxonomy — the three axes

The Complex Constraints benchmark ships each prompt tagged along
three axes. v1's analysis breaks out results along all three so a
"which failure mode dominates?" question has a concrete answer.

- **`use_case`** — the domain the prompt lives in. The dataset ships
  two:
  - **Data Processing, Formatting & Math** — structured output under
    numerical and formatting constraints (spreadsheets, reports, JSON
    schemas with arithmetic that must reconcile).
  - **Logistics, Scheduling & Event Planning** — multi-entity plans
    under real-world constraints (rotas with union/age/shift rules,
    routing with prohibitions, event flows).
- **`instruction_type`** — the constraint shape the prompt stresses:
  - **Negative** — must-_not_ constraints ("avoid X", "no Y"). The
    canonical failure is introducing the forbidden element anyway.
  - **Multistep** — sequenced operations where step N depends on
    step N−1. The canonical failure is intermediate-step errors
    compounding into an incoherent final answer.
- **`prompt_style`** — how the instructions are presented:
  - **Direct prompting** — clear, numbered/bulleted instructions.
  - **Context prompting** — instructions embedded in a narrative
    scenario (a stakeholder message, a briefing).
  - **Rambling/Stream-of-Consciousness** — instructions buried in
    conversational, redundant, or noisy prose.

### Criterion classification — the two rubric axes

The classifier step tags every criterion (not every prompt) along
two axes. This is what lets the analysis separate judge artifact
from real signal.

- **`verifiability`** — can a deterministic program check it?
  - **`auto`** — yes (exact numbers, exact strings, presence/absence,
    formatting). Judge-error immune. Where these still fail, the
    model really failed.
  - **`judge`** — no, needs subjective reading. Where judge-vs-human
    agreement matters most; the bias audit weights these by
    `judge_agreement`.
- **`gameable`** — does the wording admit a shortcut that satisfies
  the literal criterion while violating intent? A `true` here is a
  reward-hack surface. The specific shortcut is captured verbatim in
  **`reward_hack`**.

### Metrics — how to read the numbers

- **`criterion_pass_rate`** — share of all (criterion × prompt)
  cells the model passed. The number most public benchmarks report.
  Optimistic on its own.
- **`full_prompt_pass_rate`** — share of prompts where the model
  passed _every_ criterion. What actually matters for a production
  workload that expects the whole instruction honored, not most of
  it.
- **Constraint-satisfaction gap** — the delta between the two above.
  The headline finding v1 is designed to surface: models pass most
  constraints on most prompts but rarely all constraints on any one
  prompt.
- **`judge_agreement`** — the % of the 60-row human-graded
  validation sample where the judge's verdict matched the human's.
  The credibility keystone; the bias-audit deliverable depends on
  it. **Currently unmeasured** — see §Current status.

### Run metadata — what to read on the dashboard

- **Reasoning mode** — whether Qwen's internal chain-of-thought was
  enabled. v1 disables it after the smoke found thinking-mode
  burning the full token budget on internal CoT and emitting zero
  visible response. See `meta/2026-06-30-smoke-test.md`.
- **Token limits** — the `max_tokens` cap for candidate calls.
  Empirically the smallest value at which complex responses finish
  naturally (`finish_reason: stop`) rather than truncating.
- **Judge validation** — status of the 60-row human-grading sample:
  `not run`, `sample drawn (n=60, ungraded)`, or `X% agreement`.
- **Git commit** — the working-tree state when `aggregate` ran.
  `(dirty)` means uncommitted changes; reproducibility caveats
  apply.

## Known limitations

Read every headline number with these caveats in mind. Some are
inherent to the design of v1; some are the current cursor position
on the roadmap.

### Currently in the eval

- **Only 3 prompts have been evaluated.** `E01-smoke-3p` is the sole
  run. All summaries (pass rates, per-category breakdowns, per-model
  comparisons) are derived from 3 prompts × 72 criteria × 3 models
  = 216 grade cells — an n far too small to draw distributional
  conclusions. Interpret E01 as a "does the pipeline work
  end-to-end" signal, not as a model ranking.
- **Judge agreement is currently unmeasured.** The 60-row human
  validation sample exists but has not been hand-graded.
  `run_manifest` shows `validation.status: "sampled"` and
  `agreement_pct: null`. Until it's scored, judge-specific bias is
  unbounded — criterion-level `verifiability="judge"` rates should
  be discounted accordingly.
- **`E01` was captured with a dirty working tree** (`git.dirty:
  true`). The `run_manifest` records the SHA and the dirty flag,
  but exact replay requires re-running from a committed state.

### Design-inherent caveats

- **Single candidate family.** Only Qwen3.5 (9B / 35B / 397B). No
  cross-family control run, so any pattern observed may be
  Qwen-specific rather than a general open-model behavior. A
  DeepSeek addition is on the v1.5 roadmap.
- **Greedy decoding only** (`temperature=0`). Reproducible but
  unrepresentative of realistic sampling; results describe the
  model's best single-shot output, not its typical output under
  temperature.
- **Reasoning disabled on Qwen3.5.** Discovered in the smoke that
  thinking-mode consumed the full 8000-token budget on internal
  chain-of-thought and emitted no visible response. This eval
  therefore describes no-CoT instruction-following — a realistic
  production condition for the many users who deploy without
  reasoning, but not a description of Qwen3.5's ceiling with CoT
  enabled. An `E03-reasoning-on` ablation is on the v1.5 roadmap.
- **Judge: Anthropic Opus 4.8.** Non-candidate-family (self-preference
  bias is structurally absent by design), but the judge still has
  its own biases, blind spots, and preferred writing styles. The
  `judge_agreement` validation is the only mitigation and is
  currently unmeasured. An `E04-judge-swap` ablation is planned.
- **Benchmark skew.** Complex Constraints is 75 prompts weighted
  toward planning/logistics and structured data. Findings do not
  extrapolate to open-ended QA, chat, or code generation.
- **Very narrow tag vocabularies.** 2 values for `instruction_type`,
  3 for `prompt_style`, 2 for `use_case`. Per-category breakdowns
  have small denominators even at the full 75-prompt scale — cell
  counts of a few dozen, not hundreds. See §Glossary.

### Infrastructure caveats

- **API-driven pipeline.** OpenRouter (candidates) and Anthropic
  (judge) are both external services; provider-side model
  changes, rate-limit variance, or mid-stream JSON truncation (an
  observed OpenRouter failure mode) can affect reruns.
- **Sequential execution.** No parallelism between candidates or
  prompts. Correct but slow at full scale — a 75-prompt run is
  4–6 hours wall-clock. Parallelization is on the v1.5
  suggested-next-steps list.
- **Judge validation is a one-shot 60-row sample.** Not stratified
  by category or by model. A stratified re-sample is on the
  next-steps list in `meta/2026-06-30-smoke-test.md`.

## Roadmap

|          | scope                                                                                                                                                                                                           | status                                                                            |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **v1**   | Eval-only. Qwen size ladder. Blind Opus judge. Verifiability + gameability classifier. Judge validation against a human-graded sample. Single `results.json` for an external dashboard. Bias-audit deliverable. | pipeline complete; smoke green (`E01-smoke-3p`); awaiting full 75-prompt run + human validation |
| **v1.5** | Add a cross-family candidate (DeepSeek) to distinguish Qwen-specific failure modes from general open-model patterns. Add Anthropic prompt caching to cut judge cost. Possibly parallelize candidate generation. | proposed; tracked in `meta/2026-06-30-smoke-test.md` next-steps                   |
| **v2**   | Training-time interventions. The eval surface stays; the candidates become outputs of a rubric-and-verifier RL pipeline. Out of scope for this repo.                                                            | future                                                                            |

## Data & Attribution

This project evaluates models against the **Complex Constraints Benchmark Set**,
released by Surge AI under CC-BY-4.0.

- Source: <https://huggingface.co/datasets/surgeai/ComplexConstraints>
- License: CC-BY-4.0 (<https://creativecommons.org/licenses/by/4.0/>)

### Citation

```
@misc{complex_constraints_benchmark,
  title  = {Complex Constraints Benchmark Set},
  author = {TODO: authors},
  year   = {2025},
  url    = {TODO: dataset URL}
}
```

### Modifications

Generated model responses, derived per-criterion pass/fail grades via an
LLM judge, and classified criteria for verifiability and reward-
hackability. These derived artifacts and all code in this repo are
MIT-licensed (see `LICENSE`). Per CC-BY-4.0, prompts and criteria from
the benchmark remain attributable to Surge AI.
