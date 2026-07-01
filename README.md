# Failure Mode ID

A criterion-level instruction-following loss analysis for open language
models. The pipeline runs Surge AI's **Complex Constraints** benchmark
through a Qwen3.5 size ladder, grades every criterion in every response
with a blind Opus 4.8 judge, classifies criteria for verifiability and
reward-hackability, validates the judge against a human-graded sample,
and emits a single `outputs/results.json` for a separate dashboard to
consume.

**v1 is eval-only.** No training, no fine-tuning. The deliverable is a
clear picture of *where* current open models fail on constraint-dense
prompts, with explicit bias controls and an honest limitations audit
attached. Training-time interventions are v2.

## What this project is trying to learn

Open models satisfy individual instructions at high rates but struggle
when prompts pile constraints on top of each other. The headline metric
this project surfaces is the **constraint-satisfaction gap**: high
per-criterion pass rate alongside low full-prompt pass rate. Concretely:
a model passes most constraints on most prompts, but rarely all
constraints simultaneously on any one prompt.

v1 measures three things about that gap:

1. **How it behaves across a size ladder.** Qwen3.5 9B → 35B → 397B, same family, no architectural confounds within the ladder.
2. **What failure modes it concentrates in.** Broken out by `instruction_type` (Negative / Multistep / …), `prompt_style` (Direct / Context / Rambling / …), and `use_case` (Logistics / Data-Math / …).
3. **How much of the measurement is judge artifact vs. real signal.** Quantified by hand-grading 60 sampled criteria and reporting judge-vs-human agreement, plus splitting pass rates by criterion verifiability (deterministically checkable vs. judge-dependent).

## The benchmark — Complex Constraints

[Surge AI's Complex Constraints Benchmark Set](https://huggingface.co/datasets/surgeai/ComplexConstraints)
is 75 prompts × up to 40 criteria each (~1,560 total criteria across the
set), released under CC-BY-4.0. Each prompt encodes a realistic
constraint-heavy request — a week-long staffing rota with overlapping
union, age, and shift-length constraints; a structured data-formatting
task with arithmetic that must reconcile; a multi-step logistics plan
with explicit prohibitions. Each criterion expresses one narrow,
literally-judgeable condition the response must satisfy.

The prompt mix varies along three axes that v1's analysis breaks out:

| axis | values |
| --- | --- |
| `instruction_type` | e.g. Negative (must-not constraints), Multistep (sequenced operations) |
| `prompt_style` | Direct, Context prompting, Rambling/Stream-of-Consciousness |
| `use_case` | Logistics/Scheduling/Event Planning, Data Processing/Formatting/Math |

The benchmark skews toward constraint-heavy planning tasks, not all
instruction-following — a domain caveat that the bias-audit deliverable
flags explicitly. Findings extrapolate to this class of task, not to
chat or open-ended QA.

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
- **Judge + classifier run via Anthropic directly** — Opus 4.8 as `JUDGE`. The judge is structurally non-candidate-family, so self-preference bias is by-design absent.
- **Every step is resumable per-id.** Re-running skips IDs already in the output file. A mid-run crash never wastes prior work.
- **Every step honors `--limit N`.** `all` requires an explicit `--limit`, so a full 75-prompt run is always an opt-in.
- **Pre-flight `connectivity` check** runs one tiny call to every candidate and the judge before any batch step. Bad model IDs fail loudly and immediately.
- **The judge is structurally blind.** The judge user message contains task prompt + response + numbered criteria only — never the candidate's name. The candidate's identity lives in the filename (`grades/qwen-9b.jsonl`); the judge never sees it.

### Per-step contracts

| step | input | output | LLM calls |
| --- | --- | --- | --- |
| `load` | `data/ComplexConstraints.xlsx` | `data/complexconstraints.jsonl` (one JSON line per prompt with `id`, `prompt`, `use_case`, `instruction_type`, `prompt_style`, `criteria[]`) | 0 |
| `generate` | `data/complexconstraints.jsonl`, `CANDIDATES` registry | `responses/{model}.jsonl` (one line per prompt: `{id, response}`) | 1 per (prompt, candidate) |
| `grade` | prompts + responses + `prompts/judge.txt` | `grades/{model}.jsonl` (one line per prompt: `{id, verdicts[{index, verdict, reason}]}`) | 1 per (prompt, candidate) |
| `classify` | prompts + `prompts/classifier.txt` | `outputs/criteria_tags.jsonl` (one line per prompt: `{id, tags[{index, verifiability, gameable, reward_hack, ambiguous}]}`) | 1 per prompt |
| `validate --mode sample` | grades + responses | `outputs/judge_validation.json` (60 fixed-seed rows for human grading: model, id, criterion, judge_verdict, human="") | 0 |
| `validate --mode score` | filled `judge_validation.json` | merges `judge_agreement` block into `outputs/run_manifest.json` | 0 |
| `aggregate` | everything above | `outputs/results.json` (dashboard handoff) + `outputs/run_manifest.json` (run metadata) | 0 |

## Experimental process

The intended workflow for any new run is:

1. **Verify model IDs** — `connectivity` catches stale or mistyped IDs against the providers.
2. **Smoke first** — `all --limit 3` exercises every step end-to-end on the cost of a coffee. Catches schema bugs, provider quirks (Qwen reasoning-mode is *the* gotcha; see `meta/2026-06-30-smoke-test.md`), JSON parse failures, and timing surprises before they cost real hours.
3. **Full run** — `all --limit 75` runs sequentially; resumable. Roughly $50–60 and 4–6 hours wall-clock on a fresh run.
4. **Human-validate the judge** — hand-grade the 60 rows in `outputs/judge_validation.json`, run `validate --mode score`. The agreement % is the credibility keystone; the bias-audit deliverable depends on it.
5. **Re-aggregate** — `aggregate --limit 75` to fold the agreement number into `run_manifest.json` (results.json doesn't change).
6. **Write a run report** — copy `meta/TEMPLATE.md` to `meta/YYYY-MM-DD-<run-name>.md` and fill it. Sections include scope, attempt narrative, configuration deltas, results with explicit n caveats, an experimental-flaws audit, and a suggested-next-steps list.

Each step is one module under `pipeline/`. Add a candidate by appending
to `CANDIDATES` in `config.py`. Swap the judge by editing the `JUDGE`
constant and re-running `grade` (responses don't need regeneration; the
pipeline is decoupled at the JSONL boundary).

## Design choices worth knowing

- **Greedy decoding** (`temperature=0`) for both candidates and judge. Reproducibility over realism — a re-run gives the same outputs. The decoding-choice caveat is on every run report.
- **Reasoning disabled** on Qwen3.5 via OpenRouter's `reasoning.enabled: false`. Forced by the empirical discovery in the v1 smoke that thinking-mode burned the entire token budget on internal CoT and emitted zero visible content on every complex prompt. The pipeline now tests these models' no-CoT instruction-following, which is the realistic production-deployment condition for many users. The diagnostic is in `pipeline/generate.py` and the full account in `meta/2026-06-30-smoke-test.md`.
- **`max_tokens = 8000`** for candidate calls — empirically the smallest value at which complex responses finish naturally (`finish_reason: stop`) rather than truncating.
- **Defensive JSON extraction** for judge and classifier outputs — strip code fences, parse, then fall back to balanced-bracket scanning; retry once on parse failure; fall back to all-FAIL with a clear `judge_parse_error` reason on persistent failure. A flaky judge call never crashes the batch.
- **Broad retry predicate** — 429s, 5xx, network errors, *and* SDK-level parse errors (OpenRouter has been observed returning mid-stream-truncated JSON bodies). Exponential backoff with jitter.
- **Sequential by design for v1.** No parallelism between candidates or prompts. The simple model is correct; parallelism is on the suggested-next-steps list.

## Experiment tagging + dashboard handoff

Every `aggregate` step can be tagged with an **experiment slug** so the
dashboard's dropdown has a stable identifier for each run. The
convention is:

```
E<NN>-<kebab-case-label>
```

The two-digit number gives dropdown ordering. The label is 1–3
kebab-case tokens that hint at what makes the experiment different:

| slug | meaning |
| --- | --- |
| `E01-smoke-3p` | first smoke, 3 prompts |
| `E02-v1-75p` | v1 full run, 75 prompts |
| `E03-reasoning-on` | reasoning-enabled ablation |
| `E04-judge-swap` | judge replaced by a non-Anthropic model |
| `E05-cross-family` | DeepSeek added as a fourth candidate |

Passing `--experiment SLUG --description "..." --run-report meta/..."`
does three things:

1. Assembles the standardized deliverable — schema version `1.0`, `meta` block (experiment identity, dataset provenance, models array, judge block, counts, per-category populations, config snapshot, git state, judge-validation status), the six aggregate summaries, and the per-prompt array with per-model verdicts + reasons. Full field-by-field contract in [`meta/RESULTS_SCHEMA.md`](meta/RESULTS_SCHEMA.md).
2. Writes a per-experiment copy under `outputs/experiments/<slug>.json` — the sole file the dashboard reads for that experiment.
3. Updates `outputs/experiments/index.json` — a compact registry of every tagged experiment (slug, number, label, description, counts, validation status, run report link). The dashboard reads this for the dropdown, ordered by experiment number.

Untagged runs still write `outputs/results.json` (same shape) but do not
appear in the dashboard dropdown.

Concrete example (the current smoke, backfilled):

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

Every step honors `--limit`. `all` *requires* `--limit` — there is no default-to-full-set behavior, so production runs are always explicit.

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
  aggregate.py            # join all artifacts → results.json
  connectivity.py         # pre-flight model-ID check
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
outputs/                  # dashboard handoff + manifest + validation file; gitignored
  results.json
  run_manifest.json
  criteria_tags.jsonl
  judge_validation.json

meta/                     # run reports
  TEMPLATE.md             # copy + fill for every new run
  YYYY-MM-DD-<run>.md     # one per logical experiment
```

## Documentation

- **[`meta/RESULTS_SCHEMA.md`](meta/RESULTS_SCHEMA.md)** — the dashboard input contract. Field-by-field spec for `outputs/experiments/<slug>.json` and `outputs/experiments/index.json`, plus the versioning policy for shape changes. The dashboard should read this file and nothing else to know what it's binding against.
- **[`meta/TEMPLATE.md`](meta/TEMPLATE.md)** — canonical run-report shape. Every section is required, including the experimental-flaws audit and the suggested-next-steps list. Sections: TL;DR, scope, initial configuration, attempt-by-attempt run timeline, configuration adjustments + justifications, models evaluated, headline results (with explicit n caveats), experimental flaws and biases, cost & timing, output schema, lessons, next experiments (structured hypothesis tests), suggested next steps (procedural improvements), judge validation, reproducibility, open questions.
- **[`meta/2026-06-30-smoke-test.md`](meta/2026-06-30-smoke-test.md)** — first end-to-end exercise of the pipeline (3 prompts, slug `E01-smoke-3p`). Documents the two reliability surprises that surfaced (Qwen reasoning-mode trap, OpenRouter mid-stream JSON truncation), the configuration adjustments made in response, and a 15-item next-steps list. Read this before extending the pipeline or kicking off a new full run.
- The pre-implementation runbook (`ComplexConstraints-v1-master-runbook.md`) is the design spec and is gitignored — present locally as the source-of-truth for what v1 should be, intentionally not shipped.

## Roadmap

| | scope | status |
| --- | --- | --- |
| **v1** | Eval-only. Qwen size ladder. Blind Opus judge. Verifiability + gameability classifier. Judge validation against a human-graded sample. Single `results.json` for an external dashboard. Bias-audit deliverable. | pipeline complete; smoke green; awaiting full 75-prompt run + human validation |
| **v1.5** | Add a cross-family candidate (DeepSeek) to distinguish Qwen-specific failure modes from general open-model patterns. Add Anthropic prompt caching to cut judge cost. Possibly parallelize candidate generation. | proposed; tracked in `meta/2026-06-30-smoke-test.md` next-steps |
| **v2** | Training-time interventions. The eval surface stays; the candidates become outputs of a rubric-and-verifier RL pipeline. Out of scope for this repo. | future |

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
