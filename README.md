# Failure Mode ID

**A criterion-level analysis of where open language models fail on constraint-dense prompts.**

Live results: <https://smarinacseas.github.io/failure-mode-id/>

A resumable eval pipeline that runs Surge AI's **Complex Constraints** benchmark
through the Qwen3.5 size ladder, grades every criterion with a blind Opus 4.8
judge, classifies criteria for verifiability and reward-hackability, and emits a
schema-versioned `results.json` for the ConstraintLens dashboard.

**v1 is eval-only** — no training, no fine-tuning. The deliverable is a clear,
bias-audited picture of *where* current open models fail, with an honest
limitations accounting attached. Training-time interventions are v2.

## What it measures

Open models satisfy individual instructions at high rates but struggle when a
prompt piles constraints on top of each other. The headline metric is the
**constraint-satisfaction gap**: a high per-criterion pass rate alongside a low
full-prompt pass rate — the model passes *most* constraints on *most* prompts,
but rarely *all* constraints on any *one* prompt.

v1 is built to characterize that gap three ways:

1. **Across a size ladder** — Qwen3.5 9B → 35B → 397B, one family, no
   architectural confounds within the ladder.
2. **By failure mode** — broken out by `instruction_type`, `prompt_style`, and
   `use_case` (see [Glossary](#glossary)).
3. **Signal vs. judge artifact** — by hand-grading a 60-criterion sample for
   judge-vs-human agreement and splitting pass rates by criterion verifiability
   (deterministically checkable vs. judge-dependent).

## Status (2026-07-01)

- **Pipeline: complete.** Every step (`load → generate → grade → classify →
  validate → aggregate`) is implemented, resumable, and end-to-end validated.
- **Evaluated so far: 3 prompts of 75.** `E01-smoke-3p` exercised the whole
  pipeline on the first three prompts across the ladder → 216 grade cells. The
  full `E02-v1-75p` run has not been kicked off.
- **Judge validation: sampled, not scored.** The 60-row human-grading file
  exists but is ungraded, so `judge_agreement` is `null`. **Read every headline
  number with that caveat.**
- **Dashboard: live**, currently surfacing `E01-smoke-3p` as its only run.

## The benchmark — Complex Constraints

[Surge AI's Complex Constraints Set](https://huggingface.co/datasets/surgeai/ComplexConstraints)
(CC-BY-4.0) is 75 prompts × up to 40 criteria each. Each prompt is a realistic
constraint-heavy request — a staffing rota with overlapping union/age/shift
rules, a data-formatting task whose arithmetic must reconcile, a logistics plan
with explicit prohibitions — and each criterion states one narrow,
literally-judgeable condition the response must satisfy.

Every prompt is tagged along three axes that the analysis breaks out
(`instruction_type`, `prompt_style`, `use_case`); the classifier adds two more
per criterion (`verifiability`, `gameable`). Definitions in the [Glossary](#glossary).

The set skews toward constraint-heavy planning and structured-data tasks —
findings extrapolate to *that* class of task, not to chat or open-ended QA.

## How it works

A linear, resumable pipeline driven by a single CLI:

```
   xlsx                                          dashboard
    │                                                ▲
 ┌──▼──┐  ┌──────────┐  ┌───────┐  ┌────────┐  ┌────┴────┐
 │load │→ │ generate │→ │ grade │→ │classify│→ │aggregate│
 └─────┘  └────┬─────┘  └───┬───┘  └────┬───┘  └────┬────┘
               │            │           │           │
          responses/    grades/    criteria_tags  results.json
                                                     ▲
                                              ┌──────┴─────┐
                                              │  validate  │  ← judge_validation.json
                                              └────────────┘     (60-row human sample)
```

- **Candidates run via [OpenRouter](https://openrouter.ai/)** (three Qwen
  variants, swappable IDs); **judge + classifier run via Anthropic** (Opus 4.8).
- **The judge is structurally blind** — it sees task prompt + response +
  numbered criteria only, never the candidate's identity (that lives in the
  filename). A non-candidate-family judge means self-preference bias is absent
  by design.
- **Every step is resumable per-id** — re-running skips completed IDs, so a
  mid-run crash never wastes prior work.
- **`all` requires an explicit `--limit`** — a full 75-prompt run is always
  opt-in. A pre-flight `connectivity` check fails loudly on bad model IDs
  before any batch step.

Per-step I/O contracts and the exact `results.json` shape live in
[`meta/RESULTS_SCHEMA.md`](meta/RESULTS_SCHEMA.md).

## Quickstart

```bash
uv sync                                    # dependencies (pyproject + uv.lock)
cp .env.example .env                       # add OPENROUTER_API_KEY + ANTHROPIC_API_KEY

uv run python main.py connectivity         # pre-flight: fails fast on bad model IDs
uv run python main.py all --limit 3        # smoke: 3 prompts × 3 candidates, ~10–20 min
uv run python main.py all --limit 75       # full run: ~4–6 hr, ~$50–60

# then hand-grade outputs/judge_validation.json (60 rows) and fold in the score:
uv run python main.py validate --mode score
uv run python main.py aggregate --limit 75
```

```
steps:  load | generate | grade | classify | validate | aggregate | all | connectivity
```

Add a candidate by appending to `CANDIDATES` in `config.py`; swap the judge by
editing `JUDGE` and re-running `grade` (responses don't need regenerating — the
pipeline is decoupled at the JSONL boundary).

## Design choices worth knowing

- **Greedy decoding** (`temperature=0`) for candidates and judge —
  reproducibility over realism; a re-run gives identical outputs.
- **Reasoning disabled on Qwen3.5.** The smoke found thinking-mode burning the
  entire token budget on internal CoT and emitting *zero* visible content on
  complex prompts. v1 therefore measures no-CoT instruction-following — the
  realistic production condition for many deployments. Full account in
  [`meta/2026-06-30-smoke-test.md`](meta/2026-06-30-smoke-test.md).
- **`max_tokens = 8000`** — empirically the smallest cap at which complex
  responses finish naturally rather than truncating.
- **Defensive JSON extraction + broad retry** for judge/classifier output
  (strip fences → parse → balanced-bracket fallback → retry → all-FAIL with a
  clear reason). A flaky call never crashes the batch.

## Experiments & dashboard

Each `aggregate` can be tagged with an experiment slug (`E<NN>-<label>`), which
assembles a versioned per-experiment deliverable under
`outputs/experiments/<slug>.json`, updates the registry `index.json`, and syncs
the [dashboard](https://smarinacseas.github.io/failure-mode-id/). Untagged runs
still write `outputs/results.json` but don't appear in the dropdown.

| slug               | meaning                                  | status              |
| ------------------ | ---------------------------------------- | ------------------- |
| `E01-smoke-3p`     | first smoke, 3 prompts                   | **done, live**      |
| `E02-v1-75p`       | v1 full run, 75 prompts                  | planned             |
| `E03-reasoning-on` | reasoning-enabled ablation               | planned (v1.5)      |
| `E04-judge-swap`   | judge replaced by a non-Anthropic model  | planned (v1.5)      |
| `E05-cross-family` | DeepSeek added as a fourth candidate     | planned (v1.5)      |

The dashboard surfaces per-criterion pass rates broken out by each axis, a
per-model comparison, a per-prompt drilldown, and a prompt-detail modal showing
each verdict with the judge's stated reason. It's a static bundle served from
`dashboard/` by GitHub Pages — no build step.

## Limitations

The honesty of an eval *is* the product, so read every number with these in mind:

- **n = 3 so far.** All current summaries derive from `E01-smoke-3p` — 3 prompts
  × 72 criteria × 3 models. Treat E01 as a "does the pipeline work" signal, not
  a model ranking.
- **Judge agreement is unmeasured.** Until the 60-row sample is hand-graded,
  judge-specific bias is unbounded — discount `verifiability="judge"` rates.
- **Single candidate family** (Qwen3.5 only). No cross-family control, so any
  pattern may be Qwen-specific. DeepSeek addition is on the v1.5 roadmap.
- **Greedy decoding + reasoning disabled.** Results describe the reproducible,
  no-CoT single-shot output — not the model's ceiling under sampling or CoT.
- **Benchmark skew + narrow tags.** Weighted toward planning/structured-data;
  2–3 values per axis mean small per-category denominators even at full scale.
- **API-driven & sequential.** External providers can drift between reruns; a
  75-prompt run is 4–6 hr wall-clock. Parallelization is a v1.5 next-step.

## Roadmap

| version  | scope                                                                                 | status                                       |
| -------- | ------------------------------------------------------------------------------------- | -------------------------------------------- |
| **v1**   | Eval-only: Qwen ladder, blind Opus judge, classifier, judge validation, bias audit.   | pipeline complete; awaiting full run + validation |
| **v1.5** | Cross-family candidate (DeepSeek), prompt caching, possible parallelization.           | proposed                                     |
| **v2**   | Training-time interventions (rubric-and-verifier RL). Out of scope for this repo.      | future                                       |

## Project layout

```
config.py          # API clients + CANDIDATES + JUDGE + paths
main.py            # CLI orchestrator
pipeline/          # one module per step: load, generate, grade, classify,
                   #   validate, aggregate, connectivity (+ shared _io / _json / _experiment)
prompts/           # judge.txt + classifier.txt (verbatim system-prompt contracts)
data/              # ComplexConstraints.xlsx (source benchmark)
outputs/           # results.json + manifest + validation file; experiments/ is committed
meta/              # run reports, RESULTS_SCHEMA.md, TEMPLATE.md
design/, dashboard/, scripts/   # ConstraintLens design bundle → unpacked static site
```

## Documentation

- [`meta/RESULTS_SCHEMA.md`](meta/RESULTS_SCHEMA.md) — the dashboard input
  contract; field-by-field spec for the per-experiment JSON (schema `2.1`).
- [`meta/TEMPLATE.md`](meta/TEMPLATE.md) — canonical run-report shape, including
  the required experimental-flaws audit and next-steps sections.
- [`meta/2026-06-30-smoke-test.md`](meta/2026-06-30-smoke-test.md) — the first
  end-to-end run; documents the Qwen reasoning-mode trap and OpenRouter
  mid-stream JSON truncation. **Read before extending the pipeline.**

## Glossary

Project vocabulary behind every tag pill on the dashboard.

**Prompt axes** (dataset-provided, one per prompt):

- **`use_case`** — the domain: *Data Processing, Formatting & Math* (structured
  output with reconciling arithmetic) or *Logistics, Scheduling & Event
  Planning* (multi-entity plans under real-world constraints).
- **`instruction_type`** — *Negative* (must-*not* constraints; canonical failure
  is introducing the forbidden element) or *Multistep* (sequenced operations
  where errors compound).
- **`prompt_style`** — *Direct* (clear numbered instructions), *Context*
  (embedded in a scenario), or *Rambling* (buried in noisy prose).

**Criterion axes** (classifier-assigned, one per criterion):

- **`verifiability`** — `auto` (a program can check it deterministically —
  judge-error immune) or `judge` (needs subjective reading — where
  judge-vs-human agreement matters most).
- **`gameable`** — whether the wording admits a shortcut that satisfies the
  literal criterion while violating intent (a reward-hack surface).

**Metrics:**

- **`criterion_pass_rate`** — share of (criterion × prompt) cells passed. The
  number most benchmarks report; optimistic on its own.
- **`full_prompt_pass_rate`** — share of prompts where *every* criterion passed.
  What a production workload actually needs.
- **Constraint-satisfaction gap** — the delta between the two above; v1's
  headline finding.
- **`judge_agreement`** — % of the 60-row human sample where judge matched
  human. The credibility keystone — **currently unmeasured** (see Status).

## Data & attribution

Evaluated against the **Complex Constraints Benchmark Set** by Surge AI, released
under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
([source](https://huggingface.co/datasets/surgeai/ComplexConstraints)).

Generated model responses, derived per-criterion pass/fail grades via an LLM
judge, and classified criteria for verifiability and reward-hackability. These
derived artifacts and all code are MIT-licensed (see [`LICENSE`](LICENSE)); per
CC-BY-4.0, the prompts and criteria remain attributable to Surge AI.

```
@misc{complex_constraints_benchmark,
  title  = {Complex Constraints Benchmark Set},
  author = {Surge AI},
  year   = {2025},
  url    = {https://huggingface.co/datasets/surgeai/ComplexConstraints}
}
```
