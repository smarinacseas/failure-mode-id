# Failure Mode ID

Evaluation of open models (Qwen family) against the **Complex Constraints Benchmark Set** (Surge AI), graded by a blind Opus 4.8 judge. v1 is eval-only and produces a single `outputs/results.json` that a separate dashboard consumes. Includes a verifiability/reward-hack classifier and a judge-validation step against a human-graded sample.

## Quickstart

```bash
# 1. Provide API keys
cp .env.example .env  # then edit and add your keys

# 2. Connectivity check (fails fast on bad model IDs)
uv run python main.py connectivity

# 3. Smoke test on 3 prompts
uv run python main.py all --limit 3

# 4. Full run on all 75 prompts
uv run python main.py all --limit 75

# 5. Hand-grade outputs/judge_validation.json, then:
uv run python main.py validate --mode score
uv run python main.py aggregate --limit 75
```

Every step takes `--limit N` and is resumable — re-running skips IDs already
present in its output file. `all` requires an explicit `--limit` so a full
run is never accidental.

## Pipeline

| Step | Output | Purpose |
| --- | --- | --- |
| `load` | `data/complexconstraints.jsonl` | xlsx → one JSONL row per prompt |
| `generate` | `responses/{model}.jsonl` | candidate responses via OpenRouter (greedy) |
| `grade` | `grades/{model}.jsonl` | blind Opus judge, one call per response |
| `classify` | `outputs/criteria_tags.jsonl` | verifiability / gameable tags per criterion |
| `validate` | `outputs/judge_validation.json` | fixed-seed 60-row sample for human grading |
| `aggregate` | `outputs/results.json` | dashboard handoff |

## Data & Attribution

This project evaluates models against the **Complex Constraints Benchmark Set**,
released by Surge AI under CC-BY-4.0.

- Source: <https://huggingface.co/datasets/surgeai/ComplexConstraints>
- License: CC-BY-4.0 (<https://creativecommons.org/licenses/by/4.0/>)

### Citation

```
@misc{surge_complex_constraints,
  title  = {Complex Constraints Benchmark Set},
  author = {Surge AI},                            # TODO: confirm full author list / handles
  year   = {2024},                                # TODO: confirm release year
  howpublished = {\url{https://huggingface.co/datasets/surgeai/ComplexConstraints}},
  note   = {Licensed under CC-BY-4.0},
}
```

### Modifications

Generated model responses, derived per-criterion pass/fail grades via an LLM
judge, and classified criteria for verifiability and reward-hackability.
These derived artifacts and all code in this repo are MIT-licensed (see
`LICENSE`). Per CC-BY-4.0, prompts and criteria from the benchmark remain
attributable to Surge AI.
