# Failure Mode ID

**Criterion-level analysis of where open language models fail on
constraint-dense prompts, and root-cause diagnosis of *why*, in training-
protocol terms.**

Live results: <https://smarinacseas.github.io/failure-mode-id/#run=E07-reasoning-full75&tab=analysis&model=qwen-9b&judge=claude-opus-4-8>

A resumable eval pipeline that runs Surge AI's [Complex Constraints](https://huggingface.co/datasets/surgeai/ComplexConstraints)
benchmark through a candidate-model ladder (default: Qwen3.5 9B→35B→397B via
OpenRouter), grades every criterion with a blind multi-family judge panel
(Claude, GPT, Gemini, DeepSeek by default), classifies
criteria for verifiability, **diagnoses every failed criterion with a blinded
root-cause taxonomy** (never noticed vs dropped-from-CoT vs executed wrong vs
judge-suspect …), and publishes a schema-versioned JSON that drives the
dashboard, including its Failure Analysis tab.

## Quickstart (~$2, ~20 minutes)

```bash
git clone https://github.com/smarinacseas/failure-mode-id.git
cd failure-mode-id
uv sync
cp .env.example .env        # add your OpenRouter + Anthropic keys

# 2-prompt smoke across the full ladder, judges, and failure diagnosis:
uv run python main.py all --experiment E90-myfirst-smoke --limit 2

# watch progress from another terminal
uv run python main.py status

# view your run in the dashboard
python3 -m http.server 8756 --directory dashboard
```

Every experiment freezes its parameters on first run
(`runs/<slug>/experiment.json`); every stage is resumable (rerun the same
command, only missing work executes).

## Make it yours

| Knob | Flag | Notes |
| --- | --- | --- |
| Candidate models | `--candidates qwen-9b,mymodel=vendor/model-id` | registry keys and/or any OpenRouter id |
| Judges | `--judges claude-opus-4-8,claude-fable-5,gpt-5,kimi=moonshotai/kimi-k3` | registry keys, key=OpenRouter-id pairs, or bare claude-* ids; Claude judges grade via Message Batches, others via a streamed pool |
| Classifier | `--classifier claude-fable-5,claude-opus-4-8` | criterion-classifier fallback chain (frozen), same syntax as `--judges`, walked per prompt; default: the first judge |
| Reasoning mode | `--reasoning on --max-tokens 48000 --temperature 0.6` | thinking + answer share the budget |
| Prompt sampling | `--limit 20 --sample-seed 42` | seeded stratified spread across use cases/types/styles |
| Provider routing | `--provider-sort throughput` | reasoning runs want the fast end of the pool |
| Judge transport | `--judge-mode batch` | Message Batches (default) or sequential |
| Failure diagnosis | `--diagnose off` | skip the root-cause stage; backfill later with `main.py diagnose --experiment <slug>` |

Replicate the flagship 20-prompt reasoning run (E05):

```bash
uv run python main.py all --experiment E05-replica --limit 20 \
  --sample-seed 20260706 --reasoning on --max-tokens 48000 \
  --temperature 0.6 --timeout 600 --provider-sort throughput \
  --judges claude-opus-4-8,claude-fable-5
```

## Pipeline

connectivity → load → generate → grade → classify → **diagnose** → validate → aggregate

Each stage is also runnable alone: `uv run python main.py <stage> --experiment <slug>`.
Artifacts live under `runs/<slug>/`; deliverables land in
`outputs/experiments/<slug>.json` and auto-sync to `dashboard/`. The
dashboard serves three pages: the run landing page at `/`, the failure
analysis eval app at `/eval.html`, and the T01 training write-up at
`/t01.html`.

- **generate**: streamed candidate calls, 4-worker pool, wall-clock deadline guard
- **grade**: one blind judge call per (judge, model, prompt); mixed Anthropic/OpenRouter panels supported; ≥2 judges produce consensus verdicts + agreement stats; refusal/truncation/parse failures recorded distinctly, never silently dropped
- **classify**: criterion verifiability (auto vs judge) + gameability tags
- **diagnose**: blinded root-cause labels for every criterion the panel consensus marks FAIL (walks a Fable-preferred fallback chain to Opus per cell, batch; the analyst never sees judge reasons, model identity, or any judge's verdicts; a reserved `judge_suspect` label licenses disagreement), plus an iteration synthesis comparing against the previous experiment and recommending the next one
- **validate**: human-validation sampling of judge verdicts
- **aggregate**: joins everything into the schema-versioned results JSON (see `meta/RESULTS_SCHEMA.md`)

## Reports

Every experiment ships a run report in [`meta/`](meta/) (template:
`meta/TEMPLATE.md`) covering configuration, timeline, results, flaws/biases,
and queued next experiments. Start with the latest.

## License / data

Benchmark: Complex Constraints Benchmark Set (Surge AI, CC-BY-4.0). Code: MIT.
