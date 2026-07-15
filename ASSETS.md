# Repo assets

Where the assets the E-series and T-series runbooks depend on actually live
(plan Appendix B / Appendix D). Paths are repo-relative.

## Evaluation instrument (E08 = census, reused by T01 Tier-2)

| Asset | Path | Notes |
| --- | --- | --- |
| CC-75 prompts + criteria | `data/complexconstraints.jsonl` | 75 prompts; each row: `id, prompt, use_case, instruction_type, prompt_style, criteria[]` (~1,500 criteria total). Source: Surge AI Complex Constraints (CC-BY-4.0). |
| CC-75 source workbook | `data/ComplexConstraints.xlsx` | Original spreadsheet; `pipeline/load.py` derives the JSONL. |
| Results schema | `meta/RESULTS_SCHEMA.md` | The dashboard JSON contract (schema 3.3) emitted by `aggregate`. |

## Pipeline (candidate → judges → consensus → classifier → dashboard)

| Stage | Path | Role |
| --- | --- | --- |
| Orchestrator (CLI) | `main.py` | `uv run python main.py <stage> --experiment <slug>`; `all` chains every stage. |
| Per-run config freeze | `pipeline/run_config.py` | Freezes params to `runs/<slug>/experiment.json`; owns the E08 `tiebreaker_judge`, `provider_quantizations`, `seed` knobs. |
| Candidate generation | `pipeline/generate.py` | Streamed, deadline-guarded, resumable; captures serving `provider` per response (§0.2). |
| Judge grading | `pipeline/grade.py`, `pipeline/_judge_llm.py` | One blind judge call per (judge, prompt, response); Anthropic-batch + OpenRouter-pool transports; refusal/truncation/parse → artifact reason. |
| Legacy consensus | `pipeline/_consensus.py` | N-judge majority + Fleiss κ + abstentions; ties/no-quorum → FAIL (E01–E07). |
| **E08 panel policy** | `judging/panel.py` | Opus tie-break, EXCLUDE (undecidable/under-quorum, never FAIL), completeness gate (§0.3.3-5). `dispatch_consensus` selects legacy vs E08 per run. |
| Criterion classifier | `pipeline/classify.py`, `prompts/classifier.txt` | Verifiability (auto vs judge) + gameability tags. |
| Root-cause classifier | `pipeline/diagnose.py`, `pipeline/_taxonomy.py`, `prompts/judge.txt` | Blinded root-cause labels for consensus-FAIL criteria (EXCLUDE criteria are skipped under the E08 policy). |
| Decode-health detector | `pipeline/_decode_health.py` | Repetition-loop / truncation / length flags. |
| Aggregator + dashboard emit | `pipeline/aggregate.py`, `pipeline/_experiment.py`, `scripts/dashboard_sync.py` | Joins everything into `outputs/experiments/<slug>.json` + `outputs/results.json`; syncs to `dashboard/`. |
| Dashboard | `dashboard/` | Static viewer (`index.html`, per-run JSONs, `index.json`). |

## Per-run artifacts

`runs/<slug>/` — `experiment.json` (frozen params), `responses/<key>.jsonl`,
`grades/<judge>/<candidate>.jsonl`, `diagnosis/<candidate>.jsonl`,
`criteria_tags.jsonl`, `run_manifest.json`, `NOTES.md`. Every stage is resumable
(rerun the same command; only missing work executes).

## Verifier sources (T01 — verifier library, Day 1–2)

| Resource | Path | Role |
| --- | --- | --- |
| IFEval | `data/verih/Eval/evals/ifeval/` | ~25 verifiable instruction types; Tier-3 eval prompts. |
| IFBench (Ai2) | `data/verih/Eval/evals/ifbench/` | Harder verifiable constraint types (verify license at build). |
| IHEval | `data/verih/Eval/evals/iheval/` | Instruction-hierarchy eval. |
| math / mmlu | `data/verih/Eval/evals/{math,mmlu}/` | Additional eval harnesses. |

## Training track (T01 — contingent on Gate E→T)

| Asset | Path | Notes |
| --- | --- | --- |
| RLVR training loop | `training/train_rlvr.py`, `training/_rlvr_resume.py` | From the T01-ihrlvr pilot; durable checkpoints + resume. |
| Reward / verifier wiring | `training/reward.py`, `training/verih_reward.py` | Verifier-fraction reward wrappers. |
| Candidate proxy | `training/proxy.py`, `config.proxy_client` | `proxy://<name>` candidates route to a local vLLM/served model instead of OpenRouter. |
| IHEval driver | `training/iheval.py` | Live eval gate during training. |

## Run reports

`meta/*.md` — one standardized report per experiment (template: `meta/TEMPLATE.md`).
Latest first; `meta/2026-07-13-t01-ihrlvr.md` is the training pilot writeup.
