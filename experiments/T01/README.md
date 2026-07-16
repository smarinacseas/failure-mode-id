# T01 — method × cause interaction (SFT vs GRPO × coverage vs precision)

The first T-series training experiment. Tests whether **failure type determines
training method**: is GRPO's advantage over SFT larger for precision failures
(`execution_slip`) than for coverage failures (`constraint_unaddressed`)?

- **Design & hypotheses:** [`PREREG.md`](PREREG.md) — pre-registered, append-only.
  Bound at Gate E→T from the E08 census (`CAUSE_A`=coverage 28.8%,
  `CAUSE_B`=precision 36.4%). Primary: `Interaction > 0` (H1).
- **Pod setup:** [`POD_SETUP.md`](POD_SETUP.md) — RunPod A100 → Gate GT0.
- **Plan of record:** `docs/superpowers/plans/2026-07-15-e08-t01-lamma.md` (Part II).
- **Stack:** TRL (SFT + GRPO), PEFT LoRA (r=16, α=32, attn+MLP), `generate()`
  rollouts (**no vLLM** — GT0 env freeze; PREREG amendment 2026-07-16), on a
  single A100 80 GB.

## Layout

```
experiments/T01/
  PREREG.md            pre-registration (frozen at Gate E→T)
  POD_SETUP.md         pod bring-up → Gate GT0
  requirements.txt     TRL training stack (frozen lock: /requirements-t01.txt; no vLLM)
  smoke_test.py        Gate GT0 env validator (load · 5 gens · 1 LoRA step · imports)
  config/              frozen per-phase configs
  verifiers/           check(response, spec) -> {pass, detail}; coverage + precision pools (T1.1)
  datagen/             programmatic prompt composition + SFT teacher gen (T1.2)
  data/train/          300 train prompts per cause
  data/holdout/        200 holdout per cause — TRAINING CODE MUST NEVER READ THIS (test-enforced, T1.2)
  data/sft/            filtered 100%-verifier-pass teacher responses
  training/            SFT + GRPO runners; adapters → results/adapters/T01-{arm}/ (T1.3)
  eval/                Tier-1 (in-dist) · Tier-2 (CC-75 transfer) · Tier-3 (regression) (T1.4–T1.5)
  analysis/            cluster bootstrap, McNemar, interaction CI (T1.6)
  results/             adapters, per-criterion tables, dashboard JSONs
```

## Phase status

- [x] **Gate E→T** — causes bound, PREREG committed.
- [x] **T1.0** — scaffolding + pod env → **Gate GT0 PASSED** (training core validated on the A100; exact env locked in `/requirements-t01.txt`; **no vLLM** — GRPO trains on `generate()`, PREREG amendment 2026-07-16).
- [~] **T1.1** — verifier library → Gate GT1 (core coverage+precision pools + GRPO reward done, TDD; a `casing` verifier + the 20-hand-written-response check remain). ← *here*
- [ ] T1.2 — data generation → Gate GT2
- [ ] T1.3 — training (4 arms) → Gate GT3
- [ ] T1.4 — Tier-1 eval → Gate GT4
- [ ] T1.5 — Tier-2/Tier-3 → Gate GT5
- [ ] T1.6 — analysis + write-up → Gate GT6

Gates are sequential — no phase N+1 until phase N's gate passes. Long GPU runs
execute on the pod (`POD_SETUP.md`); verifier/datagen/analysis code is
GPU-free and can be built and unit-tested locally.
