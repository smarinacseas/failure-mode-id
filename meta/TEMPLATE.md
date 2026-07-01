<!--
This is a template for run reports in this directory. To use it:

  cp meta/TEMPLATE.md meta/YYYY-MM-DD-<short-run-name>.md

Then replace every italic placeholder line and HTML comment with your
actual content. The italics are visible when rendered, so leftover
placeholders signal "this section wasn't filled in."

Naming convention:
  2026-06-30-smoke-test.md          # first end-to-end smoke
  2026-07-02-full-run-v1.md         # the full 75-prompt run
  2026-07-15-deepseek-cross-family.md   # adding a cross-family candidate
  2026-08-01-judge-swap-haiku.md    # ablation: swap the judge model

Keep one file per logical experiment, dated. Don't edit prior reports;
write a new one and link to the earlier one if you're following up.
-->

# <run name> · <YYYY-MM-DD> · `<exact command run>`

**Experiment slug:** `E<NN>-<kebab-case-label>` — the identifier under which this
run appears in the dashboard dropdown and in `outputs/experiments/index.json`.
Convention: two-digit zero-padded number for ordering, then a 1–3-token
kebab-case label that hints at the axis under investigation (e.g.
`E01-smoke-3p`, `E02-v1-75p`, `E03-reasoning-on`, `E04-judge-swap`). Slug is
passed to `main.py` via `--experiment` alongside `--description` and
`--run-report`; the full `config` snapshot (candidate knobs, judge, prompt
SHAs, git commit) is captured automatically by aggregate.


*One-paragraph framing: what was the goal of this run, what's the relationship to the previous run, and what would success look like? Should be readable as the abstract of the report — someone scanning the directory should learn from this paragraph alone what the run was for.*

## TL;DR

*Three to five bullets. The single most important findings, the single most important surprises, and the single most important caveats. If a reader stops here, what do they need to know?*

- *…*
- *…*
- *…*

## Scope

| | |
| --- | --- |
| Sample size | *N of M prompts* |
| Total criteria graded | *count* |
| Prompt diversity | *e.g. "3 use cases · 4 instruction types · 3 prompt styles"* |
| Candidates | *list the model keys* |
| Judge / classifier | *model id* |
| Decoding | *temperature, top_p, etc.* |

*If the prompt mix is non-representative (e.g. first-N rows of the xlsx, only one instruction_type, only one use_case), flag it here in one sentence. n-small breakdowns later in the report inherit this caveat.*

Prompts in this run (or a representative subset if N is large):

| id | use_case | instruction_type | prompt_style | n_criteria |
| --- | --- | --- | --- | --- |
| *CIF-XXX* | *…* | *…* | *…* | *…* |

## Initial configuration (as specified)

*Paste the exact code snippet that captured the starting state — `config.py` values, `generate.py` call, judge prompt path, classifier prompt path, retry predicate, any extra_body params, max_tokens, temperature, timeout. If this run inherits from the previous run, link to that report's "Final configuration" instead and only list what changed.*

```python
# <paste the relevant snippet here>
```

## Run timeline

*Narrate the run as a story, attempt by attempt. The point is to make the experimentation process legible — what was tried, what broke, what was learned, what changed. Include hard numbers (latency, token counts, error messages, finish_reason values) so the next reader can verify the diagnoses.*

### Attempt 1 — <one-line outcome>

*What was the configuration. What happened. What the on-disk evidence showed (file sizes, content lengths, finish_reason values, usage objects). What hypothesis explained it.*

### Attempt 2 — <one-line outcome>

*What was changed and why. The diagnostic experiments run, with numbers. The decision and its justification.*

<!-- Add or remove attempt subsections as needed. -->

## Configuration adjustments + justifications

| Change | From | To | Why |
| --- | --- | --- | --- |
| *param name* | *prior value* | *new value* | *what evidence motivated the change* |

**Not** changed:

- *List the things you explicitly considered changing and chose to leave alone, with the reasoning. Useful for the next experimenter who'll wonder "why didn't they change X?"*

## Models evaluated

| key | provider ID | role |
| --- | --- | --- |
| *qwen-9b* | *qwen/qwen3.5-9b* | *candidate (small)* |
| *…* | *…* | *…* |
| *judge* | *claude-opus-4-8* | *grader + classifier* |

*If models were added or removed compared to the previous run, call that out here.*

## Headline results

*Lead with the single most important quantitative finding. Caveat n explicitly — directional findings need to be labeled as such.*

### Aggregate

| metric | *model A* | *model B* | *model C* |
| --- | --- | --- | --- |
| criterion pass rate | *…* | *…* | *…* |
| full-prompt pass rate | *…* | *…* | *…* |

*One-sentence interpretation. Is the constraint-satisfaction gap visible? Did scaling help / hurt / not move the needle?*

### Per-prompt (or per-category, depending on N)

| | *model A* | *model B* | *model C* |
| --- | --- | --- | --- |
| *CIF-XXX (…)* | *x/y* | *x/y* | *x/y* |

*Flag any leads worth investigating — pattern reversals, scale anomalies, cells that look out of distribution. Be explicit about what's a finding vs. what's a hypothesis.*

### Verifiability split

| | auto (deterministic) | judge (subjective) |
| --- | --- | --- |
| *model A* | *%* | *%* |
| *…* | *…* | *…* |

*Comment on whether the gap between auto and judge categories looks plausible or suggests judge bias / over-confidence.*

## Experimental flaws, biases, and limitations

*The headline numbers above are point estimates from this run's specific
configuration. State explicitly what would qualify or invalidate them
before they get carried into the dashboard or the writeup. Every report
must fill this section even when "no new biases relative to prior runs"
is the honest answer — reproducibility requires the absence of bias to
be stated as deliberately as its presence.*

### Sample-size + selection effects

*State the actual N. Are by-category cells meaningfully populated? Was the
sample randomized or first-N? Is the effective diversity (after deduping
the (use_case, instruction_type, prompt_style) tuples) lower than the
nominal N? Was each cell measured once or multiple times?*

### Judge biases

*For each of self-preference, verbosity, position, format, family-stake:
controlled / measured / unmeasured / not-applicable. Cite the evidence
or note its absence. If `Phase 6` validation was run, link the agreement
number from "Judge validation" below.*

### Classifier biases

*Same Opus model running both grading and classification → correlated
artifacts. State whether any classifier outputs were hand-checked, and
how many. Note whether the `auto` verifiability tag was actually
operationalized via a deterministic checker or remains descriptive only.*

### Decoding-mode caveats

*What decoding configuration was used (temperature, top_p, max_tokens,
reasoning-mode). Which production deployment modes does this configuration
**not** generalize to. Did any responses hit the token cap?*

### Anomaly hypotheses (not yet ruled in or out)

*For each per-category cell or per-prompt result that looks surprising,
list the plausible explanations — "real signal" + the most likely
artifact-driven alternatives. Future runs distinguishing between them
goes in "Next experiments" or "Suggested next steps".*

| Observation | Real? | Alternative explanations |
| --- | --- | --- |
| *…* | *…* | *…* |

### Provider-side uncontrolled variance

*OpenRouter routing, Anthropic API load, time-of-day, provider-side
caching state. Any reliability events observed (timeouts, malformed
responses, rate-limit hits) and whether they reproduce.*

### Missing validation

*Anything the v1 plan calls for that this run did not do. The most
common entry is "Phase 6 not yet run for this batch" — say so explicitly
even when it's obvious from the absence of a Judge-validation section.*

### Benchmark-internal caveats (inherited)

*Caveats that apply to every run because they live in the source data:
domain skew, language coverage, missing demographic data, IP /
attribution constraints on what can be published. Inherit by reference
from the prior report unless something changed.*

## Cost & timing

*Use per-call averages from the run to extrapolate. Include both wall-clock and dollar figures. If different from the previous run, explain the delta.*

| | per call (this run avg) | full run (×N) | total |
| --- | --- | --- | --- |
| generate (provider) | *$ / s* | *× count* | *$ · time* |
| grade (judge) | *$ / s* | *× count* | *$ · time* |
| classify (judge) | *$ / s* | *× count* | *$ · time* |
| **total** | | ***N* LLM calls** | *$ · time* |

*Note whether prompt caching, batching, or parallelism would change these numbers significantly, and whether you implemented any.*

## Output schema (what gets produced)

*The dashboard-facing deliverable shape is fixed by [`meta/RESULTS_SCHEMA.md`](RESULTS_SCHEMA.md).
This section is for run-specific artifacts (sidecar files, ad-hoc dumps,
regenerated intermediates). Note any deviation from the schema — an
added optional field, a `schema_version` bump, an unusual sidecar
produced only by this experiment. Otherwise write "unchanged — deliverable
matches `meta/RESULTS_SCHEMA.md` at v<schema_version>".*

| path | content |
| --- | --- |
| *outputs/experiments/<slug>.json* | *deliverable — see schema doc* |
| *outputs/experiments/index.json* | *dashboard dropdown source — this experiment appended / replaced* |

## Lessons / notes for the next experimenter

*The single highest-value section for a future reader. What would you tell yourself if you were about to start this same run? Include both technical gotchas and process observations. Number them so they can be cited.*

1. *…*
2. *…*

## Next experiments

*Structured proposals for what should run next. The goal is experimental
soundness: each entry should make a specific, falsifiable claim and pre-register
what result would actually update our beliefs. Vague follow-ups belong in "Open
questions / follow-ups" below — this section is for things that have been
designed enough to run.*

**Hygiene rules (apply to every entry below):**

1. **One variable per experiment.** If you change the candidate set AND the judge AND `max_tokens` in the same run, you cannot attribute a delta to any single cause. Run ablations one knob at a time.
2. **Pre-register the prior.** State the expected outcome before the run. If you can't articulate a specific expectation, you're exploring, not testing — label it as such.
3. **Pre-register the falsifying outcome.** What result would surprise you? If "any result fits my story," the experiment will not inform you; redesign or drop it.
4. **Distinguish confirmatory vs exploratory.** Confirmatory: replicate a prior finding at higher N or under perturbation. Exploratory: probe a hypothesis you have not yet tested. Both are valid; mislabeling them is not.
5. **Smoke before full.** Every new configuration knob gets a `--limit 3` shake-out before the full set. The cost is 10 minutes; the cost of skipping it was already established in `2026-06-30-smoke-test.md`.
6. **Save state before patching.** If an experiment requires code changes, commit the prior state first so the experiment can be reproduced or rolled back. Tag the commit referenced from this report.

*Entry template — copy for each proposed experiment:*

### Experiment N — *<descriptive name>*

- **Type**: *confirmatory | exploratory | ablation | replication*
- **Hypothesis**: *the specific, falsifiable claim being tested. Not "investigate X" but "X correlates with Y under condition Z."*
- **Prior**: *what outcome you expect before running, with rough magnitudes if possible. "qwen-397b will pass at least 5 pp more criteria than qwen-9b on Multistep."*
- **What would change my mind**: *the specific outcome that would update belief away from the prior. "If the pp delta is < 2 or reverses, scaling does not help Multistep instruction-following at this scale."*
- **Operationalization**: *the exact change to make — config edit, candidate add, judge swap, prompt mix change, seed change — and the exact command to run.*
- **Cost / wall-clock**: *rough $ and hours. Use cost-and-timing extrapolations from the most recent full run.*
- **Priority**: *H / M / L, with one-line reason. Reserve H for experiments that could falsify a load-bearing finding.*
- **Depends on**: *prior runs, code commits, or data this experiment requires before it can be run. Empty if it can run today.*

*Example entries demonstrating the form (delete or adapt for your run):*

### Experiment 1 — *Verify CIF-001 inverse scaling*

- **Type**: *confirmatory replication*
- **Hypothesis**: *On Negative · Context · Logistics prompts, criterion-pass rate decreases monotonically with candidate size (9b > 35b > 397b).*
- **Prior**: *Smoke (n=1) showed 13 → 7 → 4 of 19. Expectation on full sample: same direction, but smaller magnitude — likely 9b ~10–15 pp ahead of 397b.*
- **What would change my mind**: *If the ranking on Negative · Context prompts has no significant gradient (all three within 5 pp) or reverses (397b ≥ 9b), the smoke finding was an n=1 artifact, likely driven by 397b's "no final coherent rota" failure mode being prompt-specific.*
- **Operationalization**: *No code change. Run the full set, then filter `results.json.prompts[]` to instruction_type == "Negative" AND prompt_style == "Context prompting" and recompute per-model pass rate over that subset.*
- **Cost / wall-clock**: *Subsumed by the full run (~$55, ~4.5 hr) — no incremental cost.*
- **Priority**: *H — load-bearing for the scaling-story section of the writeup.*
- **Depends on**: *Full 75-prompt run completing.*

### Experiment 2 — *Reasoning-enabled ablation*

- **Type**: *ablation*
- **Hypothesis**: *Re-enabling Qwen3.5 reasoning at `max_tokens=32000` produces strictly higher criterion-pass rates than the no-CoT condition currently used, with the largest gain on Multistep prompts.*
- **Prior**: *+5 to +15 pp on criterion-pass, concentrated on Multistep · Data-Math. Full-prompt pass rate may still be ~0 % because the gating is constraint-satisfaction, not per-constraint accuracy.*
- **What would change my mind**: *If criterion-pass is within 2 pp of the no-CoT condition, the v1 deviation costs us nothing and the reasoning-disabled choice is vindicated. If reasoning hurts, that itself is a publishable finding.*
- **Operationalization**: *Smoke first with `--limit 3` at the new config to confirm responses are non-empty. Then run the full set with `EXTRA_BODY={}` and `MAX_TOKENS=32000` in `pipeline/generate.py`. Write to `responses-reasoning/` and `grades-reasoning/` to avoid clobbering the v1 results.*
- **Cost / wall-clock**: *~3–5× the v1 run on the candidate side (longer outputs), judge side unchanged. Ballpark $80–$120 and ~8–12 hr.*
- **Priority**: *M — informs the bias-audit section's "decoding choice" subsection. Defer until v1 is shipped.*
- **Depends on**: *v1 full-run results.json (the baseline to compare against).*

### Experiment 3 — *Judge self-preference cross-check*

- **Type**: *ablation*
- **Hypothesis**: *Swapping the judge from `claude-opus-4-8` to a non-Anthropic model (e.g. `openai/o4` or `deepseek/deepseek-v4`) shifts per-criterion verdicts by less than the human-vs-Opus disagreement number from Phase 6.*
- **Prior**: *Verdict shift < Phase 6 disagreement %, because the candidates are non-Anthropic and the Opus judge has no family stake. Decisive evidence against self-preference bias if the prior holds.*
- **What would change my mind**: *If swapping the judge shifts verdicts by more than the human disagreement number, the eval is judge-dependent in a way that needs to be flagged prominently in the limitations panel.*
- **Operationalization**: *Add a `--judge MODEL_ID` flag to `pipeline/grade.py`. Re-grade the existing responses with the alternate judge (no re-generation needed — candidates' outputs are already saved). Diff the two grade files at the (model, id, criterion_idx) level.*
- **Cost / wall-clock**: *One full set of judge calls, ~$30–$50 depending on alternate judge price, ~1.5 hr.*
- **Priority**: *H — directly supports the bias-audit deliverable.*
- **Depends on**: *v1 full-run grades/ files (responses are re-graded, not re-generated).*

*Add experiments below as they get designed. When an experiment is run, link from its entry to the resulting report file in this directory and mark it `[run YYYY-MM-DD → 2026-MM-DD-<run-name>.md]`. Do not delete completed entries — they are the lineage.*

## Suggested next steps

*A flat, scoped-small list of concrete actions that advance the program
from where this run leaves it. Distinct from "Next experiments" above:
that section is for designed experiments testing a hypothesis. This
section is for everything else worth doing — instrumentation upgrades,
methodological improvements, follow-up validations, infrastructure work.
Items should be individually small enough to do in one sitting; if an
item needs decomposition, decompose it.*

*Entry format — concrete action + brief rationale + what it improves:*

1. ***<action verb> <object>*** *(<rough effort estimate>).* ***Why:*** *what gap or risk this addresses and what becomes possible / safer / cheaper / more credible afterward.*
2. ***…***

*When an item gets done, link from its bullet to the resulting commit,
run report, or issue — `[done → <hash>]` or `[done → <report.md>]` —
rather than deleting it. The history of what got tried (and what
didn't) is the lineage that future-you will want.*

*Common categories of suggestions, for prompting your thinking:*

- *Validation work that the v1 plan called for but this run didn't do.*
- *Instrumentation that would have made this run's diagnostics faster.*
- *Cost / wall-clock reductions (caching, parallelism, budget guards).*
- *Methodological hardening (randomization treatments, deterministic verifiers, control prompts, reference candidates).*
- *Reproducibility hardening (capture more config in `run_manifest.json`, save raw response bodies, log provider request IDs).*
- *Bias-audit measurements that need a number for the writeup.*
- *Regression / safety nets (pre-commit smoke, budget guard, schema-version tag in outputs).*

*Pick from any category — the goal is a varied list that, taken together,
would meaningfully strengthen the next run.*

## Judge validation (if applicable)

*Only fill if `validate --mode score` was run for this experiment. Report:*

- *agreement % across the 60-row sample*
- *the number of disagreements*
- *patterns in the disagreements (judge over-strict on X, judge under-strict on Y, judge confused by Z)*
- *whether the agreement number changes any interpretation in the Headline Results section above*

## Reproducibility

*Exact commands run during this session, in order:*

```
uv run python main.py connectivity
uv run python main.py all --limit <N>
# <any ad-hoc diagnostic commands, with one-line context for each>
uv run python main.py validate --mode score
```

*Commit pointers for the state of the code at the time of this run:*

```
<hash>  <subject>
<hash>  <subject>
…
```

*If the code was patched mid-run, list each commit in the order it was applied.*

## Open questions / follow-ups

*Things this run surfaced that don't yet have a designed experiment. The bar is lower than "Next experiments" — these are leads, hunches, and "this looked weird" notes. Promote an item to a structured entry in "Next experiments" once you can write down its hypothesis, prior, and falsifying outcome.*

- *…*
- *…*
