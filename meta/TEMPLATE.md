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

*Either list the artifact files and what each contains (if the schema changed since the last run), or write "unchanged — see <prior report>".*

| path | content |
| --- | --- |
| *outputs/results.json* | *…* |

## Lessons / notes for the next experimenter

*The single highest-value section for a future reader. What would you tell yourself if you were about to start this same run? Include both technical gotchas and process observations. Number them so they can be cited.*

1. *…*
2. *…*

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

*Anything this run surfaced that the next run should answer. Be specific — "investigate scaling" is too vague; "verify whether CIF-001 inverse scaling holds on the other 24 Negative · Context prompts" is what to write.*

- *…*
- *…*
