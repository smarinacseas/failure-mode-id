# ComplexConstraints v1 — Master Execution Runbook

The complete, ordered build for v1: real eval of open models against ComplexConstraints → failure-mode analysis → bias audit → public deliverable. Integrates the model layer (OpenRouter candidates, Opus judge), maps to your existing dashboard, and specifies the two focus deliverables your prior docs didn't: the **experimental-weakness audit** and the **analysis writeup**.

Detailed grader/classifier/validation/aggregate **code lives in `SPA-real-run-pipeline-guide.md`** — this runbook references those steps and gives in full only what changed (OpenRouter generation) and what's new (bias audit, analysis, dashboard mapping, ship).

---

## Definition of done for v1

Public repo + live dashboard + blog, where: real candidate models are evaluated against ComplexConstraints with a validated judge; failure modes are visible at aggregate AND prompt-detail level; a bias/limitations audit is explicit; and an analysis closes the loop to a data/rubric spec. No training (that's v2).

## Asset inventory (where you are)

- **Built:** dashboard with loss-analysis framing, criterion-level pass rate, prompt-level full-pass rate, constraint-satisfaction gap, pass rates by instruction_type/prompt_style/use_case, model comparison, filtered-view navigation.
- **Decided:** candidates = Qwen3.5 size ladder + one cross-family model, via OpenRouter; judge + classifier = Opus 4.8 via Anthropic; v1 = eval only.
- **To add to dashboard:** prompt-detail drill-down (criteria checklist), verifiability/reward-hack flags, limitations panel.

---

## PHASE 1 — Project + OpenRouter setup

```bash
mkdir -p data responses grades && cd <repo>
uv add openai anthropic pandas openpyxl python-dotenv
```

`.env` (two keys — candidates and judge use different providers):

```
OPENROUTER_API_KEY=sk-or-...
ANTHROPIC_API_KEY=sk-ant-...
```

Central config — `config.py`:

```python
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

# Candidates via OpenRouter (OpenAI-compatible). VERIFY exact IDs on openrouter.ai/models.
router = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
CANDIDATES = {
    "qwen-9b":   "qwen/qwen3.5-9b",
    "qwen-35b":  "qwen/qwen3.5-35b-a3b",
    "qwen-397b": "qwen/qwen3.5-397b-a17b",
    "deepseek":  "deepseek/deepseek-v4",     # cross-family robustness check (optional 4th)
}
# Judge + classifier via Anthropic (NON-candidate family → no self-preference bias)
anthropic = Anthropic()
JUDGE = "claude-opus-4-8"
```

**UNVERIFIED — confirm every model ID on openrouter.ai/models before running;** the open lineup ships monthly and these strings drift.

---

## PHASE 2 — Load benchmark

`SPA-real-run-pipeline-guide.md` Step 1, unchanged → `data/complexconstraints.jsonl`.

---

## PHASE 3 — Generate candidate responses (OpenRouter — the changed code)

Greedy decoding (`temperature=0`) for reproducibility — a re-run gives the same outputs, so your findings are stable. (Trade-off noted in the bias audit: greedy ≠ typical sampled behavior.)

```python
# 3_generate.py
import json
from config import router, CANDIDATES

records = [json.loads(l) for l in open("data/complexconstraints.jsonl")]
for key, model in CANDIDATES.items():
    with open(f"responses/{key}.jsonl", "w") as f:
        for rec in records:
            resp = router.chat.completions.create(
                model=model, temperature=0, max_tokens=4000,
                messages=[{"role": "user", "content": rec["prompt"]}],
            )
            f.write(json.dumps({"id": rec["id"], "response": resp.choices[0].message.content}) + "\n")
            print(f"gen {rec['id']} · {key} ✓")
```

Loop `CANDIDATES` keys everywhere downstream instead of the old `["opus","haiku"]`.

---

## PHASE 4 — Grade (Opus judge, blind, batched)

`SPA-real-run-pipeline-guide.md` Step 3, unchanged except: judge client is `anthropic`, model `JUDGE`; iterate over all `CANDIDATES` keys. The judge is blind (never sees the candidate name) and non-Anthropic-candidate, so self-preference bias is structurally removed. → `grades/{key}.jsonl`.

## PHASE 5 — Rubric-critique classification

`SPA-real-run-pipeline-guide.md` Step 4, unchanged → `criteria_tags.jsonl` (verifiability + gameable + reward_hack per criterion). **Hand-correct ~20 — this layer is where your judgment is the product.**

## PHASE 6 — Validate the judge (credibility keystone)

`SPA-real-run-pipeline-guide.md` Step 5. Hand-grade ~60 sampled criteria blind, compute agreement, log disagreements. Carry the agreement number into the bias audit (Phase 9) and the dashboard limitations panel. **Do not skip.**

## PHASE 7 — Aggregate to results.json

`SPA-real-run-pipeline-guide.md` Step 6, iterating all `CANDIDATES` keys. The schema already carries everything your dashboard needs (Phase 8 mapping). → `results.json`.

---

## PHASE 8 — Wire results.json into the dashboard

Map each existing feature to a field, then add the three new views.

| Dashboard feature (you built)                            | results.json field                                                               |
| -------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Criterion-level pass rate                                | `summary.criterion_pass_rate[model]`                                             |
| Prompt-level full-pass rate                              | `summary.full_prompt_pass_rate[model]`                                           |
| Constraint-satisfaction gap                              | `criterion_pass_rate − full_prompt_pass_rate` (compute in UI; feature the delta) |
| Pass rates by instruction_type / prompt_style / use_case | `summary.by_instruction_type` / `by_prompt_style` / `by_use_case`                |
| Model comparison                                         | per-model values across all of the above                                         |
| Filtered-view navigation                                 | filter `prompts[]` on `instruction_type` / `prompt_style` / `use_case`           |

New views to add (your focus items):

| New view                                     | results.json field                                                                                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Prompt-detail drill-down** (failure modes) | `prompts[].prompt_text`, `prompts[].responses[model]`, `prompts[].criteria[]` with `results[model].pass` + `.reason` rendered as a PASS/FAIL checklist |
| **Verifiability filter**                     | `prompts[].criteria[].verifiability` ("auto" / "judge")                                                                                                |
| **Reward-hack flags**                        | `prompts[].criteria[].gameable` (badge) + `.reward_hack` (hover)                                                                                       |

**Wiring (recommended path):** in Claude Design, upload `results.json` and instruct it to replace mock data and bind to these fields, add the prompt-detail/verifiability/reward-hack views, and keep the design. It built the schema; it can remap.

---

## PHASE 9 — Experimental-weakness & bias audit _(first-class deliverable)_

A dedicated methodology/limitations artifact (a blog section AND a dashboard panel). This is the single strongest signal of the "calibrated confidence" trait — it's you catching your own study's flaws before AfterQuery does. Document each:

1. **Judge self-preference — controlled.** Candidates are all non-Anthropic; the Opus judge has no family stake. State this as a deliberate design choice, not luck.
2. **Residual judge bias — measured.** Verbosity, position, and format biases remain; your judge-validation agreement (Phase 6) quantifies net judge error. Report the number and the disagreement patterns.
3. **Judge ≠ ground truth, weighted by verifiability.** "auto"-verifiable criteria (deterministic) carry higher confidence than "judge"-dependent ones. Report pass rates split by verifiability so readers can discount the soft half.
4. **Benchmark-internal weaknesses.** The gameable/ambiguous criteria your classifier flagged — name them; they cap how much any score can mean.
5. **Decoding choice.** Greedy (temp=0) for reproducibility; doesn't capture sampled-behavior variance. (Optional rigor: re-run 10 prompts at temp=0.7 ×3 to report variance.)
6. **Domain skew.** ComplexConstraints is 34 logistics / 22 data-math / 0 sales — findings generalize to constraint-heavy planning tasks, not all instruction-following.
7. **Model-selection scope.** One family for the size gradient + one cross-family check; not an exhaustive survey.
8. **Sample size.** 75 prompts / ~1,560 criteria — directional, not definitive; report confidence qualitatively.

---

## PHASE 10 — Analysis & writeup _(first-class deliverable → the blog)_

Process-forward narrative. Order:

1. **Headline — the constraint-satisfaction gap.** High criterion-pass, low full-prompt-pass: models satisfy most constraints but rarely all simultaneously. Lead with this.
2. **Failure-mode analysis.** By instruction_type (expect Negative + Implicit worst), prompt_style (expect Rambling degrades), use_case. For each, a hypothesized _why_, grounded in 1–2 real transcript examples (paraphrased — see IP).
3. **The scaling story.** Does the Qwen 9B→35B→397B ladder close the hard modes, or do Negative/Implicit stay stubborn at frontier scale? Either answer is interesting.
4. **Cross-family robustness.** Does DeepSeek confirm the pattern? If yes, the finding isn't a Qwen artifact.
5. **Rubric critique.** Verifiability split + the most gameable criteria with their reward-hack vectors. The rarest, most AfterQuery-native content.
6. **The spec (close the loop).** Top 2–3 failure modes → the data/rubric that would fix them → which AfterQuery product delivers it (Rubric-and-Verifier-based RL; Custom Evals).
7. **Limitations.** Phase 9, honestly.

---

## PHASE 11 — Ship checklist

- [ ] `README.md`: thesis + proof-mapping table (artifact section → SPA/SPL JD responsibility).
- [ ] Repo ships _your_ grader, classifier, analysis, spec — **not Surge's 75 prompts/criteria** (IP). Illustrate with paraphrased/self-authored examples.
- [ ] Dashboard hosted publicly (static export or hosted artifact), linked from the blog.
- [ ] Blog published; leads with the constraint-satisfaction gap.
- [ ] Limitations panel live in the dashboard.
- [ ] Judge-validation agreement number stated prominently.

---

## Sequencing, cost, cut-lines

- **Smoke test first:** run Phases 2→8 on **3 prompts** end-to-end before the full 75. Catch schema/judge-prompt/OpenRouter-ID bugs cheaply.
- **Order:** 1→2→3 (generate) → 4 (grade) → 5 (classify) → 6 (validate) → 7 (aggregate) → 8 (wire) → 9–10 (audit + writeup) → 11 (ship).
- **Cost:** ~225–300 generation calls (75 × 3–4 candidates) + ~225–300 judge + ~75 classify, all cheap open/API rates → a few dollars, under an hour wall-clock.
- **Cut-lines:** drop to 3 candidates (the Qwen ladder; defer DeepSeek), then to ~40 prompts keeping all four instruction_types. **Never cut Phase 6 (validation), Phase 9 (audit), or the Phase 5 hand-check** — they are the credibility, not the volume.

## v1 done =

real run shipped · failure modes visible at aggregate + prompt level · judge validated and reported · bias audit explicit · analysis closes to a spec · public repo + dashboard + blog live.
