"""Grade candidate responses with each blind judge.

ONE judge call per (judge, prompt, candidate response). Never include the
model name — the judge is structurally blind. Every judge in `cfg.judges`
grades the SAME responses, so the dashboard can compare graders
apples-to-apples. JSON parsing is defensive with one retry; a truncated
response (judge spent its whole budget thinking) or a persistent parse
failure records all-FAIL with an error note — see pipeline/_judge_llm.py
for why the budget is generous and the call is streamed.
"""

from __future__ import annotations

from config import DATA_JSONL, PROMPTS_DIR
from pipeline._io import append_jsonl, read_jsonl
from pipeline._judge_llm import call_json
from pipeline._select import select_prompts
from pipeline._json_extract import extract_json_array
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import RunConfig

JUDGE_SYSTEM = (PROMPTS_DIR / "judge.txt").read_text(encoding="utf-8")


def _user_message(prompt: str, response: str, criteria: list[str]) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1))
    return (
        "TASK PROMPT:\n"
        f"{prompt}\n\n"
        "MODEL RESPONSE:\n"
        f"{response}\n\n"
        "CRITERIA:\n"
        f"{numbered}"
    )


def _normalize_verdicts(parsed: list, n_criteria: int) -> list[dict]:
    """Coerce the judge's array into exactly n_criteria {index,verdict,reason} dicts."""
    by_index: dict[int, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        verdict = str(item.get("verdict", "")).upper().strip()
        if verdict not in {"PASS", "FAIL"}:
            verdict = "FAIL"
        reason = str(item.get("reason", "")).strip()
        by_index[idx] = {"index": idx, "verdict": verdict, "reason": reason}

    out: list[dict] = []
    for i in range(1, n_criteria + 1):
        out.append(by_index.get(i, {"index": i, "verdict": "FAIL", "reason": "missing_in_judge_output"}))
    return out


def _grade_one(judge: str, prompt: str, response: str, criteria: list[str]) -> list[dict]:
    user_msg = _user_message(prompt, response, criteria)
    last_err: str | None = None
    for attempt in (1, 2):
        try:
            raw, stop_reason = call_json(judge, JUDGE_SYSTEM, user_msg, label=f"anthropic:{judge}")
            if stop_reason == "max_tokens":
                # Truncated before finishing the JSON — almost always thinking eating
                # the whole budget. Distinct from a genuine parse failure.
                last_err = "judge_truncated: stop_reason=max_tokens (raise JUDGE_MAX_TOKENS)"
                if attempt == 2:
                    break
                continue
            parsed = extract_json_array(raw)
            return _normalize_verdicts(parsed, len(criteria))
        except Exception as e:  # noqa: BLE001
            last_err = f"judge_parse_error: {type(e).__name__}: {e}"
            if attempt == 2:
                break
    return [
        {"index": i, "verdict": "FAIL", "reason": last_err}
        for i in range(1, len(criteria) + 1)
    ]


def run(cfg: RunConfig, monitor: RunMonitor | None = None) -> None:
    records = select_prompts(read_jsonl(DATA_JSONL), cfg.limit, cfg.sample_seed)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")
    by_id = {r["id"]: r for r in records}

    with stage_ctx(monitor, "grade", len(records)) as mon:
        # total spans every (judge, candidate, prompt) cell.
        total = len(records) * len(cfg.candidates) * len(cfg.judges)
        plan: list[tuple[str, str, list[dict]]] = []  # (judge, candidate, todo responses)
        already = 0
        for judge in cfg.judges:
            for key in cfg.candidates:
                responses = read_jsonl(cfg.responses_path(key))
                done_ids = {r["id"] for r in read_jsonl(cfg.grades_path(judge, key))}
                todo = [r for r in responses if r["id"] in by_id and r["id"] not in done_ids]
                already += len(done_ids)
                if not responses:
                    mon.note(f"grade {judge}/{key}: no responses yet — skipping.")
                plan.append((judge, key, todo))

        mon.start_stage("grade", total=total, already_done=already)
        for judge, key, todo in plan:
            out_path = cfg.grades_path(judge, key)
            for resp_rec in todo:
                rid = resp_rec["id"]
                rec = by_id[rid]
                mon.item_start(model=f"{key}@{judge}", prompt_id=rid)
                # Defense-in-depth: an empty stored response has nothing to grade.
                # generate.py should never persist one, but if it does, skip it
                # (no grade record → prompt excluded at aggregate) rather than
                # spending judge calls to produce a misleading 0/N.
                if not (resp_rec.get("response") or "").strip():
                    mon.record_error(f"grade {judge}/{key} {rid}: empty response — skipped (regenerate)")
                    mon.item_done()
                    continue
                verdicts = _grade_one(judge, rec["prompt"], resp_rec["response"], rec["criteria"])
                append_jsonl(out_path, {"id": rid, "verdicts": verdicts})
                reason0 = str(verdicts[0].get("reason", "")) if verdicts else ""
                if reason0.startswith("judge_parse_error") or reason0.startswith("judge_truncated"):
                    mon.record_error(f"grade {judge}/{key} {rid}: {reason0.split(':', 1)[0]}")
                mon.item_done()
        mon.end_stage()
