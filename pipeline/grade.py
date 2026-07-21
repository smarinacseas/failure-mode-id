"""Grade candidate responses with each blind judge.

ONE judge call per (judge, prompt, candidate response). Never include the
model name; the judge is structurally blind. Every judge in `cfg.judges`
grades the SAME responses, so the dashboard can compare graders
apples-to-apples. JSON parsing is defensive with one retry; a truncated
response (judge spent its whole budget thinking) or a persistent parse
failure records all-FAIL with an error note; see pipeline/_judge_llm.py
for why the budget is generous and the call is streamed.

Two transports, selected by the frozen `judge_mode` param (default "batch"):

  batch:     the panel is split by judge client (Task 3). Anthropic-client
               judges: one Message Batch per judge; submit every missing
               grade cell, poll every POLL_INTERVAL_S with logged status,
               then collect results through the same locked append_jsonl
               path. `custom_id` is the item's stable id
               ("<candidate>__<prompt-id>", the prompt id alone would
               collide across candidates within a judge's batch). On resume,
               done_ids are re-derived from the output JSONL exactly as
               before and only the missing ids are submitted. Per-item batch
               errors and unparseable texts get the same one-retry-then-
               all-FAIL treatment as the sequential path. OpenRouter-client
               judges: OpenRouter has no batch API, so their cells grind on
               a small ThreadPoolExecutor (config.JUDGE_WORKERS) instead,
               submitted up front and awaited WHILE the Anthropic batches
               submit/poll/collect, all inside the same `run()` invocation
               with no barrier between the two transports (see `_run_pool`).
  sequential: the pre-batch path: one streamed call per cell, in order,
               for every judge regardless of client. `_run_sequential`
               already dispatches per spec via `call_json`, so this mode
               needs no pool.

Grading params (model, max_tokens, adaptive thinking, system prompt, user
message) and judge-blindness are identical across every transport; batch
custom_ids are request metadata and never enter the judge's context.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_JSONL, JUDGE_MAX_TOKENS, JUDGE_WORKERS, PROMPTS_DIR, anthropic
from pipeline._batch_guard import BatchStaleGuard
from pipeline._io import append_jsonl, read_jsonl, retry
from pipeline._judge_llm import call_json
from pipeline._select import select_prompts
from pipeline._json_extract import extract_json_array
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import JudgeSpec, RunConfig

JUDGE_SYSTEM = (PROMPTS_DIR / "judge.txt").read_text(encoding="utf-8")

POLL_INTERVAL_S = 60.0
CUSTOM_ID_SEP = "__"
_sleep = time.sleep          # module attr so tests can stub the poll wait


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


def _all_fail(n_criteria: int, reason: str) -> list[dict]:
    return [{"index": i, "verdict": "FAIL", "reason": reason}
            for i in range(1, n_criteria + 1)]


def _text_to_verdicts(raw: str, stop_reason: str | None,
                      n_criteria: int) -> tuple[list[dict] | None, str]:
    """Parse one judge response text. Returns (verdicts, "") on success or
    (None, error_reason) when the judge refused, the text is truncated, or it
    is unparseable; the single source of truth for both transports."""
    if stop_reason == "refusal":
        # Model-level refusal to grade (empty output). Sticky: 7/7 identical
        # refusals observed on E05 Fable × CIF-006, so callers must not spend
        # a second attempt on it.
        return None, ("judge_refusal: model declined to grade this cell "
                      "(stop_reason=refusal)")
    if stop_reason == "max_tokens":
        # Truncated before finishing the JSON; almost always thinking eating
        # the whole budget. Distinct from a genuine parse failure.
        return None, "judge_truncated: stop_reason=max_tokens (raise JUDGE_MAX_TOKENS)"
    try:
        return _normalize_verdicts(extract_json_array(raw), n_criteria), ""
    except Exception as e:  # noqa: BLE001
        return None, f"judge_parse_error: {type(e).__name__}: {e}"


def _grade_one(spec: JudgeSpec, prompt: str, response: str, criteria: list[str]) -> list[dict]:
    user_msg = _user_message(prompt, response, criteria)
    last_err: str | None = None
    for attempt in (1, 2):
        try:
            raw, stop_reason = call_json(spec, JUDGE_SYSTEM, user_msg,
                                         label=f"{spec.client}:{spec.key}")
        except Exception as e:  # noqa: BLE001
            last_err = f"judge_parse_error: {type(e).__name__}: {e}"
            continue
        verdicts, err = _text_to_verdicts(raw, stop_reason, len(criteria))
        if verdicts is not None:
            return verdicts
        last_err = err
        if err.startswith("judge_refusal"):
            break               # sticky: a second attempt is a wasted call
    return _all_fail(len(criteria), last_err or "judge_parse_error: unknown")


def _write_cell(cfg: RunConfig, mon, judge_key: str, key: str, rid: str,
                verdicts: list[dict]) -> None:
    """Append one grade record with the sequential path's exact accounting.
    `judge_key` is the panel member's path-safe key, not a provider model id."""
    mon.item_start(model=f"{key}@{judge_key}", prompt_id=rid)
    append_jsonl(cfg.grades_path(judge_key, key), {"id": rid, "verdicts": verdicts})
    reason0 = str(verdicts[0].get("reason", "")) if verdicts else ""
    if reason0.startswith("judge_parse_error") or reason0.startswith("judge_truncated"):
        mon.record_error(f"grade {judge_key}/{key} {rid}: {reason0.split(':', 1)[0]}")
    mon.item_done(model=f"{key}@{judge_key}", prompt_id=rid)


def _skip_empty(cfg: RunConfig, mon, judge_key: str, key: str, rid: str) -> None:
    # Defense-in-depth: an empty stored response has nothing to grade.
    # generate.py should never persist one UNFLAGGED, but if it does, skip it
    # (no grade record → prompt excluded at aggregate) rather than
    # spending judge calls to produce a misleading 0/N.
    mon.item_start(model=f"{key}@{judge_key}", prompt_id=rid)
    mon.record_error(f"grade {judge_key}/{key} {rid}: empty response, skipped (regenerate)")
    mon.item_done(model=f"{key}@{judge_key}", prompt_id=rid)


def _write_loop_failure(cfg: RunConfig, mon, judge_key: str, key: str, rid: str,
                        n_criteria: int, loop: dict) -> None:
    """Mechanical all-FAIL for a stored loop-failure record; no judge call.

    The model produced no answer: a runaway repetition loop consumed the
    budget (generate stores these flagged instead of retrying; ruling
    2026-07-09: loops count as failures). Grading emptiness would waste
    judge tokens to reach the same verdict, and skipping would silently
    EXCLUDE the prompt from the aggregate; the explicit 0/N keeps it in
    the results as the failure it is. Diagnose targets these cells like
    any other FAILs (the reason prefix is not a grading artifact)."""
    reason = (f"loop_failure: no answer, runaway repetition loop in "
              f"{loop.get('channel', 'reasoning')} "
              f"(period={loop.get('period')}, onset={loop.get('onset')})")
    _write_cell(cfg, mon, judge_key, key, rid, _all_fail(n_criteria, reason))


# --------------------------------------------------------------------------- #
# Sequential transport (pre-batch behavior, unchanged).
# --------------------------------------------------------------------------- #
def _run_sequential(cfg: RunConfig, mon, by_id: dict,
                    plan: list[tuple[JudgeSpec, str, list[dict]]]) -> None:
    for spec, key, todo in plan:
        for resp_rec in todo:
            rid = resp_rec["id"]
            rec = by_id[rid]
            if resp_rec.get("loop_failure"):
                _write_loop_failure(cfg, mon, spec.key, key, rid,
                                    len(rec["criteria"]), resp_rec["loop_failure"])
                continue
            if not (resp_rec.get("response") or "").strip():
                _skip_empty(cfg, mon, spec.key, key, rid)
                continue
            verdicts = _grade_one(spec, rec["prompt"], resp_rec["response"], rec["criteria"])
            _write_cell(cfg, mon, spec.key, key, rid, verdicts)


# --------------------------------------------------------------------------- #
# Batch transport (Anthropic Message Batches API).
# --------------------------------------------------------------------------- #
def _custom_id(key: str, rid: str) -> str:
    # Must be unique within one judge's batch: the same prompt id appears once
    # per candidate, so the cell's stable identity is candidate + prompt id.
    # (Anthropic custom_id charset is [a-zA-Z0-9_-]; keys/ids here are short
    # slug-like strings, and the separator must not appear inside the key.)
    if CUSTOM_ID_SEP in key:
        raise ValueError(
            f"candidate key {key!r} contains {CUSTOM_ID_SEP!r}; cannot build "
            "a unique batch custom_id"
        )
    return f"{key}{CUSTOM_ID_SEP}{rid}"


def _batch_request(spec: JudgeSpec, cid: str, user_msg: str) -> dict:
    # Params identical to the sequential call in pipeline/_judge_llm.call_json
    # (model, budget, adaptive thinking, system, single user message); the
    # transport differs, the treatment does not. `spec.model` is the provider
    # id; the batch is keyed downstream by spec.key.
    return {
        "custom_id": cid,
        "params": {
            "model": spec.model,
            "max_tokens": JUDGE_MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "system": JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": user_msg}],
        },
    }


def _run_batch(cfg: RunConfig, mon, by_id: dict,
               plan: list[tuple[JudgeSpec, str, list[dict]]]) -> None:
    # Batch transport is Anthropic-only; the caller passes the anthropic-client
    # slice of the plan. Cells are keyed by the panel member's KEY (path-safe,
    # batch-scope-safe); specs_by_key recovers the provider model id when the
    # request is built.
    specs_by_key: dict[str, JudgeSpec] = {spec.key: spec for spec, _key, _todo in plan}
    # Build the outstanding cells per judge from the SAME todo sets (done_ids
    # already filtered) the sequential path uses.
    pending: dict[str, dict[str, dict]] = {}      # judge_key -> custom_id -> cell
    for spec, key, todo in plan:
        for resp_rec in todo:
            rid = resp_rec["id"]
            rec = by_id[rid]
            if resp_rec.get("loop_failure"):
                _write_loop_failure(cfg, mon, spec.key, key, rid,
                                    len(rec["criteria"]), resp_rec["loop_failure"])
                continue
            if not (resp_rec.get("response") or "").strip():
                _skip_empty(cfg, mon, spec.key, key, rid)
                continue
            pending.setdefault(spec.key, {})[_custom_id(key, rid)] = {
                "key": key,
                "rid": rid,
                "user_msg": _user_message(rec["prompt"], resp_rec["response"], rec["criteria"]),
                "n_criteria": len(rec["criteria"]),
                "last_err": "",
            }

    submitted = collected = errored = 0

    def _counts() -> None:
        mon.set_batch_counts(
            submitted=submitted,
            pending=sum(len(cells) for cells in pending.values()),
            collected=collected,
            errored=errored,
        )

    # Two rounds mirror the sequential path's two grade attempts: a cell whose
    # result is truncated, unparseable, or batch-errored is resubmitted once,
    # then falls through to the all-FAIL record below.
    for attempt in (1, 2):
        live = {judge_key: cells for judge_key, cells in pending.items() if cells}
        if not live:
            break

        open_batches: dict[str, str] = {}
        for judge_key, cells in live.items():
            spec = specs_by_key[judge_key]
            requests = [_batch_request(spec, cid, cell["user_msg"])
                        for cid, cell in cells.items()]
            batch = retry(
                lambda requests=requests:
                    anthropic.messages.batches.create(requests=requests),
                label=f"anthropic:batches:create:{judge_key}",
            )
            open_batches[judge_key] = batch.id
            submitted += len(requests)
            mon.note(f"grade batch {judge_key}: submitted {len(requests)} request(s) "
                     f"(attempt {attempt}, batch {batch.id})")
        guards = {judge_key: BatchStaleGuard(anthropic, batch_id,
                                             f"grade batch {judge_key}", note=mon.note)
                  for judge_key, batch_id in open_batches.items()}
        _counts()

        # Poll every POLL_INTERVAL_S with logged status; collect each judge's
        # batch as soon as it ends (no barrier on the other judge).
        while open_batches:
            for judge_key, batch_id in list(open_batches.items()):
                b = retry(
                    lambda batch_id=batch_id:
                        anthropic.messages.batches.retrieve(batch_id),
                    label=f"anthropic:batches:retrieve:{judge_key}",
                )
                rc = getattr(b, "request_counts", None)
                status = getattr(b, "processing_status", "unknown")
                detail = ""
                if rc is not None:
                    detail = (f", processing={getattr(rc, 'processing', '?')}, "
                              f"succeeded={getattr(rc, 'succeeded', '?')}, "
                              f"errored={getattr(rc, 'errored', '?')}")
                mon.note(f"grade batch {judge_key}: {status}{detail}")
                guards[judge_key].observe(b)
                if status != "ended":
                    continue
                del open_batches[judge_key]

                results = retry(
                    lambda batch_id=batch_id:
                        list(anthropic.messages.batches.results(batch_id)),
                    label=f"anthropic:batches:results:{judge_key}",
                )
                cells = pending[judge_key]
                for res in results:
                    cell = cells.get(res.custom_id)
                    if cell is None:
                        continue                     # not one of ours (stale id)
                    rtype = res.result.type
                    if rtype == "succeeded":
                        msg = res.result.message
                        raw = "".join(blk.text for blk in msg.content
                                      if getattr(blk, "type", None) == "text")
                        verdicts, err = _text_to_verdicts(
                            raw, getattr(msg, "stop_reason", None), cell["n_criteria"])
                        if verdicts is not None:
                            _write_cell(cfg, mon, judge_key, cell["key"], cell["rid"], verdicts)
                            del cells[res.custom_id]
                            collected += 1
                        elif err.startswith("judge_refusal"):
                            # Sticky (see _text_to_verdicts); write the
                            # terminal record now; a resubmission round would
                            # only refuse again.
                            _write_cell(cfg, mon, judge_key, cell["key"], cell["rid"],
                                        _all_fail(cell["n_criteria"], err))
                            del cells[res.custom_id]
                            collected += 1
                        else:
                            cell["last_err"] = err   # resubmit next round
                    else:
                        # errored / canceled / expired: the batch analog of the
                        # sequential call raising; one resubmission, then the
                        # all-FAIL fallback (continue-on-error, as before).
                        errored += 1
                        err_obj = getattr(res.result, "error", None)
                        etype = getattr(err_obj, "type", None) or ""
                        cell["last_err"] = (
                            f"judge_parse_error: batch result {rtype}"
                            + (f" ({etype})" if etype else "")
                        )
                _counts()
            if open_batches:
                _sleep(POLL_INTERVAL_S)

    # Cells that failed both rounds: same terminal record the sequential path
    # writes after its second attempt: all-FAIL verdicts carrying the reason,
    # recorded as an error, item marked done. Never silently dropped.
    for judge_key, cells in pending.items():
        for cid, cell in list(cells.items()):
            reason = cell["last_err"] or "judge_parse_error: batch result missing after 2 attempts"
            _write_cell(cfg, mon, judge_key, cell["key"], cell["rid"],
                        _all_fail(cell["n_criteria"], reason))
            del cells[cid]
    _counts()


# --------------------------------------------------------------------------- #
# OpenRouter transport: streamed calls on a small worker pool. No batch API
# exists on OpenRouter; concurrency here is transport, not treatment
# (JUDGE_WORKERS is a config constant, recorded in meta.config, never frozen).
# --------------------------------------------------------------------------- #
def _pool_item(cfg: RunConfig, mon, spec: JudgeSpec, key: str,
               resp_rec: dict, by_id: dict) -> None:
    """One (judge, candidate, prompt) grade cell, run on a pool worker.
    Mirrors _run_sequential's per-cell body exactly (mechanical loop-failure
    and empty-response short-circuits included) so the same cell graded via
    either transport lands an identical record."""
    rid = resp_rec["id"]
    rec = by_id[rid]
    if resp_rec.get("loop_failure"):
        _write_loop_failure(cfg, mon, spec.key, key, rid,
                            len(rec["criteria"]), resp_rec["loop_failure"])
        return
    if not (resp_rec.get("response") or "").strip():
        _skip_empty(cfg, mon, spec.key, key, rid)
        return
    verdicts = _grade_one(spec, rec["prompt"], resp_rec["response"], rec["criteria"])
    _write_cell(cfg, mon, spec.key, key, rid, verdicts)


def _run_pool(cfg: RunConfig, mon, by_id: dict,
              plan: list[tuple[JudgeSpec, str, list[dict]]],
              pool: ThreadPoolExecutor) -> list:
    """Submit every OpenRouter grade cell to `pool`; returns the futures
    (caller awaits; see run()). Cells share only the locked append_jsonl
    path and the monitor's own lock; no other mutable state crosses workers."""
    return [pool.submit(_pool_item, cfg, mon, spec, key, resp_rec, by_id)
            for spec, key, todo in plan for resp_rec in todo]


def run(cfg: RunConfig, monitor: RunMonitor | None = None) -> None:
    records = select_prompts(read_jsonl(DATA_JSONL), cfg.limit, cfg.sample_seed)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")
    by_id = {r["id"]: r for r in records}

    with stage_ctx(monitor, "grade", len(records)) as mon:
        # total spans every (judge, candidate, prompt) cell.
        total = len(records) * len(cfg.candidates) * len(cfg.judges)
        plan: list[tuple[JudgeSpec, str, list[dict]]] = []  # (spec, candidate, todo responses)
        already = 0
        for spec in cfg.judges:
            for key in cfg.candidates:
                responses = read_jsonl(cfg.responses_path(key))
                done_ids = {r["id"] for r in read_jsonl(cfg.grades_path(spec.key, key))}
                todo = [r for r in responses if r["id"] in by_id and r["id"] not in done_ids]
                already += len(done_ids)
                if not responses:
                    mon.note(f"grade {spec.key}/{key}: no responses yet, skipping.")
                plan.append((spec, key, todo))

        mon.start_stage("grade", total=total, already_done=already)
        anth_plan = [p for p in plan if p[0].client == "anthropic"]
        or_plan = [p for p in plan if p[0].client == "openrouter"]
        if cfg.judge_mode == "sequential":
            # judge_mode=="sequential" intentionally runs the WHOLE plan
            # sequentially, OpenRouter judges included; one streamed call
            # per cell is exactly what that mode means.
            _run_sequential(cfg, mon, by_id, plan)
        else:
            # Mixed transports, one invocation: OpenRouter cells grind on the
            # pool WHILE the Anthropic batches submit/poll/collect.
            or_futures: list = []
            with ThreadPoolExecutor(max_workers=max(1, int(JUDGE_WORKERS)),
                                    thread_name_prefix="grade-pool") as pool:
                try:
                    or_futures = _run_pool(cfg, mon, by_id, or_plan, pool)
                    if anth_plan:
                        _run_batch(cfg, mon, by_id, anth_plan)
                    for f in as_completed(or_futures):
                        f.result()
                except BaseException:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
        mon.end_stage()
