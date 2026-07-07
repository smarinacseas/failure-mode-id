"""Diagnose WHY each failed criterion failed (root-cause labels).

One BLINDED batch call per (candidate, prompt) cell with >=1 real FAIL under
the canonical judge's grades: the analyst sees the prompt, all criteria, the
unmet indices, the response, and the reasoning trace — never judge reasons,
candidate identity, or the second judge's verdicts (spec §4). Output appended
to runs/<slug>/diagnosis/<candidate>.jsonl. Resumable. Batch-only transport.
"""

from __future__ import annotations

import time

from config import (
    DATA_JSONL,
    DIAGNOSE_JUDGE,
    DIAGNOSE_MAX_FIELD_CHARS,
    DIAGNOSE_MAX_TOKENS,
    anthropic,
)
from pipeline import _taxonomy
from pipeline._io import append_jsonl, read_jsonl, retry
from pipeline._json_extract import extract_json_array
from pipeline._select import select_prompts
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import RunConfig

POLL_INTERVAL_S = 60.0
CUSTOM_ID_SEP = "__"
_sleep = time.sleep          # module attr so tests can stub the poll wait

# Grading-artifact reasons: these FAILs carry no information about the
# candidate, so they are never diagnose targets (targeting reads reasons;
# the analyst payload never does — spec §4 rule 1).
_ARTIFACT_PREFIXES = ("judge_refusal", "judge_parse_error", "judge_truncated",
                      "missing_in_judge_output")


def _failed_cells(cfg: RunConfig, records: list[dict]) -> list[dict]:
    """(candidate, prompt) cells with >=1 real FAIL under cfg.judge's grades,
    minus cells already present in the diagnosis output (resume)."""
    by_id = {r["id"]: r for r in records}
    cells: list[dict] = []
    for key in cfg.candidates:
        grades = {g["id"]: g["verdicts"] for g in read_jsonl(cfg.grades_path(cfg.judge, key))}
        responses = {r["id"]: r for r in read_jsonl(cfg.responses_path(key))}
        done = {r["id"] for r in read_jsonl(cfg.diagnosis_path(key))}
        for rid, verdicts in grades.items():
            if rid in done or rid not in by_id or rid not in responses:
                continue
            failed = [v["index"] for v in verdicts
                      if v["verdict"] == "FAIL"
                      and not str(v.get("reason", "")).startswith(_ARTIFACT_PREFIXES)]
            if not failed:
                continue
            rec = by_id[rid]
            cells.append({
                "key": key,
                "rid": rid,
                "prompt": rec["prompt"],
                "criteria": rec["criteria"],
                "failed_indices": sorted(failed),
                "response": responses[rid].get("response") or "",
                "reasoning": responses[rid].get("reasoning") or "",
            })
    return cells


def _clip(text: str) -> tuple[str, bool]:
    """Head+tail clip for degenerate-length fields (spec §4): keeps the start
    (where instructions get restated) and the end (where answers conclude)."""
    if len(text) <= DIAGNOSE_MAX_FIELD_CHARS:
        return text, False
    head = DIAGNOSE_MAX_FIELD_CHARS * 2 // 3
    tail = DIAGNOSE_MAX_FIELD_CHARS - head
    return (text[:head]
            + f"\n\n[… TRUNCATED: {len(text) - head - tail} chars removed …]\n\n"
            + text[-tail:]), True


def _user_message(cell: dict) -> tuple[str, str]:
    """Blinded analyst payload (spec §4): prompt + ALL criteria (needed for
    conflict-tradeoff diagnosis) + unmet indices + response + trace. Never
    judge reasons, candidate identity, or verdict words."""
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(cell["criteria"], start=1))
    unmet = ", ".join(str(i) for i in cell["failed_indices"])
    response, resp_clipped = _clip(cell["response"])
    reasoning = cell["reasoning"]
    trace_clipped = False
    if reasoning:
        reasoning, trace_clipped = _clip(reasoning)
        trace_status = "truncated" if (trace_clipped or resp_clipped) else "present"
        trace_block = f"MODEL REASONING TRACE:\n{reasoning}\n\n"
    else:
        # Absence beats truncation in the flag: no trace stays "absent" even
        # when the response itself was clipped.
        trace_status = "absent"
        trace_block = "MODEL REASONING TRACE: (none recorded)\n\n"
    payload = (
        "TASK PROMPT GIVEN TO A LANGUAGE MODEL:\n"
        f"{cell['prompt']}\n\n"
        "ALL CRITERIA THE RESPONSE WAS EVALUATED AGAINST:\n"
        f"{numbered}\n\n"
        f"CRITERIA JUDGED UNMET — diagnose exactly these indices: {unmet}\n\n"
        f"{trace_block}"
        "MODEL RESPONSE:\n"
        f"{response}"
    )
    return payload, trace_status


_CONFIDENCES = {"high", "medium", "low"}


def _text_to_diagnoses(raw: str, stop_reason: str | None,
                       failed_indices: list[int],
                       trace_present: bool) -> tuple[list[dict] | None, str]:
    """Parse one analyst response. Mirrors grade._text_to_verdicts: refusal is
    sticky-terminal, max_tokens and malformed/incomplete output are retriable
    (one resubmission round), everything is validated against the exact
    category set this cell was offered."""
    if stop_reason == "refusal":
        return None, ("diagnose_refusal: model declined to analyze this cell "
                      "(stop_reason=refusal)")
    if stop_reason == "max_tokens":
        return None, "diagnose_truncated: stop_reason=max_tokens (raise DIAGNOSE_MAX_TOKENS)"
    allowed = _taxonomy.allowed_keys(trace_present)
    try:
        parsed = extract_json_array(raw)
        by_index: dict[int, dict] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = int(item.get("index"))
            root = str(item.get("root_cause", "")).strip()
            if root not in allowed:
                raise ValueError(f"root_cause {root!r} not in offered taxonomy")
            secondary = item.get("secondary")
            secondary = str(secondary).strip() if secondary else None
            if secondary is not None and secondary not in allowed:
                raise ValueError(f"secondary {secondary!r} not in offered taxonomy")
            confidence = str(item.get("confidence", "")).lower().strip()
            if confidence not in _CONFIDENCES:
                confidence = "low"
            if idx in by_index:
                raise ValueError(f"duplicate diagnosis for index {idx}")
            by_index[idx] = {
                "index": idx,
                "evidence": str(item.get("evidence", "")).strip(),
                "root_cause": root,
                "secondary": secondary,
                "confidence": confidence,
                "rationale": str(item.get("rationale", "")).strip(),
            }
        missing = [i for i in failed_indices if i not in by_index]
        if missing:
            raise ValueError(f"missing diagnoses for unmet indices {missing}")
        return [by_index[i] for i in failed_indices], ""
    except Exception as e:  # noqa: BLE001
        return None, f"diagnose_parse_error: {type(e).__name__}: {e}"


def _error_rows(failed_indices: list[int], reason: str) -> list[dict]:
    """Terminal rows after both rounds fail: honest 'other' at low confidence,
    with the machine-readable reason preserved in rationale (the aggregate
    fold-in and the dashboard can surface it)."""
    return [{"index": i, "evidence": "", "root_cause": "other",
             "secondary": None, "confidence": "low", "rationale": reason}
            for i in failed_indices]


def _batch_request(cid: str, system: str, user_msg: str) -> dict:
    return {
        "custom_id": cid,
        "params": {
            "model": DIAGNOSE_JUDGE,
            "max_tokens": DIAGNOSE_MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "system": system,
            "messages": [{"role": "user", "content": user_msg}],
        },
    }


def _write_cell(cfg: RunConfig, mon, cell: dict, trace_status: str,
                rows: list[dict], error: str = "") -> None:
    mon.item_start(model=cell["key"], prompt_id=cell["rid"])
    append_jsonl(cfg.diagnosis_path(cell["key"]),
                 {"id": cell["rid"], "trace_status": trace_status, "diagnoses": rows})
    if error:
        mon.record_error(f"diagnose {cell['key']} {cell['rid']}: {error.split(':', 1)[0]}")
    mon.item_done(model=cell["key"], prompt_id=cell["rid"])


def run(cfg: RunConfig, monitor: RunMonitor | None = None) -> None:
    records = select_prompts(read_jsonl(DATA_JSONL), cfg.limit, cfg.sample_seed)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")

    with stage_ctx(monitor, "diagnose", len(records)) as mon:
        cells = _failed_cells(cfg, records)
        already = sum(len(read_jsonl(cfg.diagnosis_path(key))) for key in cfg.candidates)
        mon.start_stage("diagnose", total=len(cells) + already, already_done=already)
        if not cells:
            mon.note("diagnose: no undiagnosed failed cells — nothing to do.")
            mon.end_stage()
            return

        pending: dict[str, dict] = {}
        for cell in cells:
            if CUSTOM_ID_SEP in cell["key"]:
                raise ValueError(f"candidate key {cell['key']!r} contains "
                                 f"{CUSTOM_ID_SEP!r}; cannot build a unique custom_id")
            user_msg, trace_status = _user_message(cell)
            pending[f"{cell['key']}{CUSTOM_ID_SEP}{cell['rid']}"] = {
                **cell, "user_msg": user_msg, "trace_status": trace_status,
                "last_err": "",
            }

        # Two rounds mirror grade's batch transport: truncated/unparseable/
        # batch-errored cells resubmit once; refusals are sticky-terminal.
        for attempt in (1, 2):
            if not pending:
                break
            requests = [
                _batch_request(cid, _taxonomy.diagnose_system(c["trace_status"] != "absent"),
                               c["user_msg"])
                for cid, c in pending.items()
            ]
            batch = retry(
                lambda requests=requests:
                    anthropic.messages.batches.create(requests=requests),
                label="anthropic:batches:create:diagnose",
            )
            mon.note(f"diagnose batch: submitted {len(requests)} cell(s) "
                     f"(attempt {attempt}, batch {batch.id})")

            while True:
                b = retry(lambda: anthropic.messages.batches.retrieve(batch.id),
                          label="anthropic:batches:retrieve:diagnose")
                status = getattr(b, "processing_status", "unknown")
                mon.note(f"diagnose batch: {status}")
                if status == "ended":
                    break
                _sleep(POLL_INTERVAL_S)

            results = retry(lambda: list(anthropic.messages.batches.results(batch.id)),
                            label="anthropic:batches:results:diagnose")
            for res in results:
                cell = pending.get(res.custom_id)
                if cell is None:
                    continue
                if res.result.type == "succeeded":
                    msg = res.result.message
                    raw = "".join(blk.text for blk in msg.content
                                  if getattr(blk, "type", None) == "text")
                    rows, err = _text_to_diagnoses(
                        raw, getattr(msg, "stop_reason", None),
                        cell["failed_indices"], cell["trace_status"] != "absent")
                    if rows is not None:
                        _write_cell(cfg, mon, cell, cell["trace_status"], rows)
                        del pending[res.custom_id]
                    elif err.startswith("diagnose_refusal"):
                        _write_cell(cfg, mon, cell, cell["trace_status"],
                                    _error_rows(cell["failed_indices"], err), error=err)
                        del pending[res.custom_id]
                    else:
                        cell["last_err"] = err          # resubmit next round
                else:
                    err_obj = getattr(res.result, "error", None)
                    etype = getattr(err_obj, "type", None) or ""
                    cell["last_err"] = (f"diagnose_parse_error: batch result "
                                        f"{res.result.type}"
                                        + (f" ({etype})" if etype else ""))

        for cid, cell in list(pending.items()):
            reason = cell["last_err"] or "diagnose_parse_error: batch result missing after 2 attempts"
            _write_cell(cfg, mon, cell, cell["trace_status"],
                        _error_rows(cell["failed_indices"], reason), error=reason)
            del pending[cid]
        mon.end_stage()
