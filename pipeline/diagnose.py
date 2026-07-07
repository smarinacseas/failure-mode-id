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
        trace_status = "absent"
        trace_block = "MODEL REASONING TRACE: (none recorded)\n\n"
    if not cell["reasoning"] and resp_clipped:
        trace_status = "absent"          # absence beats truncation in the flag
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
