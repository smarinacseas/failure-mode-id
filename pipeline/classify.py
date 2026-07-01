"""Classify each criterion for verifiability / gameability / ambiguity.

ONE call per prompt (model-independent). Output appended to
outputs/criteria_tags.jsonl. Resumable.
"""

from __future__ import annotations

from config import (
    CRITERIA_TAGS_PATH,
    DATA_JSONL,
    JUDGE,
    JUDGE_MAX_TOKENS,
    PROMPTS_DIR,
    anthropic,
)
from pipeline._io import append_jsonl, limited, read_jsonl, retry
from pipeline._json_extract import extract_json_array
from pipeline.monitor import RunMonitor, stage_ctx

CLASSIFIER_SYSTEM = (PROMPTS_DIR / "classifier.txt").read_text(encoding="utf-8")


def _user_message(criteria: list[str]) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1))
    return f"CRITERIA:\n{numbered}"


def _classifier_call(user_msg: str) -> str:
    def _call():
        msg = anthropic.messages.create(
            model=JUDGE,
            max_tokens=JUDGE_MAX_TOKENS,
            system=CLASSIFIER_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))

    return retry(_call, label=f"anthropic:{JUDGE}:classify")


def _normalize_tags(parsed: list, n_criteria: int) -> list[dict]:
    by_index: dict[int, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        verifiability = str(item.get("verifiability", "")).lower().strip()
        if verifiability not in {"auto", "judge"}:
            verifiability = "judge"
        gameable = bool(item.get("gameable", False))
        reward_hack = str(item.get("reward_hack", "") or "")
        ambiguous = bool(item.get("ambiguous", False))
        by_index[idx] = {
            "index": idx,
            "verifiability": verifiability,
            "gameable": gameable,
            "reward_hack": reward_hack if gameable else "",
            "ambiguous": ambiguous,
        }

    out: list[dict] = []
    for i in range(1, n_criteria + 1):
        out.append(by_index.get(i, {
            "index": i,
            "verifiability": "judge",
            "gameable": False,
            "reward_hack": "",
            "ambiguous": False,
        }))
    return out


def _classify_one(criteria: list[str]) -> list[dict]:
    user_msg = _user_message(criteria)
    last_err: str | None = None
    for attempt in (1, 2):
        try:
            raw = _classifier_call(user_msg)
            parsed = extract_json_array(raw)
            return _normalize_tags(parsed, len(criteria))
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 2:
                break
    from pipeline.monitor import note_error
    note_error(f"classifier failed twice: {last_err} — defaulting all tags.")
    return _normalize_tags([], len(criteria))


def run(limit: int | None = None, monitor: RunMonitor | None = None) -> None:
    records = limited(read_jsonl(DATA_JSONL), limit)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")

    with stage_ctx(monitor, "classify", len(records)) as mon:
        done_ids = {r["id"] for r in read_jsonl(CRITERIA_TAGS_PATH)}
        todo = [r for r in records if r["id"] not in done_ids]
        mon.start_stage("classify", total=len(records), already_done=len(done_ids))
        for rec in todo:
            mon.item_start(prompt_id=rec["id"])
            tags = _classify_one(rec["criteria"])
            append_jsonl(CRITERIA_TAGS_PATH, {"id": rec["id"], "tags": tags})
            mon.item_done()
        mon.end_stage()


if __name__ == "__main__":
    run()
