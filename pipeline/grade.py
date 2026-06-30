"""Grade candidate responses with the blind Opus judge.

ONE judge call per (prompt, candidate response). Never include the model
name — the judge is structurally blind. JSON parsing is defensive with
one retry; persistent parse failure records all-FAIL with an error note.
"""

from __future__ import annotations

from config import (
    CANDIDATES,
    DATA_JSONL,
    GRADES_DIR,
    JUDGE,
    PROMPTS_DIR,
    RESPONSES_DIR,
    anthropic,
)
from pipeline._io import append_jsonl, limited, read_jsonl, retry
from pipeline._json_extract import extract_json_array

JUDGE_SYSTEM = (PROMPTS_DIR / "judge.txt").read_text(encoding="utf-8")


def _grade_path(key: str):
    return GRADES_DIR / f"{key}.jsonl"


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


def _judge_call(user_msg: str) -> str:
    def _call():
        msg = anthropic.messages.create(
            model=JUDGE,
            max_tokens=4000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))

    return retry(_call, label=f"anthropic:{JUDGE}")


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


def _grade_one(prompt: str, response: str, criteria: list[str]) -> list[dict]:
    user_msg = _user_message(prompt, response, criteria)
    last_err: str | None = None
    for attempt in (1, 2):
        try:
            raw = _judge_call(user_msg)
            parsed = extract_json_array(raw)
            return _normalize_verdicts(parsed, len(criteria))
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 2:
                break
            print(f"  judge parse failed, retrying once ({last_err})")
    # Both attempts failed → record all FAIL with the error note.
    return [
        {"index": i, "verdict": "FAIL", "reason": f"judge_parse_error: {last_err}"}
        for i in range(1, len(criteria) + 1)
    ]


def run(limit: int | None = None) -> None:
    records = limited(read_jsonl(DATA_JSONL), limit)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")
    by_id = {r["id"]: r for r in records}

    for key, _model_id in CANDIDATES.items():
        responses = read_jsonl(RESPONSES_DIR / f"{key}.jsonl")
        if not responses:
            print(f"grade · {key}: no responses yet — skipping.")
            continue
        out_path = _grade_path(key)
        done_ids = {r["id"] for r in read_jsonl(out_path)}
        todo = [r for r in responses if r["id"] in by_id and r["id"] not in done_ids]
        print(f"grade · {key}: {len(todo)} todo / {len(responses)} responses (skipping {len(done_ids)} done)")

        for resp_rec in todo:
            rid = resp_rec["id"]
            rec = by_id[rid]
            verdicts = _grade_one(rec["prompt"], resp_rec["response"], rec["criteria"])
            append_jsonl(out_path, {"id": rid, "verdicts": verdicts})
            n_pass = sum(1 for v in verdicts if v["verdict"] == "PASS")
            print(f"  grade {rid} · {key}: {n_pass}/{len(verdicts)} pass ✓")


if __name__ == "__main__":
    run()
