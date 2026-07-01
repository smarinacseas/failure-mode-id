"""Generate candidate responses via OpenRouter (greedy, resumable)."""

from __future__ import annotations

from config import (
    CANDIDATE_EXTRA_BODY,
    CANDIDATE_MAX_TOKENS,
    CANDIDATE_TEMPERATURE,
    CANDIDATE_TIMEOUT_S,
    CANDIDATES,
    DATA_JSONL,
    RESPONSES_DIR,
    router,
)
from pipeline._io import append_jsonl, limited, read_jsonl, retry
from pipeline.monitor import RunMonitor, stage_ctx


def _response_path(key: str):
    return RESPONSES_DIR / f"{key}.jsonl"


# Two deviations from the original spec's `max_tokens=4000`, both forced by
# Qwen3.5 being a thinking-mode family:
#   1. `CANDIDATE_EXTRA_BODY = {"reasoning": {"enabled": False}}` —
#      without this, Qwen burns the entire token budget on internal
#      chain-of-thought (15k+ reasoning tokens observed on the rota
#      prompt) and emits zero visible `content`.
#   2. `CANDIDATE_MAX_TOKENS = 8000` — complex multi-constraint prompts
#      produce 4k+ token responses; at 4000 the model truncated
#      mid-answer with finish_reason=length, which is unfair to the
#      grader.
# Both knobs live in `config.py` so every experiment's snapshot records
# what was actually used. Net effect: a faithful no-CoT eval.
def _generate_one(model_id: str, prompt: str) -> str:
    def _call():
        resp = router.chat.completions.create(
            model=model_id,
            temperature=CANDIDATE_TEMPERATURE,
            max_tokens=CANDIDATE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            extra_body=CANDIDATE_EXTRA_BODY,
            timeout=CANDIDATE_TIMEOUT_S,
        )
        return resp.choices[0].message.content or ""

    return retry(_call, label=f"openrouter:{model_id}")


def run(limit: int | None = None, monitor: RunMonitor | None = None) -> None:
    records = limited(read_jsonl(DATA_JSONL), limit)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")

    with stage_ctx(monitor, "generate", len(records)) as mon:
        total = len(records) * len(CANDIDATES)
        plan: list[tuple[str, str, list[dict]]] = []
        already = 0
        for key, model_id in CANDIDATES.items():
            done_ids = {r["id"] for r in read_jsonl(_response_path(key))}
            todo = [r for r in records if r["id"] not in done_ids]
            already += len(done_ids)
            plan.append((key, model_id, todo))

        mon.start_stage("generate", total=total, already_done=already)
        for key, model_id, todo in plan:
            out_path = _response_path(key)
            for rec in todo:
                mon.item_start(model=key, prompt_id=rec["id"])
                try:
                    text = _generate_one(model_id, rec["prompt"])
                except Exception as e:  # noqa: BLE001 — post-retry failure
                    mon.record_error(f"generate {key} {rec['id']}: {type(e).__name__}: {e}")
                    mon.item_done()
                    continue
                append_jsonl(out_path, {"id": rec["id"], "response": text})
                mon.item_done()
        mon.end_stage()


if __name__ == "__main__":
    run()
