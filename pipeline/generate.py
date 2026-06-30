"""Generate candidate responses via OpenRouter (greedy, resumable)."""

from __future__ import annotations

from config import CANDIDATES, DATA_JSONL, RESPONSES_DIR, router
from pipeline._io import append_jsonl, limited, read_jsonl, retry


def _response_path(key: str):
    return RESPONSES_DIR / f"{key}.jsonl"


# Two deviations from the original spec's `max_tokens=4000`, both forced by
# Qwen3.5 being a thinking-mode family:
#   1. `reasoning.enabled: False` — without this, Qwen burns the entire
#      token budget on internal chain-of-thought (15k+ reasoning tokens
#      observed on the rota prompt) and emits zero visible `content`.
#   2. `max_tokens=8000` — complex multi-constraint prompts produce 4k+
#      token responses; at 4000 the model truncated mid-answer with
#      finish_reason=length, which is unfair to the grader.
# Net effect: a faithful no-CoT eval of the model's instruction-following.
MAX_TOKENS = 8000
EXTRA_BODY = {"reasoning": {"enabled": False}}


def _generate_one(model_id: str, prompt: str) -> str:
    def _call():
        resp = router.chat.completions.create(
            model=model_id,
            temperature=0,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
            extra_body=EXTRA_BODY,
            timeout=300.0,
        )
        return resp.choices[0].message.content or ""

    return retry(_call, label=f"openrouter:{model_id}")


def run(limit: int | None = None) -> None:
    records = limited(read_jsonl(DATA_JSONL), limit)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")

    for key, model_id in CANDIDATES.items():
        out_path = _response_path(key)
        done_ids = {r["id"] for r in read_jsonl(out_path)}
        todo = [r for r in records if r["id"] not in done_ids]
        print(f"generate · {key}: {len(todo)} todo / {len(records)} total (skipping {len(done_ids)} done)")
        for rec in todo:
            text = _generate_one(model_id, rec["prompt"])
            append_jsonl(out_path, {"id": rec["id"], "response": text})
            print(f"  gen {rec['id']} · {key} ✓")


if __name__ == "__main__":
    run()
