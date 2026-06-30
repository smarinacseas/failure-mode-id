"""Generate candidate responses via OpenRouter (greedy, resumable)."""

from __future__ import annotations

from config import CANDIDATES, DATA_JSONL, RESPONSES_DIR, router
from pipeline._io import append_jsonl, limited, read_jsonl, retry


def _response_path(key: str):
    return RESPONSES_DIR / f"{key}.jsonl"


def _generate_one(model_id: str, prompt: str) -> str:
    def _call():
        resp = router.chat.completions.create(
            model=model_id,
            temperature=0,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
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
