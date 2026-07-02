"""Generate candidate responses via OpenRouter (greedy, resumable)."""

from __future__ import annotations

from config import DATA_JSONL, router
from pipeline._io import append_jsonl, limited, read_jsonl, retry
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import RunConfig


# Two knobs with non-obvious defaults, both forced by Qwen3.5 being a
# thinking-mode family (see meta/2026-06-30-smoke-test.md):
#   1. reasoning disabled — without it, Qwen burns the entire token budget
#      on internal chain-of-thought and emits zero visible content.
#   2. max_tokens=8000 — at 4000 complex responses truncate mid-answer.
# Both live on the per-experiment RunConfig so every run's snapshot records
# what was actually used.
def _generate_one(cfg: RunConfig, model_id: str, prompt: str) -> str:
    def _call():
        resp = router.chat.completions.create(
            model=model_id,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            extra_body=cfg.extra_body,
            timeout=cfg.timeout_s,
        )
        return resp.choices[0].message.content or ""

    return retry(_call, label=f"openrouter:{model_id}")


def run(cfg: RunConfig, monitor: RunMonitor | None = None) -> None:
    records = limited(read_jsonl(DATA_JSONL), cfg.limit)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")

    with stage_ctx(monitor, "generate", len(records)) as mon:
        total = len(records) * len(cfg.candidates)
        plan: list[tuple[str, str, list[dict]]] = []
        already = 0
        for key, model_id in cfg.candidates.items():
            done_ids = {r["id"] for r in read_jsonl(cfg.responses_path(key))}
            todo = [r for r in records if r["id"] not in done_ids]
            already += len(done_ids)
            plan.append((key, model_id, todo))

        mon.start_stage("generate", total=total, already_done=already)
        for key, model_id, todo in plan:
            out_path = cfg.responses_path(key)
            for rec in todo:
                mon.item_start(model=key, prompt_id=rec["id"])
                try:
                    text = _generate_one(cfg, model_id, rec["prompt"])
                except Exception as e:  # noqa: BLE001 — post-retry failure
                    mon.record_error(f"generate {key} {rec['id']}: {type(e).__name__}: {e}")
                    mon.item_done()
                    continue
                append_jsonl(out_path, {"id": rec["id"], "response": text})
                mon.item_done()
        mon.end_stage()
