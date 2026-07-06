"""Generate candidate responses via OpenRouter (greedy, resumable)."""

from __future__ import annotations

from config import DATA_JSONL, router
from pipeline._io import append_jsonl, read_jsonl, retry
from pipeline._select import select_prompts
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import RunConfig


# Three knobs with non-obvious defaults, all forced by Qwen3.5 being a
# thinking-mode family (see meta/2026-06-30-smoke-test.md):
#   1. reasoning off by default — thinking and the visible answer share one
#      completion budget, so the answer only appears if thinking terminates.
#   2. max_tokens=8000 — at 4000 complex responses truncate mid-answer.
#      Reasoning-on runs need far more headroom (thinking + answer).
#   3. temperature — thinking mode must NOT run greedy: Qwen's model card
#      warns greedy decoding causes endless repetitions, observed 2026-07-02
#      as temp 0.0 + reasoning burning the whole 32k budget on CoT with zero
#      visible content on every call. Qwen recommends 0.6 for thinking mode.
# All live on the per-experiment RunConfig so every run's snapshot records
# what was actually used.
def _generate_one(cfg: RunConfig, model_id: str, prompt: str) -> dict:
    """Return response-record fields: the visible answer, finish_reason, and
    (when the provider returns one) the chain-of-thought — kept out of the
    judge's view but stored for post-hoc failure-mode analysis."""
    def _call():
        resp = router.chat.completions.create(
            model=model_id,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            extra_body=cfg.extra_body,
            timeout=cfg.timeout_s,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        reasoning = (getattr(choice.message, "reasoning", None)
                     or (getattr(choice.message, "model_extra", None) or {}).get("reasoning")
                     or "")
        finish = getattr(choice, "finish_reason", None)
        # Two zero-content cases, both retriable: a transient empty body from
        # a provider, and a thinking model exhausting max_tokens mid-CoT
        # (finish_reason=length with large reasoning_chars). Stored as-is
        # either would silently grade 0/N downstream; the error text says
        # which case it was.
        if not text.strip():
            raise RuntimeError(
                f"empty completion content from {model_id} "
                f"(finish_reason={finish}, reasoning_chars={len(reasoning)})"
            )
        fields = {"response": text, "finish_reason": finish}
        if reasoning:
            fields["reasoning"] = reasoning
        return fields

    return retry(_call, label=f"openrouter:{model_id}")


def run(cfg: RunConfig, monitor: RunMonitor | None = None) -> None:
    records = select_prompts(read_jsonl(DATA_JSONL), cfg.limit, cfg.sample_seed)
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
                    fields = _generate_one(cfg, model_id, rec["prompt"])
                except Exception as e:  # noqa: BLE001 — post-retry failure
                    mon.record_error(f"generate {key} {rec['id']}: {type(e).__name__}: {e}")
                    mon.item_done()
                    continue
                append_jsonl(out_path, {"id": rec["id"], **fields})
                mon.item_done()
        mon.end_stage()
