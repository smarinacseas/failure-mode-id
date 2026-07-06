"""Generate candidate responses via OpenRouter (streamed, deadline-guarded,
resumable, threaded).

Calls are STREAMED with a hard wall-clock deadline (config.GENERATION_DEADLINE_S)
because `timeout_s` alone provably cannot bound them: httpx's read timeout
resets on every trickled chunk, and a half-open socket (client slept
mid-stream) delivers neither bytes nor EOF, so no timeout ever fires — E04
lost 10.3 hours to exactly that (meta/2026-07-03-reasoning-smoke.md, lesson 4).
Two mechanisms cover the two failure shapes:
  1. per-chunk elapsed check — catches slow-trickle generations that keep
     resetting the read timer;
  2. a watchdog timer that closes the stream at the deadline — a close from
     another thread makes a read blocked on a dead socket raise, catching the
     no-chunks-at-all hang.
Either abort raises GenerationDeadlineExceeded into the existing retry()
path: it counts as a retry event, then an error if retries exhaust —
identical to every other transient failure (continue-on-error per item).

The (candidate × prompt) work items run on a small thread pool
(config.GENERATION_WORKERS, recorded in the run manifest). Items are
filtered by done_ids BEFORE submission, exactly as the sequential path did;
append_jsonl is per-file-locked so concurrent workers cannot interleave lines.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_JSONL, GENERATION_DEADLINE_S, GENERATION_WORKERS, router
from pipeline._io import append_jsonl, read_jsonl, retry
from pipeline._select import select_prompts
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import RunConfig


class GenerationDeadlineExceeded(TimeoutError):
    """A generation call exceeded the hard wall-clock deadline.

    Retriable: _io.retry() matches "deadline" in the name/message, so an
    abort becomes a retry event and, after attempts exhaust, a per-item error.
    """


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
def _generate_one(cfg: RunConfig, model_id: str, prompt: str,
                  deadline_s: float | None = None) -> dict:
    """Return response-record fields: the visible answer, finish_reason, and
    (when the provider returns one) the chain-of-thought — kept out of the
    judge's view but stored for post-hoc failure-mode analysis.

    The request is identical to the sequential-era call except for the
    transport: `stream=True` so the wall-clock deadline can be enforced
    per chunk. Model params, extra_body (reasoning + provider routing), and
    timeout_s are unchanged — the frozen treatment is untouched.
    """
    deadline = GENERATION_DEADLINE_S if deadline_s is None else deadline_s

    def _call():
        started = time.monotonic()
        stream = router.chat.completions.create(
            model=model_id,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            extra_body=cfg.extra_body,
            timeout=cfg.timeout_s,
            stream=True,
        )

        deadline_hit = threading.Event()

        def _close_stream() -> None:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001 — teardown must never mask the abort
                    pass

        def _abort() -> None:
            # Runs on the watchdog thread: closing the response makes a read
            # blocked on a silent/half-open socket raise instead of waiting
            # forever — the case the per-chunk check can never see.
            deadline_hit.set()
            _close_stream()

        watchdog = threading.Timer(
            max(0.0, deadline - (time.monotonic() - started)), _abort)
        watchdog.daemon = True
        watchdog.start()

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish = None
        try:
            for chunk in stream:
                elapsed = time.monotonic() - started
                if deadline_hit.is_set() or elapsed > deadline:
                    raise GenerationDeadlineExceeded(
                        f"wall-clock deadline exceeded for {model_id}: "
                        f"{elapsed:.0f}s > {deadline:.0f}s (stream aborted; "
                        f"{sum(len(p) for p in reasoning_parts)} reasoning chars, "
                        f"{sum(len(p) for p in content_parts)} content chars so far)"
                    )
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    piece = getattr(delta, "content", None)
                    if piece:
                        content_parts.append(piece)
                    reasoning_piece = (
                        getattr(delta, "reasoning", None)
                        or (getattr(delta, "model_extra", None) or {}).get("reasoning")
                    )
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                if getattr(choice, "finish_reason", None):
                    finish = choice.finish_reason
        except GenerationDeadlineExceeded:
            raise
        except Exception as e:  # noqa: BLE001 — translate watchdog-induced teardown
            if deadline_hit.is_set():
                elapsed = time.monotonic() - started
                raise GenerationDeadlineExceeded(
                    f"wall-clock deadline exceeded for {model_id}: "
                    f"{elapsed:.0f}s > {deadline:.0f}s (stream closed by watchdog "
                    f"while blocked; underlying: {type(e).__name__})"
                ) from e
            raise
        finally:
            watchdog.cancel()
            _close_stream()

        text = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
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


def _run_item(cfg: RunConfig, mon: RunMonitor, key: str, model_id: str, rec: dict) -> None:
    """One (candidate × prompt) work item; continue-on-error, as before."""
    rid = rec["id"]
    mon.item_start(model=key, prompt_id=rid)
    try:
        fields = _generate_one(cfg, model_id, rec["prompt"])
    except Exception as e:  # noqa: BLE001 — post-retry failure
        mon.record_error(f"generate {key} {rid}: {type(e).__name__}: {e}")
        mon.item_done(model=key, prompt_id=rid)
        return
    append_jsonl(cfg.responses_path(key), {"id": rid, **fields})
    mon.item_done(model=key, prompt_id=rid)


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
        items = [(key, model_id, rec) for key, model_id, todo in plan for rec in todo]
        workers = max(1, int(GENERATION_WORKERS))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="generate") as pool:
            futures = [pool.submit(_run_item, cfg, mon, key, model_id, rec)
                       for key, model_id, rec in items]
            try:
                for f in as_completed(futures):
                    f.result()   # workers swallow per-item errors; anything else propagates
            except BaseException:
                # Abrupt termination (Ctrl-C, kill): stop feeding queued items so
                # the process can die promptly; completed appends are already on
                # disk and resume re-derives done_ids from them.
                pool.shutdown(wait=False, cancel_futures=True)
                raise
        mon.end_stage()
