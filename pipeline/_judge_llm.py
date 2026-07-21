"""Shared judge/classifier LLM call: provider-dispatched, streamed, generous budget.

Anthropic branch: byte-identical to the pre-panel code, streamed with
adaptive thinking (thinking tokens count against max_tokens, hence the
generous JUDGE_MAX_TOKENS; streaming sidesteps the SDK's non-streaming
timeout guard).

OpenRouter branch: the same `router` client generation uses, streamed chat
completion with reasoning enabled, plus the per-chunk wall-clock deadline +
watchdog pair generate.py grew after the E04 half-open-socket incident
(judge calls share that exact failure surface). finish_reason is normalized
into the Anthropic stop_reason vocabulary at THIS boundary so
grade._text_to_verdicts (the single parse/refusal/truncation authority)
never learns about providers:

    finish_reason == "length"          -> "max_tokens"
    finish_reason == "content_filter"  -> "refusal"
    empty final text                   -> "refusal"   (reasoning-only output)
    otherwise                          -> "stop"
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Sequence, TypeVar

import config
from config import JUDGE_MAX_TOKENS, anthropic, router
from pipeline._io import retry
from pipeline.run_config import JudgeSpec

T = TypeVar("T")


class JudgeDeadlineExceeded(TimeoutError):
    """An OpenRouter judge call exceeded JUDGE_DEADLINE_S. Retriable:
    _io.retry() matches "deadline" in the exception name."""


def _call_anthropic(spec: JudgeSpec, system: str, user_msg: str) -> tuple[str, str | None]:
    with anthropic.messages.stream(
        model=spec.model,
        max_tokens=JUDGE_MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return text, msg.stop_reason


def _call_openrouter(spec: JudgeSpec, system: str, user_msg: str) -> tuple[str, str | None]:
    started = time.monotonic()
    stream = router.chat.completions.create(
        model=spec.model,
        max_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_msg}],
        extra_body={"reasoning": {"enabled": True}},
        stream=True,
    )

    deadline = config.JUDGE_DEADLINE_S
    deadline_hit = threading.Event()

    def _close_stream() -> None:
        close = getattr(stream, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001 (teardown must never mask the abort)
                pass

    def _abort() -> None:
        deadline_hit.set()
        _close_stream()

    watchdog = threading.Timer(max(0.0, deadline - (time.monotonic() - started)), _abort)
    watchdog.daemon = True
    watchdog.start()

    parts: list[str] = []
    finish = None
    try:
        for chunk in stream:
            elapsed = time.monotonic() - started
            if deadline_hit.is_set() or elapsed > deadline:
                raise JudgeDeadlineExceeded(
                    f"judge deadline exceeded for {spec.key} ({spec.model}): "
                    f"{elapsed:.0f}s > {deadline:.0f}s")
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                piece = getattr(delta, "content", None)
                if piece:
                    parts.append(piece)
            if getattr(choice, "finish_reason", None):
                finish = choice.finish_reason
    except JudgeDeadlineExceeded:
        raise
    except Exception as e:  # noqa: BLE001 (translate watchdog-induced teardown)
        if deadline_hit.is_set():
            raise JudgeDeadlineExceeded(
                f"judge deadline exceeded for {spec.key} ({spec.model}): stream "
                f"closed by watchdog while blocked; underlying: {type(e).__name__}") from e
        raise
    finally:
        watchdog.cancel()
        _close_stream()

    text = "".join(parts)
    if finish == "length":
        return text, "max_tokens"
    if finish == "content_filter" or not text.strip():
        return text, "refusal"
    return text, "stop"


def call_json(judge: JudgeSpec | str, system: str, user_msg: str,
              label: str) -> tuple[str, str | None]:
    """Return (final_text, stop_reason) from one judge call, dispatched on
    the spec's client. Plain strings hydrate to Anthropic specs (legacy)."""
    spec = JudgeSpec.from_value(judge)

    def _call() -> tuple[str, str | None]:
        if spec.client == "openrouter":
            return _call_openrouter(spec, system, user_msg)
        return _call_anthropic(spec, system, user_msg)

    return retry(_call, label=label)


def call_json_chain(
    chain: Sequence[JudgeSpec], system: str, user_msg: str, label_suffix: str,
    parse: Callable[[str, str | None], tuple[T | None, str]],
) -> tuple[T | None, JudgeSpec | None, str]:
    """Walk a fallback chain per item (spec §4). Per member: a refusal
    (parse returned an err while stop_reason == "refusal") advances the chain
    with NO second attempt; a parse failure/truncation gets ONE retry with the
    same member, then advances; a transport error (retry() exhausted) advances.
    Returns (result, producing_spec, "") on success or (None, None, last_err)."""
    last_err = "chain empty"
    for spec in chain:
        for attempt in (1, 2):
            try:
                raw, stop = call_json(spec, system, user_msg,
                                      label=f"{spec.client}:{spec.key}:{label_suffix}")
            except Exception as e:  # noqa: BLE001 (post-retry transport failure)
                last_err = f"{type(e).__name__}: {e}"
                break                      # next chain member
            result, err = parse(raw, stop)
            if result is not None:
                return result, spec, ""
            last_err = err
            if stop == "refusal":
                break                      # sticky per model, next member now
    return None, None, last_err
