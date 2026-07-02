"""Shared judge/classifier LLM call: adaptive thinking, streamed, generous budget.

Both graders and the criterion classifier go through here. Judges reason before
emitting their JSON — thinking is always on for Fable and turned on explicitly
for Opus (`thinking={"type": "adaptive"}`) so the comparison is apples-to-apples.

Thinking tokens count against `max_tokens`; with only a few thousand tokens a
thinking model truncates *before* the JSON array is emitted, which surfaced as
`no parseable JSON array found` and silently poisoned a whole run. So we give a
generous JUDGE_MAX_TOKENS and stream the call — streaming sidesteps the SDK's
non-streaming timeout guard on large budgets. Empty-text thinking blocks (the
default `display: "omitted"`) are skipped by the `type == "text"` filter, so the
returned text is exactly the model's final answer. `stop_reason` is returned so
callers can tell a truncation (`max_tokens`) apart from a genuine parse failure.
"""

from __future__ import annotations

from config import JUDGE_MAX_TOKENS, anthropic
from pipeline._io import retry


def call_json(judge: str, system: str, user_msg: str, label: str) -> tuple[str, str | None]:
    """Return (final_text, stop_reason) from a streamed, thinking-enabled judge call."""
    def _call() -> tuple[str, str | None]:
        with anthropic.messages.stream(
            model=judge,
            max_tokens=JUDGE_MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return text, msg.stop_reason

    return retry(_call, label=label)
