"""Cancel-to-collect guard for parked Anthropic Message Batches.

Incident (E07, 2026-07-10): a 187-request diagnose batch sat `in_progress`
with request_counts frozen at processing=187/succeeded=0 for 15.5 hours —
while grade's same-sized batch had drained in 50 minutes the previous
evening. Manual cancellation flipped the batch to `ended`, and collection
revealed 184 of 187 requests had ALREADY SUCCEEDED server-side: the
status/counts endpoint was stale, not the processing. The wall-clock was
spent waiting on an endpoint that misreported an almost-finished batch.

The guard automates that remedy. Feed it every `batches.retrieve()` result;
once a batch has been `in_progress` with NO request_counts movement for
`cancel_after_s`, it cancels the batch ONCE. Cancellation is safe by
construction here:
  - a canceled batch reaches `ended`, so the caller's poll loop exits;
  - `batches.results()` still returns every request that completed
    server-side (and only completed requests are billed);
  - both grade and diagnose collect inside a two-attempt loop that
    resubmits whatever did not succeed — identical to what happens at the
    24h expiry, just without the wait.

Cost of a false-positive cancel (batch genuinely still working): requests
in flight at cancellation come back canceled/unbilled and are resubmitted,
burning one of the caller's two attempts. The default window is therefore
sized well above the largest healthy drain observed (~50 min for a
220-request Opus batch) — see config.BATCH_CANCEL_COLLECT_AFTER_S.

Counts CAN legitimately stay frozen while work happens (small batches often
show succeeded=0 until the end), so "no movement" is only meaningful after
a long window — this guard is a backstop against queue purgatory, not a
progress meter.
"""

from __future__ import annotations

import time
from typing import Callable

import config
from pipeline._io import retry

_COUNT_FIELDS = ("processing", "succeeded", "errored", "canceled", "expired")

# Default sentinel: "use config.BATCH_CANCEL_COLLECT_AFTER_S". Passing
# cancel_after_s=None explicitly DISABLES the guard.
_USE_CONFIG = object()


def _counts_key(b) -> tuple:
    rc = getattr(b, "request_counts", None)
    if rc is None:
        return ()
    return tuple(getattr(rc, f, None) for f in _COUNT_FIELDS)


class BatchStaleGuard:
    """One guard per submitted batch; call observe() with each retrieve()."""

    def __init__(self, client, batch_id: str, label: str,
                 note: Callable[[str], None],
                 cancel_after_s=_USE_CONFIG,
                 clock: Callable[[], float] = time.monotonic):
        self._client = client
        self._batch_id = batch_id
        self._label = label
        self._note = note
        self._cancel_after_s = (config.BATCH_CANCEL_COLLECT_AFTER_S
                                if cancel_after_s is _USE_CONFIG
                                else cancel_after_s)
        self._clock = clock
        self._last_counts: tuple | None = None
        self._last_change = clock()
        self._canceled = False

    def observe(self, b) -> None:
        now = self._clock()
        counts = _counts_key(b)
        if counts != self._last_counts:
            self._last_counts = counts
            self._last_change = now
        if self._canceled or self._cancel_after_s is None:
            return
        if getattr(b, "processing_status", None) != "in_progress":
            return
        stale_for = now - self._last_change
        if stale_for < self._cancel_after_s:
            return
        self._canceled = True
        self._note(
            f"{self._label}: request_counts frozen for {stale_for / 60:.0f}m "
            f"— canceling to force collection (results completed server-side "
            f"are preserved; the remainder resubmits)"
        )
        try:
            retry(lambda: self._client.messages.batches.cancel(self._batch_id),
                  label=f"{self._label}:cancel")
        except Exception as e:  # noqa: BLE001 — guard must never kill the poll loop
            self._note(f"{self._label}: cancel failed "
                       f"({type(e).__name__}: {e}) — will retry next poll")
            self._canceled = False
