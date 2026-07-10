"""BatchStaleGuard: cancel-to-collect for parked Message Batches.

E07 incident (2026-07-10): a 187-request diagnose batch showed
processing=187/succeeded=0 for 15.5h; manual cancellation flipped it to
ended and collection revealed 184 requests had already succeeded
server-side. The guard automates that remedy: after `cancel_after_s` of
in_progress with NO request_counts movement, cancel once — the caller's
existing collect+resubmit attempt loop does the rest.
"""
from types import SimpleNamespace

from pipeline._batch_guard import BatchStaleGuard


def _batch(status="in_progress", processing=10, succeeded=0, errored=0):
    return SimpleNamespace(
        processing_status=status,
        request_counts=SimpleNamespace(processing=processing,
                                       succeeded=succeeded, errored=errored,
                                       canceled=0, expired=0),
    )


class _FakeClient:
    def __init__(self):
        self.cancel_calls = []
        self.messages = SimpleNamespace(batches=SimpleNamespace(
            cancel=lambda batch_id: self.cancel_calls.append(batch_id) or None))


def _guard(client, t, cancel_after_s=100.0):
    return BatchStaleGuard(client, "b1", "test batch", note=lambda m: None,
                           cancel_after_s=cancel_after_s, clock=lambda: t[0])


def test_no_cancel_before_threshold():
    client, t = _FakeClient(), [0.0]
    g = _guard(client, t)
    g.observe(_batch())
    t[0] = 99.0
    g.observe(_batch())
    assert client.cancel_calls == []


def test_cancel_fires_once_after_frozen_counts():
    client, t = _FakeClient(), [0.0]
    g = _guard(client, t)
    g.observe(_batch())
    t[0] = 101.0
    g.observe(_batch())
    assert client.cancel_calls == ["b1"]
    t[0] = 500.0
    g.observe(_batch())          # one-shot: no second cancel
    assert client.cancel_calls == ["b1"]


def test_count_movement_resets_staleness():
    client, t = _FakeClient(), [0.0]
    g = _guard(client, t)
    g.observe(_batch(succeeded=0))
    t[0] = 90.0
    g.observe(_batch(succeeded=5))   # movement — clock resets
    t[0] = 180.0
    g.observe(_batch(succeeded=5))   # frozen for 90s < 100s
    assert client.cancel_calls == []
    t[0] = 191.0
    g.observe(_batch(succeeded=5))   # frozen for 101s
    assert client.cancel_calls == ["b1"]


def test_no_cancel_when_ended_or_canceling():
    client, t = _FakeClient(), [0.0]
    g = _guard(client, t)
    g.observe(_batch())
    t[0] = 500.0
    g.observe(_batch(status="canceling"))
    g.observe(_batch(status="ended"))
    assert client.cancel_calls == []


def test_disabled_guard_never_cancels():
    client, t = _FakeClient(), [0.0]
    g = _guard(client, t, cancel_after_s=None)
    g.observe(_batch())
    t[0] = 1e9
    g.observe(_batch())
    assert client.cancel_calls == []
