"""Tiny I/O + retry helpers shared across pipeline steps."""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")

# One lock per output file so concurrent workers can never interleave lines.
# Keyed by resolved path; the guard lock only protects dict access (cheap),
# the per-file lock is held for the actual write.
_APPEND_LOCKS: dict[str, threading.Lock] = {}
_APPEND_LOCKS_GUARD = threading.Lock()


def _append_lock(path: Path) -> threading.Lock:
    key = str(Path(path).resolve())
    with _APPEND_LOCKS_GUARD:
        return _APPEND_LOCKS.setdefault(key, threading.Lock())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record as a single line, atomically w.r.t. other threads.

    Contract: on success the record is appended whole (serialize first, then
    one write() of line+newline under this file's lock, then flush); on any
    failure nothing is written. Records from concurrent workers can therefore
    never interleave or tear — resume logic (`done_ids` from re-reading the
    file) stays byte-simple.
    """
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _append_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def retry(
    fn: Callable[[], T],
    *,
    label: str,
    attempts: int = 5,
    base_delay: float = 1.0,
) -> T:
    """Call fn() with exponential backoff on transient errors.

    Retries 429s, 5xx, network/timeout, and provider SDK parse errors —
    OpenRouter occasionally returns mid-stream-truncated bodies that
    bubble up as JSONDecodeError before our code ever sees them.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — provider SDKs raise heterogeneous types
            last_exc = e
            msg = str(e).lower()
            name = type(e).__name__.lower()
            retriable = (
                "429" in msg
                or "rate" in msg
                or "timeout" in msg
                or "deadline" in msg              # wall-clock deadline guard abort
                or "deadline" in name
                or "overloaded" in msg
                or "empty completion" in msg       # transient empty body from a provider
                or "503" in msg
                or "502" in msg
                or "504" in msg
                or "500" in msg
                or "connection" in msg
                or "jsondecode" in name           # malformed body from upstream
                or "remoteprotocol" in name        # httpx mid-stream cutoff
                or "readtimeout" in name
                or "apierror" in name
                or "apiconnection" in name
            )
            if i == attempts - 1 or not retriable:
                raise
            delay = base_delay * (2**i) + random.uniform(0, 0.5)
            from pipeline import monitor
            monitor.note_retry(f"{label} in {delay:.1f}s ({type(e).__name__})")
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]
