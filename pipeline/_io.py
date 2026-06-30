"""Tiny I/O + retry helpers shared across pipeline steps."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
                or "overloaded" in msg
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
            print(f"  retry {label} in {delay:.1f}s ({type(e).__name__}: {e})")
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def limited(records: list[dict], limit: int | None) -> list[dict]:
    return records if limit is None else records[:limit]


def iter_progress(records: list[dict], label: str) -> Iterator[tuple[int, dict]]:
    n = len(records)
    for i, r in enumerate(records, start=1):
        yield i, r
        if i % 25 == 0:
            print(f"  {label}: {i}/{n}")
