"""Live progress monitoring for the eval pipeline.

A single `RunMonitor` owns all counting/timing and fans state changes out to
pluggable sinks (console bars, log file, progress.json). Pipeline steps emit
semantic events and know nothing about rendering. See
docs/superpowers/specs/2026-07-01-experiment-progress-monitoring-design.md.

This file is built up across several tasks; Task 2 adds Stage + WorkPlan.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.text import Text

from config import CANDIDATES, LOGS_DIR, PROGRESS_PATH

# Stage order + how each stage's total is derived from (limit L, n_candidates M).
_STEP_ORDER = ("connectivity", "load", "generate", "grade", "classify",
               "diagnose", "validate", "aggregate")


def _stage_total(name: str, limit: int, m: int) -> int:
    return {
        "connectivity": m + 1,
        "load": limit,
        "generate": limit * m,
        "grade": limit * m,
        "classify": limit,
        "diagnose": limit * m,     # upper bound; start_stage() sets the real total
        "validate": 1,
        "aggregate": 1,
    }[name]


@dataclass
class Stage:
    name: str
    total: int
    done: int = 0
    state: str = "pending"                     # pending | running | done
    # In-flight work items: (model, prompt_id) -> monotonic start time.
    # Replaces the old single-item current_model/current_prompt_id fields,
    # since with a worker pool several items are live at once. Insertion-ordered,
    # so a bare item_done() (sequential callers) pops the oldest.
    in_flight: dict = field(default_factory=dict, repr=False)
    _durations: list[float] = field(default_factory=list, repr=False)

    WINDOW = 20

    @property
    def pct(self) -> float:
        return round(100.0 * self.done / self.total, 1) if self.total else 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.done)

    @property
    def rate_s(self) -> float | None:
        """Mean seconds/item over the rolling window, or None if no data yet."""
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)

    def record_duration(self, seconds: float) -> None:
        self._durations.append(seconds)
        if len(self._durations) > self.WINDOW:
            self._durations.pop(0)

    def eta_s(self) -> float | None:
        r = self.rate_s
        return r * self.remaining if r is not None else None


@dataclass
class WorkPlan:
    stages: list[Stage]

    @classmethod
    def for_step(cls, step: str, limit: int, n_candidates: int) -> "WorkPlan":
        names = list(_STEP_ORDER) if step == "all" else [step]
        return cls([Stage(name=n, total=_stage_total(n, limit, n_candidates)) for n in names])

    def get(self, name: str) -> Stage:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)


# --------------------------------------------------------------------------- #
# Sinks (base + test recorder). Concrete file/console sinks land in later tasks.
# --------------------------------------------------------------------------- #
class Sink:
    """Receives monitor updates. Subclasses override only what they need."""

    def update(self, monitor: "RunMonitor") -> None: ...
    def log(self, monitor: "RunMonitor", level: str, message: str) -> None: ...
    def close(self, monitor: "RunMonitor") -> None: ...


class RecordingSink(Sink):
    """In-memory sink for tests: captures every snapshot + log line."""

    def __init__(self) -> None:
        self.snapshots: list[dict] = []
        self.logs: list[tuple[str, str]] = []
        self.closes = 0

    def update(self, monitor: "RunMonitor") -> None:
        self.snapshots.append(monitor.snapshot())

    def log(self, monitor: "RunMonitor", level: str, message: str) -> None:
        self.logs.append((level, message))

    def close(self, monitor: "RunMonitor") -> None:
        self.closes += 1


# --------------------------------------------------------------------------- #
# Module-level active monitor, lets deep helpers (e.g. _io.retry) report
# without threading the monitor through every call signature.
# --------------------------------------------------------------------------- #
ACTIVE: "RunMonitor | None" = None


def note_retry(label: str) -> None:
    if ACTIVE is not None:
        ACTIVE.record_retry(label)


def note_error(context: str) -> None:
    if ACTIVE is not None:
        ACTIVE.record_error(context)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunMonitor:
    def __init__(self, plan: WorkPlan, experiment: str | None = None,
                 sinks: list[Sink] | None = None) -> None:
        self.plan = plan
        self.experiment = experiment
        self.sinks = sinks if sinks is not None else []
        self.state = "idle"
        self.retries = 0
        self.errors = 0
        self.started_at = ""
        self._start_monotonic: float | None = None
        self._current: Stage | None = None
        # Batch-judge stage accounting (submitted/pending/collected/errored);
        # None outside a batch stage. Reported in the heartbeat.
        self.batch: dict | None = None
        # One re-entrant lock guards ALL state mutation and every sink write
        # (progress.json included): note_retry()/note_error()/item_* arrive
        # from worker threads, and an unlocked snapshot mid-mutation could
        # write a torn heartbeat. Re-entrant because sinks call snapshot()
        # back on this monitor from inside _emit().
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "RunMonitor":
        global ACTIVE
        with self._lock:
            ACTIVE = self
            self.state = "running"
            self.started_at = _now_iso()
            self._start_monotonic = time.monotonic()
            self._emit()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        global ACTIVE
        with self._lock:
            self.state = "error" if exc_type else "done"
            self._emit()
            for s in self.sinks:
                s.close(self)
            ACTIVE = None
        return False                          # never suppress exceptions

    # -- events ------------------------------------------------------------- #
    def start_stage(self, name: str, total: int, already_done: int = 0) -> None:
        with self._lock:
            st = self.plan.get(name)
            st.total = total
            st.done = min(already_done, total)
            st.state = "running"
            self._current = st
            self._log("INFO", f"stage {name} started (total={total}, resumed={st.done})")
            self._emit()

    def item_start(self, model: str | None = None, prompt_id: str | None = None) -> None:
        with self._lock:
            if self._current is None:
                return
            self._current.in_flight[(model, prompt_id)] = time.monotonic()
            self._emit()

    def item_done(self, model: str | None = None, prompt_id: str | None = None) -> None:
        """Complete one item. Concurrent callers pass the same identifiers they
        gave item_start(); a bare call (sequential stages) pops the oldest
        in-flight item, preserving the old one-at-a-time behavior."""
        with self._lock:
            st = self._current
            if st is None:
                return
            key = (model, prompt_id)
            if key not in st.in_flight and st.in_flight:
                key = next(iter(st.in_flight))          # oldest (insertion order)
            started = st.in_flight.pop(key, None)
            if started is not None:
                st.record_duration(time.monotonic() - started)
            st.done = min(st.done + 1, st.total)
            self._log("INFO", f"{st.name} {key[0]} {key[1]} done")
            self._emit()

    def end_stage(self) -> None:
        with self._lock:
            st = self._current
            if st is not None:
                st.state = "done"
                st.done = st.total
                st.in_flight.clear()
            self._emit()

    def record_retry(self, label: str) -> None:
        with self._lock:
            self.retries += 1
            self._log("WARNING", f"retry {label}")
            self._emit()

    def record_error(self, context: str) -> None:
        with self._lock:
            self.errors += 1
            self._log("ERROR", context)
            self._emit()

    def set_batch_counts(self, submitted: int, pending: int,
                         collected: int, errored: int) -> None:
        """Heartbeat accounting for the batch-judge stage (updated at submit
        and on every poll): requests submitted, still processing, results
        collected (succeeded), per-item errored results observed."""
        with self._lock:
            self.batch = {"submitted": submitted, "pending": pending,
                          "collected": collected, "errored": errored}
            self._emit()

    def note(self, message: str) -> None:
        with self._lock:
            self._log("NOTE", message)
            self._emit()

    # -- math + snapshot ---------------------------------------------------- #
    @property
    def elapsed_s(self) -> float:
        return 0.0 if self._start_monotonic is None else time.monotonic() - self._start_monotonic

    def _overall(self) -> tuple[int, int]:
        done = sum(s.done for s in self.plan.stages)
        total = sum(s.total for s in self.plan.stages)
        return done, total

    def eta_s(self) -> float | None:
        fallback = self._current.rate_s if self._current is not None else None
        total = 0.0
        seen = False
        for s in self.plan.stages:
            if s.state == "done":
                continue
            rate = s.rate_s if s.rate_s is not None else fallback
            if rate is None:
                continue
            seen = True
            total += rate * s.remaining
        return total if seen else None

    def snapshot(self) -> dict:
        with self._lock:
            cur = self._current
            done, total = self._overall()
            eta = self.eta_s()
            in_flight_items = [
                {"model": m, "prompt_id": p} for (m, p) in cur.in_flight
            ] if cur else []
            return {
                "experiment": self.experiment,
                "state": self.state,
                "pid": os.getpid(),
                "started_at": self.started_at,
                "updated_at": _now_iso(),
                "elapsed_s": round(self.elapsed_s, 1),
                "eta_s": round(eta, 1) if eta is not None else None,
                "current": {
                    "stage": cur.name if cur else None,
                    "in_flight": len(in_flight_items),
                    "in_flight_items": in_flight_items,
                },
                "overall": {"done": done, "total": total,
                            "pct": round(100.0 * done / total, 1) if total else 0.0},
                "stages": [
                    {"name": s.name, "state": s.state, "done": s.done,
                     "total": s.total, "pct": s.pct}
                    for s in self.plan.stages
                ],
                "batch": self.batch,
                "retries": self.retries,
                "errors": self.errors,
            }

    # -- fan-out ------------------------------------------------------------ #
    def _emit(self) -> None:
        for s in self.sinks:
            s.update(self)

    def _log(self, level: str, message: str) -> None:
        for s in self.sinks:
            s.log(self, level, message)


# --------------------------------------------------------------------------- #
# Rendering: shared by ConsoleSink (Task 5) and `main.py status` (Task 12).
# --------------------------------------------------------------------------- #
def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m"
    return f"{sec}s"


def _bar(pct: float, width: int = 16) -> str:
    filled = max(0, min(width, int(round(width * pct / 100.0))))
    return "█" * filled + "░" * (width - filled)


def _in_flight_display(cur: dict) -> tuple[int, list[str]]:
    """(count, labels) from a snapshot's `current` block. Tolerates the
    pre-concurrency single-item shape ({model, prompt_id}) so `status` can
    still render a progress.json written by an older run."""
    items = cur.get("in_flight_items")
    if items is None:
        if cur.get("model") or cur.get("prompt_id"):
            items = [{"model": cur.get("model"), "prompt_id": cur.get("prompt_id")}]
        else:
            items = []
    labels = [" ".join(str(x) for x in (i.get("model"), i.get("prompt_id"))
                       if x is not None) for i in items]
    count = cur.get("in_flight")
    return (len(labels) if count is None else count), labels


def render_lines(snap: dict) -> str:
    o = snap["overall"]
    cur = snap.get("current") or {}
    header = (f"{snap.get('experiment') or 'untagged'}   {snap['state']} · "
              f"{int(o['pct'])}% · elapsed {_fmt_dur(snap['elapsed_s'])} · "
              f"ETA {_fmt_dur(snap['eta_s'])}")
    lines = [header, ""]
    for st in snap["stages"]:
        name = st["name"]
        if st["state"] == "done":
            lines.append(f"{name:<12} ✓ done   {st['done']}/{st['total']}")
        elif st["state"] == "pending":
            lines.append(f"{name:<12} pending  {st['done']}/{st['total']}")
        else:
            suffix = ""
            if cur.get("stage") == name:
                n_flight, labels = _in_flight_display(cur)
                if labels:
                    shown = " · ".join(labels[:4])
                    more = f" (+{n_flight - 4} more)" if n_flight > 4 else ""
                    suffix = f" · {n_flight} in flight · {shown}{more}"
            lines.append(f"{name:<10}{_bar(st['pct'])}  {st['done']}/{st['total']}  "
                         f"{int(st['pct'])}%{suffix}")
    batch = snap.get("batch")
    if batch:
        lines.append("")
        lines.append(f"batch: submitted {batch.get('submitted', 0)} · "
                     f"pending {batch.get('pending', 0)} · "
                     f"collected {batch.get('collected', 0)} · "
                     f"errored {batch.get('errored', 0)}")
    lines.append("")
    lines.append(f"retries: {snap['retries']}   errors: {snap['errors']}   "
                 f"elapsed: {_fmt_dur(snap['elapsed_s'])}")
    return "\n".join(lines)


class StatusSink(Sink):
    """Write snapshot() to a JSON heartbeat, throttled, with atomic replace."""

    def __init__(self, path: Path, min_interval: float = 1.0) -> None:
        self.path = path
        self.min_interval = min_interval
        self._last: float | None = None         # None → the first update always writes

    def _write(self, monitor: "RunMonitor") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(monitor.snapshot(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, self.path)              # atomic: readers never see a partial file
        self._last = time.monotonic()

    def update(self, monitor: "RunMonitor") -> None:
        if self._last is None or time.monotonic() - self._last >= self.min_interval:
            self._write(monitor)

    def close(self, monitor: "RunMonitor") -> None:
        self._write(monitor)                    # always flush the terminal state


_LEVELS = {"NOTE": logging.INFO, "INFO": logging.INFO,
           "WARNING": logging.WARNING, "ERROR": logging.ERROR}


class LogSink(Sink):
    """Append every event to a timestamped log file via stdlib logging."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.logger = logging.getLogger(f"failure_mode_id.monitor.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._handler = logging.FileHandler(path, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(self._handler)

    def log(self, monitor: "RunMonitor", level: str, message: str) -> None:
        self.logger.log(_LEVELS.get(level, logging.INFO), message)

    def close(self, monitor: "RunMonitor") -> None:
        self._handler.flush()
        self.logger.removeHandler(self._handler)
        self._handler.close()


_NOTICE_LEVELS = {"WARNING": "yellow", "ERROR": "red", "NOTE": "cyan"}


class ConsoleSink(Sink):
    """Live rich bars in the terminal; notices print above the live region."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._live: Live | None = None

    def _render(self, monitor: "RunMonitor") -> Text:
        return Text(render_lines(monitor.snapshot()))

    def update(self, monitor: "RunMonitor") -> None:
        if self._live is None:
            self._live = Live(self._render(monitor), console=self.console,
                              refresh_per_second=8, transient=False)
            self._live.start()
        else:
            self._live.update(self._render(monitor))

    def log(self, monitor: "RunMonitor", level: str, message: str) -> None:
        color = _NOTICE_LEVELS.get(level)
        if color is None:                       # INFO item ticks: bars already show them
            return
        self.console.print(f"[{color}]{level}[/{color}] {message}")

    def close(self, monitor: "RunMonitor") -> None:
        if self._live is not None:
            self._live.update(self._render(monitor))
            self._live.stop()
            self._live = None
        done, total = monitor._overall()
        self.console.print(
            f"{monitor.experiment or 'untagged'} {monitor.state} · {done}/{total} items "
            f"· {monitor.retries} retries · {monitor.errors} errors "
            f"· {_fmt_dur(monitor.elapsed_s)}"
        )


def default_sinks(experiment: str | None) -> list[Sink]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOGS_DIR / f"{experiment or 'untagged'}-{stamp}.log"
    return [ConsoleSink(), LogSink(log_path), StatusSink(PROGRESS_PATH)]


def build_monitor(step: str, limit: int, experiment: str | None = None,
                  sinks: list[Sink] | None = None,
                  n_candidates: int | None = None) -> RunMonitor:
    plan = WorkPlan.for_step(step, limit, n_candidates or len(CANDIDATES))
    return RunMonitor(plan, experiment=experiment,
                      sinks=default_sinks(experiment) if sinks is None else sinks)


def stage_ctx(monitor: "RunMonitor | None", step: str, limit_hint: int):
    """Reuse the caller's monitor, or build a standalone one for this step.

    Steps call this so they run both under `main`'s shared monitor (passed in)
    and standalone (`python -m pipeline.generate` / tests) with a fresh one.
    """
    if monitor is not None:
        return nullcontext(monitor)
    return build_monitor(step, limit_hint)
