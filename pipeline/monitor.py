"""Live progress monitoring for the eval pipeline.

A single `RunMonitor` owns all counting/timing and fans state changes out to
pluggable sinks (console bars, log file, progress.json). Pipeline steps emit
semantic events and know nothing about rendering. See
docs/superpowers/specs/2026-07-01-experiment-progress-monitoring-design.md.

This file is built up across several tasks; Task 2 adds Stage + WorkPlan.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Stage order + how each stage's total is derived from (limit L, n_candidates M).
_STEP_ORDER = ("connectivity", "load", "generate", "grade", "classify", "validate", "aggregate")


def _stage_total(name: str, limit: int, m: int) -> int:
    return {
        "connectivity": m + 1,
        "load": limit,
        "generate": limit * m,
        "grade": limit * m,
        "classify": limit,
        "validate": 1,
        "aggregate": 1,
    }[name]


@dataclass
class Stage:
    name: str
    total: int
    done: int = 0
    state: str = "pending"                     # pending | running | done
    current_model: str | None = None
    current_prompt_id: str | None = None
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
# Module-level active monitor — lets deep helpers (e.g. _io.retry) report
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
        self._item_start: float | None = None
        self._current: Stage | None = None

    # -- lifecycle ---------------------------------------------------------- #
    def __enter__(self) -> "RunMonitor":
        global ACTIVE
        ACTIVE = self
        self.state = "running"
        self.started_at = _now_iso()
        self._start_monotonic = time.monotonic()
        self._emit()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        global ACTIVE
        self.state = "error" if exc_type else "done"
        self._emit()
        for s in self.sinks:
            s.close(self)
        ACTIVE = None
        return False                          # never suppress exceptions

    # -- events ------------------------------------------------------------- #
    def start_stage(self, name: str, total: int, already_done: int = 0) -> None:
        st = self.plan.get(name)
        st.total = total
        st.done = min(already_done, total)
        st.state = "running"
        self._current = st
        self._log("INFO", f"stage {name} started (total={total}, resumed={st.done})")
        self._emit()

    def item_start(self, model: str | None = None, prompt_id: str | None = None) -> None:
        if self._current is None:
            return
        self._current.current_model = model
        self._current.current_prompt_id = prompt_id
        self._item_start = time.monotonic()
        self._emit()

    def item_done(self) -> None:
        st = self._current
        if st is None:
            return
        if self._item_start is not None:
            st.record_duration(time.monotonic() - self._item_start)
            self._item_start = None
        st.done = min(st.done + 1, st.total)
        self._log("INFO", f"{st.name} {st.current_model} {st.current_prompt_id} done")
        self._emit()

    def end_stage(self) -> None:
        st = self._current
        if st is not None:
            st.state = "done"
            st.done = st.total
            st.current_model = st.current_prompt_id = None
        self._emit()

    def record_retry(self, label: str) -> None:
        self.retries += 1
        self._log("WARNING", f"retry {label}")
        self._emit()

    def record_error(self, context: str) -> None:
        self.errors += 1
        self._log("ERROR", context)
        self._emit()

    def note(self, message: str) -> None:
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
        cur = self._current
        done, total = self._overall()
        eta = self.eta_s()
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
                "model": cur.current_model if cur else None,
                "prompt_id": cur.current_prompt_id if cur else None,
            },
            "overall": {"done": done, "total": total,
                        "pct": round(100.0 * done / total, 1) if total else 0.0},
            "stages": [
                {"name": s.name, "state": s.state, "done": s.done,
                 "total": s.total, "pct": s.pct}
                for s in self.plan.stages
            ],
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
# Rendering — shared by ConsoleSink (Task 5) and `main.py status` (Task 12).
# --------------------------------------------------------------------------- #
def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
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
            if cur.get("stage") == name and cur.get("model"):
                suffix = f" · {cur['model']} · {cur.get('prompt_id') or ''}"
            lines.append(f"{name:<10}{_bar(st['pct'])}  {st['done']}/{st['total']}  "
                         f"{int(st['pct'])}%{suffix}")
    lines.append("")
    lines.append(f"retries: {snap['retries']}   errors: {snap['errors']}   "
                 f"elapsed: {_fmt_dur(snap['elapsed_s'])}")
    return "\n".join(lines)
