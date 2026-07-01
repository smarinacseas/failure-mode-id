"""Live progress monitoring for the eval pipeline.

A single `RunMonitor` owns all counting/timing and fans state changes out to
pluggable sinks (console bars, log file, progress.json). Pipeline steps emit
semantic events and know nothing about rendering. See
docs/superpowers/specs/2026-07-01-experiment-progress-monitoring-design.md.

This file is built up across several tasks; Task 2 adds Stage + WorkPlan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
