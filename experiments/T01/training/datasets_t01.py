"""Dataset builders for the T1.3 runners.

SFT (SA, SB): TRL conversational *prompt-completion* rows; prompt = the bare
user task, completion = the teacher's scaffold + ===FINAL=== + answer. With
completion_only_loss the user prompt is masked, so the student learns to emit the
enumerate/compute-then-verify behaviour from the bare task (PREREG §5 / amendment (c)).

GRPO (RA, RB): prompt + a per-prompt `specs` column (JSON string) that the reward
adapter parses to grade each rollout.

Isolation (PREREG §5): each loader hardcodes its base directory (`sft` / `train`)
and validates the cause; there is no code path that can read the reserved
evaluation pool (a datagen static guard additionally enforces that no
training-side source names that pool at all).
"""
from __future__ import annotations

import json
import pathlib

from datasets import Dataset

from common import CAUSES

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"


def _read(base: str, cause: str) -> list[dict]:
    if cause not in CAUSES:
        raise ValueError(f"unknown cause {cause!r}; expected one of {CAUSES}")
    path = DATA_DIR / base / f"{cause}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_sft_dataset(cause: str, limit: int | None = None,
                     only_ids: "set[str] | list[str] | None" = None) -> Dataset:
    rows = _read("sft", cause)
    if only_ids is not None:
        keep = set(only_ids)
        rows = [r for r in rows if r["id"] in keep]   # parity down-sample manifest
    if limit is not None:
        rows = rows[:limit]
    return Dataset.from_dict({
        "prompt": [[{"role": "user", "content": r["prompt"]}] for r in rows],
        "completion": [[{"role": "assistant", "content": r["completion"]}] for r in rows],
    })


def load_grpo_dataset(cause: str, limit: int | None = None) -> Dataset:
    rows = _read("train", cause)
    if limit is not None:
        rows = rows[:limit]
    return Dataset.from_dict({
        "prompt": [[{"role": "user", "content": r["prompt"]}] for r in rows],
        "specs": [json.dumps(r["specs"]) for r in rows],   # carried per-prompt for the reward
    })
