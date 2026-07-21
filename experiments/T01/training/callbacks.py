"""Training callbacks + a generate() throughput meter for the T1.3 runners.

* CSVLogger      : streams every TRL/HF `on_log` dict to JSONL (crash-safe) and
                   rebuilds a CSV (union of all keys) so `step,loss` (SFT) and
                   `step,reward` (GRPO) land under /workspace even if a run is
                   killed at the 3-hour hardcap.
* TimeBudget     : sets should_training_stop once wall time exceeds a budget
                   (the GRPO probe's 3-hour hardcap).
* GenerateMeter  : wraps the policy's `.generate` / `.generate_batch` to measure
                   true rollout throughput (tokens/sec) on the stock generate()
                   backend (PREREG amendment 2026-07-16 (d): no vLLM).
"""
from __future__ import annotations

import csv
import json
import time

from transformers import TrainerCallback


class CSVLogger(TrainerCallback):
    def __init__(self, csv_path: str, jsonl_path: str | None = None):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path or (csv_path.rsplit(".", 1)[0] + ".jsonl")
        self._rows: list[dict] = []
        self._keys: list[str] = []
        self._start: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if self._start is None:
            self._start = time.time()
        row = {"step": state.global_step, "elapsed_s": round(time.time() - self._start, 1)}
        for k, v in logs.items():
            if isinstance(v, (int, float, str)):
                row[k] = v
        self._rows.append(row)
        for k in row:
            if k not in self._keys:
                self._keys.append(k)
        # live JSONL append (survives a hard kill)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self._flush_csv()

    def _flush_csv(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._rows)

    def on_train_end(self, args, state, control, **kwargs):
        self._flush_csv()


class TimeBudget(TrainerCallback):
    """Stop training gracefully once `budget_s` of wall time elapses (hardcap)."""

    def __init__(self, budget_s: float):
        self.budget_s = budget_s
        self._start: float | None = None
        self.hit = False

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._start is not None and time.time() - self._start > self.budget_s:
            self.hit = True
            control.should_training_stop = True
        return control


class GenerateMeter:
    """Monkeypatch a model's generation methods to accumulate wall time and the
    number of newly-generated tokens. Not a TrainerCallback; attach it to the
    unwrapped policy right after the trainer is built. Measures the real
    generate() rollout throughput (the number the probe must report)."""

    def __init__(self):
        self.tokens = 0
        self.seconds = 0.0
        self.calls = 0
        self._orig: list[tuple] = []

    def _wrap(self, obj, name):
        import torch
        orig = getattr(obj, name, None)
        if orig is None:
            return
        self._orig.append((obj, name, orig))

        def timed(*args, **kwargs):
            inp = kwargs.get("input_ids")
            if inp is None and args:
                inp = args[0]
            in_len = int(inp.shape[1]) if hasattr(inp, "shape") and inp.ndim == 2 else 0
            # sync only around real CUDA tensors (keeps unit tests CUDA-free)
            on_cuda = bool(getattr(inp, "is_cuda", False)) and torch.cuda.is_available()
            if on_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            out = orig(*args, **kwargs)
            if on_cuda:
                torch.cuda.synchronize()
            dt = time.time() - t0
            seqs = out if hasattr(out, "shape") else getattr(out, "sequences", None)
            if hasattr(seqs, "shape") and seqs.ndim == 2:
                new = (int(seqs.shape[1]) - in_len) * int(seqs.shape[0])
                self.tokens += max(new, 0)
            self.seconds += dt
            self.calls += 1
            return out

        setattr(obj, name, timed)

    def attach(self, policy):
        """policy = the underlying causal LM (peft_model.base_model.model)."""
        for name in ("generate", "generate_batch"):
            self._wrap(policy, name)
        return self

    @property
    def tok_per_s(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    def summary(self) -> dict:
        return {"calls": self.calls, "gen_tokens": self.tokens,
                "gen_seconds": round(self.seconds, 1), "tok_per_s": round(self.tok_per_s, 1)}
