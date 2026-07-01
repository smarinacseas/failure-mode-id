"""Load ComplexConstraints.xlsx into data/complexconstraints.jsonl.

Each output record:
    {id, prompt, use_case, instruction_type, prompt_style, criteria: [str, ...]}
"""

from __future__ import annotations

import math

import pandas as pd

from config import DATA_JSONL, DATA_XLSX
from pipeline._io import write_jsonl
from pipeline.monitor import RunMonitor, stage_ctx

REQUIRED_COLS = ("benchmark_id", "prompt", "use_case", "instruction_type", "prompt_style")


def _clean_criteria(row: pd.Series) -> list[str]:
    out: list[str] = []
    for col in row.index:
        if not col.startswith("criterion_"):
            continue
        v = row[col]
        if isinstance(v, float) and math.isnan(v):
            continue
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def run(limit: int | None = None, monitor: RunMonitor | None = None) -> int:
    df = pd.read_excel(DATA_XLSX)
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column: {col}")
    if limit is not None:
        df = df.head(limit)

    with stage_ctx(monitor, "load", len(df)) as mon:
        mon.start_stage("load", total=len(df))
        records: list[dict] = []
        for _, row in df.iterrows():
            mon.item_start(prompt_id=str(row["benchmark_id"]))
            records.append({
                "id": str(row["benchmark_id"]),
                "prompt": str(row["prompt"]),
                "use_case": str(row["use_case"]),
                "instruction_type": str(row["instruction_type"]),
                "prompt_style": str(row["prompt_style"]),
                "criteria": _clean_criteria(row),
            })
            mon.item_done()
        write_jsonl(DATA_JSONL, records)
        mon.note(f"load: wrote {len(records)} prompts → {DATA_JSONL}")
        mon.end_stage()
    return len(records)


if __name__ == "__main__":
    run()
