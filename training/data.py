"""Load and format the VerIH instruction-hierarchy dataset for RLVR.

VerIH pairs a system instruction with a (possibly conflicting) user
instruction; the OUTPUT constraints (format/quantity/keyword) are
deterministically checkable — that check is the reward (training/reward.py).
The dataset lives in the skai-research/VerIH GitHub repo, NOT on HuggingFace
Hub — clone into data/verih/ (gitignored; see Step 4 in the task brief):

    git clone https://github.com/skai-research/VerIH data/verih

The canonical files are data/verih/RLVR/dataset/verih/{train,test}.json
(confirmed against data/verih/RLVR/dataset/make_parquet.py, VerIH's own
verl-parquet builder, which reads exactly these two files with
target_keys=["sys_prompt", "user_prompt", "gt", "type"]). Real row shape:

    {
      "sys_prompt": str,       # system instruction (may be "")
      "user_prompt": str,      # user instruction, sometimes conflicting
      "explain": str,          # why aligned/conflicting (may be "")
      "raw_messages": [...],   # VerIH's own pre-rendered chat turns (unused here)
      "gt": str,                # JSON-encoded verifier spec, e.g.
                                 #   {"func_name": "validate_forbidden_words",
                                 #    "forbidden_words": [...]}
      "type": str,              # VerIH's own category+alignment tag, e.g.
                                 #   "forbidden words:aligned" / "...:conflict"
    }

`gt.func_name` is one of 24 IFEval-style verifier names and does NOT map
1:1 to training/reward.py's 6-type taxonomy (max_words/min_words/
keyword_include/keyword_forbid/sentence_count/json_parses) — see the task
report for the full func_name inventory and mapping gap. `to_sample` does
not attempt that remapping (out of scope here); it parses `gt` and returns
it verbatim as the sole constraint dict, matching how VerIH's own
make_parquet.py passes `gt` straight through as `reward_model.ground_truth`.

`to_sample` isolates the field mapping so only this one function changes if
the real column names differ; keep it a pure dict→dict transform (unit-tested)
and keep `load_verih` a thin JSON/JSONL reader.
"""

from __future__ import annotations

import json
from pathlib import Path


def to_sample(row: dict) -> dict:
    """One raw VerIH row → {messages, constraints}.

    messages: system (from sys_prompt, omitted if blank) + user (from
    user_prompt). constraints: a single-element list holding the parsed
    `gt` verifier spec verbatim (empty list if `gt` is absent/blank) —
    `gt`'s "func_name"/params are VerIH's own schema, not training/reward.py's
    constraint "type" taxonomy (see module docstring).
    """
    messages = []
    if row.get("sys_prompt"):
        messages.append({"role": "system", "content": row["sys_prompt"]})
    messages.append({"role": "user", "content": row["user_prompt"]})
    gt = row.get("gt")
    constraints = [json.loads(gt)] if gt else []
    return {"messages": messages, "constraints": constraints}


def load_verih(path: str, limit: int | None = None) -> list[dict]:
    """Read a VerIH JSON (array) or JSONL file → formatted samples."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    rows = (json.loads(text) if stripped.startswith("[")
            else [json.loads(line) for line in text.splitlines() if line.strip()])
    if limit:
        rows = rows[:limit]
    return [to_sample(r) for r in rows]
