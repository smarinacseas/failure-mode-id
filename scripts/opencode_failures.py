"""Pass-1 open coding (spec §3): free-text root-cause descriptions for a
stratified sample of failed criteria, WITHOUT any taxonomy imposed. The
consolidation of these descriptions into pipeline/_taxonomy.py DERIVED is a
one-off, documented in the findings report.

Usage:
    uv run python scripts/opencode_failures.py E05-reasoning-rand20p [--sample 70]

Blinding: reuses diagnose._user_message, so the analyst payload is identical
to Pass 2's: no judge reasons, no model identity (spec §4 applies to both
passes). The instruction differs only in asking for free text instead of a
taxonomy key.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_JSONL, OUTPUTS_DIR, anthropic  # noqa: E402
from pipeline._io import read_jsonl, retry  # noqa: E402
from pipeline._json_extract import extract_json_array  # noqa: E402
from pipeline._select import select_prompts  # noqa: E402
from pipeline.diagnose import _batch_request, _failed_cells, _user_message  # noqa: E402
from pipeline.run_config import resolve  # noqa: E402

OPENCODE_SEED = 20260707

OPENCODE_SYSTEM = (
    "You are a failure analyst for language-model evaluations. Independent "
    "grading determined that the response left specific criteria unmet. For "
    "EACH unmet criterion index: first quote the shortest span of the "
    "response (or reasoning trace, when provided) that shows the failure — "
    "or name what is absent — then describe the ROOT CAUSE of the failure in "
    "your own words, 1–3 sentences, at the level of model behavior (what "
    "went wrong in producing the response), not a restatement of the "
    "criterion. If you cannot independently find the failure, say exactly "
    "that. Do NOT use predefined category names; describe freshly.\n\n"
    "Reply with ONLY a JSON array:\n"
    '[{"index": <int>, "evidence": "<quote or omission note>", '
    '"free_text": "<your root-cause description>"}]'
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--sample", type=int, default=70)
    args = ap.parse_args()

    cfg = resolve(args.slug, {})
    records = select_prompts(read_jsonl(DATA_JSONL), cfg.limit, cfg.sample_seed)
    # Pass 1 bypasses the Pass-2 resume filter by design (spec §3): it samples
    # from ALL failed cells, whether or not they have already been diagnosed.
    all_cells = _failed_cells(cfg, records, include_diagnosed=True)

    # Stratify (criterion, cell) pairs by (model, instruction_type) and draw
    # round-robin with a fixed seed so the sample is reproducible.
    by_id = {r["id"]: r for r in records}
    pairs = [(cell, idx) for cell in all_cells for idx in cell["failed_indices"]]
    strata: dict[tuple, list] = defaultdict(list)
    for cell, idx in pairs:
        strata[(cell["key"], by_id[cell["rid"]]["instruction_type"])].append((cell, idx))
    rng = random.Random(OPENCODE_SEED)
    for group in strata.values():
        rng.shuffle(group)
    picked: list[tuple] = []
    while len(picked) < min(args.sample, len(pairs)):
        for key in sorted(strata, key=lambda k: (-len(strata[k]), k)):
            if strata[key] and len(picked) < min(args.sample, len(pairs)):
                picked.append(strata[key].pop())

    # One request per sampled (cell, criterion): open coding wants independent
    # descriptions, so no cell-level bundling here (unlike Pass 2).
    requests = []
    meta = {}
    for n, (cell, idx) in enumerate(picked):
        single = dict(cell, failed_indices=[idx])
        user_msg, _trace = _user_message(single)
        cid = f"oc{n}"
        meta[cid] = (cell["key"], cell["rid"], idx)
        requests.append(_batch_request(cid, OPENCODE_SYSTEM, user_msg))

    if not requests:
        print(f"no failed criteria to sample for {args.slug}, nothing to do.")
        return 0

    print(f"open-coding {len(requests)} sampled failures from {args.slug} …")
    batch = retry(lambda: anthropic.messages.batches.create(requests=requests),
                  label="anthropic:batches:create:opencode")
    print(f"  batch {batch.id}")
    while True:
        b = retry(lambda: anthropic.messages.batches.retrieve(batch.id),
                  label="anthropic:batches:retrieve:opencode")
        print(f"  {b.processing_status}")
        if b.processing_status == "ended":
            break
        time.sleep(60)

    out = OUTPUTS_DIR / "opencode" / f"{args.slug}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    results = retry(lambda: list(anthropic.messages.batches.results(batch.id)),
                    label="anthropic:batches:results:opencode")
    with out.open("w", encoding="utf-8") as fh:
        for res in results:
            ids = meta.get(res.custom_id)
            if ids is None:
                continue
            key, rid, idx = ids
            if res.result.type != "succeeded":
                print(f"  ! {key}/{rid}#{idx}: {res.result.type}")
                continue
            raw = "".join(blk.text for blk in res.result.message.content
                          if getattr(blk, "type", None) == "text")
            try:
                (item,) = extract_json_array(raw)
                fh.write(json.dumps({"key": key, "rid": rid, "index": idx,
                                     "evidence": item.get("evidence", ""),
                                     "free_text": item.get("free_text", "")},
                                    ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ! {key}/{rid}#{idx}: parse {type(e).__name__}: {e}")
    print(f"wrote {n_ok}/{len(requests)} descriptions → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
