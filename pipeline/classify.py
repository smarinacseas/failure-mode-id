"""Classify each criterion for verifiability / gameability / ambiguity.

ONE call per prompt (model-independent). Output appended to
runs/<slug>/criteria_tags.jsonl. Resumable. Walks classifier_chain per prompt:
refusal advances with no retry; parse failure retries once then advances.
"""

from __future__ import annotations

from config import DATA_JSONL, PROMPTS_DIR
from pipeline._io import append_jsonl, read_jsonl
from pipeline._judge_llm import call_json_chain
from pipeline._select import select_prompts
from pipeline._json_extract import extract_json_array
from pipeline.monitor import RunMonitor, stage_ctx
from pipeline.run_config import RunConfig

CLASSIFIER_SYSTEM = (PROMPTS_DIR / "classifier.txt").read_text(encoding="utf-8")


def _user_message(criteria: list[str]) -> str:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria, start=1))
    return f"CRITERIA:\n{numbered}"


def _normalize_tags(parsed: list, n_criteria: int) -> list[dict]:
    by_index: dict[int, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        verifiability = str(item.get("verifiability", "")).lower().strip()
        if verifiability not in {"auto", "judge"}:
            verifiability = "judge"
        gameable = bool(item.get("gameable", False))
        reward_hack = str(item.get("reward_hack", "") or "")
        ambiguous = bool(item.get("ambiguous", False))
        by_index[idx] = {
            "index": idx,
            "verifiability": verifiability,
            "gameable": gameable,
            "reward_hack": reward_hack if gameable else "",
            "ambiguous": ambiguous,
        }

    out: list[dict] = []
    for i in range(1, n_criteria + 1):
        out.append(by_index.get(i, {
            "index": i,
            "verifiability": "judge",
            "gameable": False,
            "reward_hack": "",
            "ambiguous": False,
        }))
    return out


def _classify_one(cfg: RunConfig, criteria: list[str]) -> tuple[list[dict], str | None]:
    """Walk cfg.classifier_chain per prompt (spec §4): refusal advances with
    no retry; parse failure retries once then advances; chain dry -> default
    tags + error note. Returns (tags, producing_judge_key_or_None)."""
    user_msg = _user_message(criteria)

    def _parse(raw: str, stop: str | None):
        if stop == "refusal":
            return None, "classifier_refusal"
        if stop == "max_tokens":
            return None, "classifier_truncated"
        try:
            return _normalize_tags(extract_json_array(raw), len(criteria)), ""
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"

    tags, spec, err = call_json_chain(cfg.classifier_chain, CLASSIFIER_SYSTEM,
                                      user_msg, "classify", parse=_parse)
    if tags is not None:
        return tags, spec.key
    from pipeline.monitor import note_error
    note_error(f"classifier chain exhausted: {err}, defaulting all tags.")
    return _normalize_tags([], len(criteria)), None


def run(cfg: RunConfig, monitor: RunMonitor | None = None) -> None:
    records = select_prompts(read_jsonl(DATA_JSONL), cfg.limit, cfg.sample_seed)
    if not records:
        raise RuntimeError(f"No records in {DATA_JSONL}. Run `load` first.")

    with stage_ctx(monitor, "classify", len(records)) as mon:
        done_ids = {r["id"] for r in read_jsonl(cfg.criteria_tags_path)}
        todo = [r for r in records if r["id"] not in done_ids]
        mon.start_stage("classify", total=len(records), already_done=len(done_ids))
        for rec in todo:
            mon.item_start(prompt_id=rec["id"])
            tags, model_key = _classify_one(cfg, rec["criteria"])
            append_jsonl(cfg.criteria_tags_path, {"id": rec["id"], "tags": tags, "model": model_key})
            mon.item_done()
        mon.end_stage()
