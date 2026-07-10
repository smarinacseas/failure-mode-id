"""decode_health: mechanical loop census over stored + rejected responses.

User ruling (2026-07-09): any repetition loop — reasoning or content, escaped
or not — counts as a failure and must be captured as auto-verifiable signal.
"""
import config
from conftest import make_cfg
from pipeline._decode_health import decode_health_block, detect_repetition_loop
from pipeline._io import append_jsonl

LOOP = "and around it goes again, the very same words. " * 400   # unit 47
PROSE = " ".join(f"w{i}" for i in range(2500))                   # varied, no loop


def test_detector_reexport_matches_probe():
    from scripts.probe_degenerate import detect_repetition_loop as probe_detector
    assert probe_detector is detect_repetition_loop


def test_decode_health_block_tallies_and_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()

    # healthy: no loop, finish stop -> not in rows
    append_jsonl(cfg.responses_path("m1"),
                 {"id": "p1", "response": PROSE, "finish_reason": "stop"})
    # escaped reasoning loop: loop + stop -> flagged, escaped
    append_jsonl(cfg.responses_path("m1"),
                 {"id": "p2", "response": "fine answer", "reasoning": LOOP,
                  "finish_reason": "stop"})
    # died content loop: loop + length -> flagged, died
    append_jsonl(cfg.responses_path("m1"),
                 {"id": "p3", "response": LOOP, "finish_reason": "length"})
    # truncated but not looping -> flagged via finish only
    append_jsonl(cfg.responses_path("m1"),
                 {"id": "p4", "response": PROSE, "finish_reason": "length"})
    # rejected attempts sidecar: one looping draft, one empty
    append_jsonl(cfg.run_dir / "rejected" / "m1.jsonl",
                 {"id": "p9", "error": "empty_completion", "response": "",
                  "reasoning": LOOP, "finish_reason": "length"})
    append_jsonl(cfg.run_dir / "rejected" / "m1.jsonl",
                 {"id": "p9", "error": "deadline", "response": "",
                  "reasoning": PROSE, "finish_reason": None})

    block = decode_health_block(cfg)

    t = block["by_model"]["m1"]
    assert t["n_responses"] == 4
    assert t["n_loop_reasoning"] == 1
    assert t["n_loop_content"] == 1
    assert t["n_loop_any"] == 2
    assert t["n_loop_escaped"] == 1
    assert t["n_loop_died"] == 1
    assert t["n_nonstop_finish"] == 2
    assert t["rejected_attempts"] == {"n_captured": 2, "n_looping": 1}

    flagged = {r["id"] for r in block["rows"]}
    assert flagged == {"p2", "p3", "p4"}
    p2 = next(r for r in block["rows"] if r["id"] == "p2")
    assert p2["reasoning_loop"]["looping"] is True
    assert p2["reasoning_loop"]["period"] == 47
    assert block["detector"]["ngram"] == 40
    assert "loop" in block["verdict_note"]


def test_decode_health_block_empty_run(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()
    block = decode_health_block(cfg)
    assert block["by_model"]["m1"]["n_responses"] == 0
    assert block["rows"] == []


def test_aggregate_folds_decode_health(tmp_path, monkeypatch):
    """results.json must carry the decode_health block (schema 3.2)."""
    import json
    from types import SimpleNamespace

    import pipeline.aggregate as aggregate
    from pipeline._io import write_jsonl

    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, [{"id": "p1", "prompt": "p1", "use_case": "A",
                        "instruction_type": "Negative", "prompt_style": "Direct",
                        "criteria": ["does the thing"]}])
    monkeypatch.setattr(aggregate, "DATA_JSONL", data)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    cfg = make_cfg()

    append_jsonl(cfg.responses_path("m1"),
                 {"id": "p1", "response": LOOP, "finish_reason": "stop"})
    append_jsonl(cfg.grades_path("claude-fable-5", "m1"),
                 {"id": "p1", "verdicts": [{"index": 1, "verdict": "PASS", "reason": ""}]})
    append_jsonl(cfg.criteria_tags_path,
                 {"id": "p1", "tags": [{"index": 1, "verifiability": "auto",
                                        "gameable": False, "reward_hack": "",
                                        "ambiguous": False}]})

    results_path = tmp_path / "results.json"
    monkeypatch.setattr(aggregate, "RESULTS_PATH", results_path)
    monkeypatch.setattr(aggregate, "write_experiment_copy",
                        lambda slug, results: tmp_path / "e.json")
    monkeypatch.setattr(aggregate, "update_index", lambda slug, meta: tmp_path / "i.json")
    monkeypatch.setattr(aggregate, "_sync_dashboard", lambda mon: None)
    monkeypatch.setattr(aggregate, "_merge_manifest", lambda cfg, update: None)
    monkeypatch.setattr(aggregate, "build_meta", lambda cfg, rr, rd, prompts: {
        "counts": {"n_prompts": len(prompts), "n_criteria": 1},
        "experiment": {}, "models": [], "judges": [], "judge": "",
        "git": {}, "config": {}, "validation": {},
    })

    aggregate._run(cfg, None, SimpleNamespace(note=lambda *a, **k: None))

    results = json.loads(results_path.read_text())
    dh = results["decode_health"]
    assert dh["by_model"]["m1"]["n_loop_content"] == 1
    assert dh["by_model"]["m1"]["n_loop_escaped"] == 1
    assert dh["rows"][0]["id"] == "p1"
