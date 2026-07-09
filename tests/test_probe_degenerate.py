"""Loop-detector and forensic-row unit tests for the E07 degenerate probe."""
import json
import random

import pytest

import config
import scripts.probe_degenerate as probe
from pipeline._io import append_jsonl, read_jsonl
from pipeline.run_config import RunConfig
from scripts.probe_degenerate import detect_repetition_loop, forensic_row


def test_healthy_varied_prose_not_looping():
    rng = random.Random(7)
    text = " ".join(f"w{rng.randrange(1000)}" for _ in range(3000))
    assert detect_repetition_loop(text)["looping"] is False


def test_pure_repetition_loops_with_period():
    text = "The same sentence keeps repeating here. " * 500  # unit length 40
    r = detect_repetition_loop(text)
    assert r["looping"] is True
    assert r["period"] == 40
    assert r["onset"] == 0


def test_single_char_runaway():
    assert detect_repetition_loop("a" * 5000) == {"looping": True, "period": 1, "onset": 0}


def test_loop_after_healthy_prefix_reports_onset():
    prefix = " ".join(f"w{i}" for i in range(600))
    unit = "and then it loops forever with this exact phrase. "  # length 51
    text = prefix + unit * 300
    r = detect_repetition_loop(text)
    assert r["looping"] is True
    assert r["period"] == len(unit)
    assert len(prefix) <= r["onset"] < len(prefix) + len(unit)


def test_short_text_not_looping():
    assert detect_repetition_loop("abc " * 10)["looping"] is False


def test_forensic_row_shape():
    rec = {"id": "CIF-012", "response": "x" * 10, "finish_reason": "stop"}
    row = forensic_row("E07-reasoning-full75", "qwen-9b", rec)
    assert row["source"] == "E07-reasoning-full75"
    assert row["model"] == "qwen-9b"
    assert row["id"] == "CIF-012"
    assert row["finish_reason"] == "stop"
    assert row["content_chars"] == 10
    assert row["reasoning_chars"] == 0
    assert row["content_loop"]["looping"] is False
    assert row["reasoning_loop"]["looping"] is False


def test_load_frozen_cfg_refuses_missing_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    with pytest.raises(SystemExit):
        probe.load_frozen_cfg("E99-missing")


def test_load_frozen_cfg_reads_frozen_params(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    run_dir = tmp_path / "E99-probe-test"
    run_dir.mkdir(parents=True)
    params = {
        "candidates": {"qwen-9b": "prov/9b", "qwen-35b": "prov/35b"},
        "judges": ["claude-opus-4-8"], "max_tokens": 64000, "temperature": 0.6,
        "reasoning": True, "timeout_s": 600.0, "limit": 75,
        "description": "t", "provider_sort": "throughput",
        "sample_seed": None, "judge_mode": "batch",
    }
    (run_dir / "experiment.json").write_text(
        json.dumps({"schema": 1, "params": params}), encoding="utf-8")
    cfg = probe.load_frozen_cfg("E99-probe-test")
    assert cfg.max_tokens == 64000 and cfg.reasoning is True


def test_run_draws_resumes_and_skips_done(tmp_path, monkeypatch):
    calls = []

    def fake_generate(cfg, model_id, prompt):
        calls.append(model_id)
        return {"response": "ok", "finish_reason": "stop"}

    monkeypatch.setattr(probe, "_generate_one", fake_generate)
    out = tmp_path / "draws.jsonl"
    for d in range(1, 6):  # qwen-9b fully done
        append_jsonl(out, {"model": "qwen-9b", "draw": d,
                           "response": "x", "finish_reason": "stop"})
    for d in range(1, 4):  # qwen-35b 3 of 5 done
        append_jsonl(out, {"model": "qwen-35b", "draw": d,
                           "response": "x", "finish_reason": "stop"})
    cfg = RunConfig(slug="E99-probe-test",
                    candidates={"qwen-9b": "prov/9b", "qwen-35b": "prov/35b"},
                    judges=("claude-opus-4-8",), max_tokens=10, temperature=0.0,
                    reasoning=False, timeout_s=1.0, limit=1, description="t")
    made = probe.run_draws(cfg, "prompt text", out)
    assert made == 2
    assert calls == ["prov/35b", "prov/35b"]
    assert len(read_jsonl(out)) == 10


def test_run_draws_continues_past_per_draw_errors(tmp_path, monkeypatch):
    def flaky(cfg, model_id, prompt):
        if model_id == "prov/9b":
            raise RuntimeError("boom")
        return {"response": "ok", "finish_reason": "stop"}

    monkeypatch.setattr(probe, "_generate_one", flaky)
    out = tmp_path / "draws.jsonl"
    cfg = RunConfig(slug="E99-probe-test",
                    candidates={"qwen-9b": "prov/9b", "qwen-35b": "prov/35b"},
                    judges=("claude-opus-4-8",), max_tokens=10, temperature=0.0,
                    reasoning=False, timeout_s=1.0, limit=1, description="t")
    made = probe.run_draws(cfg, "prompt text", out, k=2)
    assert made == 2  # qwen-35b's two draws landed despite qwen-9b failing
    assert {r["model"] for r in read_jsonl(out)} == {"qwen-35b"}
