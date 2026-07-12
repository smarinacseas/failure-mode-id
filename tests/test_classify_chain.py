import json

import config
import pipeline.classify as classify
from pipeline import _judge_llm
from tests.conftest import make_cfg


def _setup(tmp_path, monkeypatch, chain):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    data = tmp_path / "cc.jsonl"
    data.write_text(json.dumps({"id": "P1", "prompt": "p", "criteria": ["c1"]}) + "\n")
    monkeypatch.setattr(classify, "DATA_JSONL", data)
    return make_cfg(judges=("claude-fable-5",), classifier_chain=chain, limit=None)


def test_refusal_falls_through_to_next_member(tmp_path, monkeypatch):
    def fake_call(spec, system, user, label):
        if spec.key == "claude-fable-5":
            return "", "refusal"
        return '[{"index":1,"verifiability":"auto","gameable":false,"reward_hack":"","ambiguous":false}]', "stop"
    monkeypatch.setattr(_judge_llm, "call_json", fake_call)
    cfg = _setup(tmp_path, monkeypatch, ("claude-fable-5", "claude-opus-4-8"))
    classify.run(cfg)
    rec = json.loads(cfg.criteria_tags_path.read_text())
    assert rec["model"] == "claude-opus-4-8"
    assert rec["tags"][0]["verifiability"] == "auto"


def test_chain_dry_defaults_tags(tmp_path, monkeypatch):
    monkeypatch.setattr(_judge_llm, "call_json", lambda *a, **k: ("not json", "stop"))
    cfg = _setup(tmp_path, monkeypatch, ("claude-fable-5",))
    classify.run(cfg)
    rec = json.loads(cfg.criteria_tags_path.read_text())
    assert rec["model"] is None
    assert rec["tags"][0]["verifiability"] == "judge"   # default-all fallback
