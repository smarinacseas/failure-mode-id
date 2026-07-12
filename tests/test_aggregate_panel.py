import json

import config
import pipeline.aggregate as aggregate
from pipeline import _experiment
from tests.conftest import make_cfg


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _isolate_aggregate_io(tmp_path, monkeypatch):
    """Redirect every real-filesystem touchpoint `aggregate.run()` has to
    tmp_path. `config.RESULTS_PATH`/`EXPERIMENTS_DIR`/`EXPERIMENT_INDEX_PATH`
    are bound into `aggregate`/`pipeline._experiment`'s namespaces via
    `from config import ...` at module-import time, so patching the `config`
    module's attribute alone doesn't redirect them — the consuming module's
    own binding has to be patched. `_sync_dashboard` shells out to
    scripts/dashboard_sync.py in a subprocess that always reads the repo's
    real `outputs/`/`dashboard/`, independent of any in-process monkeypatch,
    so it's stubbed out too (it's an explicitly best-effort, non-fatal
    convenience per aggregate.py's module docstring — not under test here)."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(aggregate, "RESULTS_PATH", tmp_path / "results.json")
    monkeypatch.setattr(_experiment, "EXPERIMENTS_DIR", tmp_path / "experiments")
    monkeypatch.setattr(_experiment, "EXPERIMENT_INDEX_PATH", tmp_path / "experiments" / "index.json")
    monkeypatch.setattr(aggregate, "_sync_dashboard", lambda mon: None)


def test_panel_block_and_default_view(tmp_path, monkeypatch):
    _isolate_aggregate_io(tmp_path, monkeypatch)
    data = tmp_path / "cc.jsonl"
    _write_jsonl(data, [{"id": "P1", "prompt": "p", "criteria": ["c1", "c2"],
                         "use_case": "u", "instruction_type": "it", "prompt_style": "ps"}])
    monkeypatch.setattr(aggregate, "DATA_JSONL", data)

    cfg = make_cfg(slug="E98-panel",
                   judges=("claude-fable-5", "claude-opus-4-8"),
                   candidates={"m1": "prov/m1"}, limit=None)
    _write_jsonl(cfg.responses_path("m1"), [{"id": "P1", "response": "ans"}])
    _write_jsonl(cfg.grades_path("claude-fable-5", "m1"), [{"id": "P1", "verdicts": [
        {"index": 1, "verdict": "PASS", "reason": ""},
        {"index": 2, "verdict": "FAIL", "reason": "missed"}]}])
    _write_jsonl(cfg.grades_path("claude-opus-4-8", "m1"), [{"id": "P1", "verdicts": [
        {"index": 1, "verdict": "PASS", "reason": ""},
        {"index": 2, "verdict": "FAIL", "reason": "judge_refusal: x"}]}])
    _write_jsonl(cfg.criteria_tags_path, [{"id": "P1", "tags": [
        {"index": 1, "verifiability": "auto", "gameable": False, "reward_hack": "", "ambiguous": False},
        {"index": 2, "verifiability": "judge", "gameable": False, "reward_hack": "", "ambiguous": False}]}])

    aggregate.run(cfg)
    doc = json.loads((tmp_path / "results.json").read_text())

    assert doc["schema_version"] == "3.3"
    assert doc["meta"]["verdict_basis"] == "panel"
    panel = doc["panel"]
    assert panel["judges"] == ["claude-fable-5", "claude-opus-4-8"]
    c1, c2 = panel["prompts"][0]["criteria"]
    assert c1["results"]["m1"]["votes"] == {"pass": 2, "fail": 0, "abstain": 0}
    # c2: fable FAIL (real vote), opus abstains -> 2-judge quorum=2 not met
    assert c2["results"]["m1"]["pass"] is False
    assert c2["results"]["m1"]["reason"] == "panel_no_quorum"
    assert panel["agreement"]["abstentions"]["claude-opus-4-8"] == 1
    # top-level view mirrors the panel
    assert doc["summary"] == panel["summary"]
    # by_judge views survive untouched, keyed by key, with 3.3 judge_details
    jd = doc["by_judge"]["claude-fable-5"]["judge_details"]
    assert jd["provider"] == "anthropic" and jd["family_overlap"] is False


def test_single_judge_keeps_first_judge_default(tmp_path, monkeypatch):
    _isolate_aggregate_io(tmp_path, monkeypatch)
    data = tmp_path / "cc.jsonl"
    _write_jsonl(data, [{"id": "P1", "prompt": "p", "criteria": ["c1"],
                         "use_case": "u", "instruction_type": "it", "prompt_style": "ps"}])
    monkeypatch.setattr(aggregate, "DATA_JSONL", data)

    cfg = make_cfg(slug="E98-solo", judges=("claude-fable-5",),
                   candidates={"m1": "prov/m1"}, limit=None)
    _write_jsonl(cfg.responses_path("m1"), [{"id": "P1", "response": "ans"}])
    _write_jsonl(cfg.grades_path("claude-fable-5", "m1"), [{"id": "P1", "verdicts": [
        {"index": 1, "verdict": "PASS", "reason": ""}]}])
    _write_jsonl(cfg.criteria_tags_path, [{"id": "P1", "tags": [
        {"index": 1, "verifiability": "auto", "gameable": False, "reward_hack": "",
         "ambiguous": False}]}])

    aggregate.run(cfg)
    doc = json.loads((tmp_path / "results.json").read_text())
    assert "panel" not in doc
    assert doc["meta"]["verdict_basis"] == "claude-fable-5"
    assert doc["summary"] == doc["by_judge"]["claude-fable-5"]["summary"]
