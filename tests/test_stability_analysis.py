"""Core-statistics tests for the E07-vs-E05 stability analysis."""
import json

import pytest

import config
import scripts.stability_analysis as sa
from scripts.stability_analysis import (
    bootstrap_delta_ci, cause_shares, cluster_bootstrap_shares, decision_rule,
    kendall_tau, rank_causes, top_k_set, wilson_interval,
)


def _rows(spec):
    """spec: list of (prompt_id, [root_causes]) -> failure_analysis-style rows."""
    return [
        {"id": pid, "root_cause": c, "model": "m", "criterion_index": i,
         "judge_concurrence": "both_fail"}
        for pid, causes in spec
        for i, c in enumerate(causes, 1)
    ]


def test_wilson_bounds():
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0)
    assert lo == pytest.approx(0.722, abs=0.005)
    lo0, _ = wilson_interval(0, 10)
    assert lo0 == pytest.approx(0.0)
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_rank_causes_breaks_ties_alphabetically():
    rows = _rows([("P1", ["b", "a", "a", "b", "c"])])
    assert rank_causes(rows) == ["a", "b", "c"]
    assert top_k_set(rows, 2) == {"a", "b"}


def test_cause_shares():
    rows = _rows([("P1", ["a", "a", "b"])])
    assert cause_shares(rows) == {"a": 2 / 3, "b": 1 / 3}
    assert cause_shares([]) == {}


def test_kendall_tau():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
    assert kendall_tau(["a", "b", "c", "d"], ["b", "a", "c"]) == pytest.approx(1 / 3)
    assert kendall_tau(["a", "x"], ["y", "z"]) is None


def test_bootstrap_deterministic_and_brackets_share():
    rows = _rows([(f"P{i:02d}", ["a", "a", "b"]) for i in range(10)])
    ci1 = cluster_bootstrap_shares(rows, n_boot=500, seed=1)
    ci2 = cluster_bootstrap_shares(rows, n_boot=500, seed=1)
    assert ci1 == ci2
    lo, hi = ci1["a"]
    assert lo <= 2 / 3 <= hi
    # identical clusters -> zero between-cluster variance -> degenerate CI
    assert hi - lo < 1e-9


def test_bootstrap_wider_when_cause_concentrated_in_one_cluster():
    concentrated = _rows(
        [(f"P{i}", ["a", "a"]) for i in range(7)] + [("P7", ["b", "b"])])
    spread = _rows(
        [(f"P{i}", ["a", "a"]) for i in range(6)]
        + [("P6", ["a", "b"]), ("P7", ["a", "b"])])
    ci_c = cluster_bootstrap_shares(concentrated, n_boot=2000, seed=2)["b"]
    ci_s = cluster_bootstrap_shares(spread, n_boot=2000, seed=2)["b"]
    assert (ci_c[1] - ci_c[0]) > (ci_s[1] - ci_s[0])


def test_delta_ci_contains_zero_for_identically_distributed_sets():
    rows = _rows([("P1", ["a"]), ("P2", ["b", "b"]), ("P3", ["a", "b"]),
                  ("P4", ["a", "a"]), ("P5", ["b"])])
    lo, hi = bootstrap_delta_ci(rows, rows, n_boot=500, seed=3)["b"]
    assert lo <= 0.0 <= hi


def test_decision_rule_confirmed_and_reshuffled():
    e05 = _rows([("P1", ["dropped"] * 5 + ["misread"] * 4),
                 ("P2", ["slip"] * 3 + ["other"])])
    e07_same = _rows([("Q1", ["dropped"] * 6 + ["misread"] * 4),
                      ("Q2", ["slip"] * 3 + ["other"])])
    ci05 = cluster_bootstrap_shares(e05, n_boot=300, seed=4)
    ci07 = cluster_bootstrap_shares(e07_same, n_boot=300, seed=5)
    d = decision_rule(e07_same, e05, ci07, ci05)
    assert d["top3_set_holds"] is True
    assert d["pareto_confirmed"] is True

    e07_flip = _rows([("Q1", ["slip"] * 6 + ["degenerate"] * 5),
                      ("Q2", ["other"] * 4 + ["dropped"])])
    d2 = decision_rule(e07_flip, e05, cluster_bootstrap_shares(e07_flip, 300, 6), ci05)
    assert d2["top3_set_holds"] is False
    assert d2["pareto_confirmed"] is False


def _write_experiment(exp_dir, slug, rows):
    (exp_dir / f"{slug}.json").write_text(
        json.dumps({"failure_analysis": {"rows": rows}}), encoding="utf-8")


def test_analyze_cuts_proxies_and_decision(tmp_path, monkeypatch):
    exp = tmp_path / "experiments"
    exp.mkdir()
    runs = tmp_path / "runs"
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", exp)
    monkeypatch.setattr(config, "RUNS_DIR", runs)
    monkeypatch.setattr(sa, "e05_sample_ids", lambda: {"P1", "P2"})

    e05_rows = _rows([("P1", ["a", "a", "b"]), ("P2", ["a", "b", "c"])])
    e07_rows = _rows([("P1", ["a", "a", "b"]), ("P2", ["a", "b", "c"]),
                      ("P3", ["a", "b"]), ("P4", ["a", "c"])])
    _write_experiment(exp, "E05-test", e05_rows)
    _write_experiment(exp, "E07-test", e07_rows)

    run7 = runs / "E07-test"
    run7.mkdir(parents=True)
    with open(run7 / "criteria_tags.jsonl", "w", encoding="utf-8") as f:
        for pid in ("P1", "P2", "P3", "P4"):
            f.write(json.dumps({"id": pid, "tags": [
                {"index": i, "verifiability": "auto"} for i in range(1, 4)
            ]}) + "\n")
    (run7 / "experiment.json").write_text(
        json.dumps({"params": {"candidates": {"m": "prov/m"}}}), encoding="utf-8")
    grades = run7 / "grades" / "claude-fable-5"
    grades.mkdir(parents=True)
    with open(grades / "m.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "P1", "verdicts": [
            {"index": 1, "verdict": "FAIL",
             "reason": "judge_refusal: model declined to grade this cell"}
        ]}) + "\n")

    result = sa.analyze("E05-test", "E07-test", n_boot=200, seed=1)

    assert result["cuts"]["e05_full"]["n_prompts"] == 2
    assert result["cuts"]["e07_full"]["n_prompts"] == 4
    assert result["cuts"]["e07_shared"]["n_prompts"] == 2
    assert result["cuts"]["e07_disjoint"]["n_prompts"] == 2
    assert result["cuts"]["e07_full"]["ranking"][0] == "a"
    assert set(result["decision"]) >= {"top3_set_holds", "pareto_confirmed"}
    assert result["rank_stability"]["kendall_tau_e07full_vs_e05"] == 1.0
    assert result["proxies"]["fable_refusals"]["n_cells"] == 1
    assert result["proxies"]["auto_verifiable"]["n_rows"] == len(e07_rows)
    assert result["proxies"]["concurrence_both_fail"]["top3_matches_full"] is True
    assert "delta_e07full_minus_e05" in result
