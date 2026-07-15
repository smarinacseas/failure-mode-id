"""E08 judge-panel policy (plan §0.3.3-5): majority consensus with an Opus
tie-break, an EXCLUDE verdict for genuinely-undecidable criteria (never FAIL),
and a completeness gate requiring >=2 non-abstaining verdicts per criterion.

These rules deliberately diverge from pipeline/_consensus.py, which resolves
ties and below-quorum cells to FAIL. The panel module keeps the census clean
at the source: a refusal is never a FAIL, and an undecidable criterion is
dropped, not counted against the model.
"""

import pytest

from judging import panel


def V(verdict, reason=""):
    return {"index": 1, "verdict": verdict, "reason": reason}


# --- consensus: clear cases -------------------------------------------------

def test_unanimous_pass_and_fail():
    p = panel.consensus({"opus": V("PASS"), "gem": V("PASS"), "gpt": V("PASS")},
                        n_judges=3, tiebreaker="opus")
    assert p["verdict"] == "PASS"
    assert p["votes"] == {"pass": 3, "fail": 0, "abstain": 0}

    f = panel.consensus({"opus": V("FAIL", "missed X"), "gem": V("FAIL", "missed X"),
                         "gpt": V("FAIL", "missed X")}, n_judges=3, tiebreaker="opus")
    assert f["verdict"] == "FAIL"
    assert f["reason"] == "missed X"


def test_clear_majority_carries_common_fail_reason():
    p = panel.consensus({"opus": V("PASS"), "gem": V("PASS"), "gpt": V("FAIL", "r")},
                        n_judges=3, tiebreaker="opus")
    assert p["verdict"] == "PASS"

    f = panel.consensus({"opus": V("FAIL", "count wrong"), "gem": V("FAIL", "count wrong"),
                         "gpt": V("PASS")}, n_judges=3, tiebreaker="opus")
    assert f["verdict"] == "FAIL"
    assert f["reason"] == "count wrong"        # most common FAIL-voter reason


# --- consensus: a refusal is never a FAIL (reuses vote_of) ------------------

def test_refusal_abstains_not_fails():
    # opus refuses; the two real PASS votes carry it, NOT a 2-1 with the refusal.
    p = panel.consensus({"opus": V("FAIL", "judge_refusal: declined"),
                         "gem": V("PASS"), "gpt": V("PASS")},
                        n_judges=3, tiebreaker="opus")
    assert p["verdict"] == "PASS"
    assert p["votes"] == {"pass": 2, "fail": 0, "abstain": 1}


# --- consensus: Opus tie-break (§0.3.3) -------------------------------------

def test_tiebreak_opus_present_breaks_to_pass():
    # gpt abstains → gem/opus split 1-1 → opus (tiebreaker) voted PASS → PASS.
    p = panel.consensus({"opus": V("PASS"), "gem": V("FAIL", "r"),
                         "gpt": V("FAIL", "judge_parse_error: x")},
                        n_judges=3, tiebreaker="opus")
    assert p["verdict"] == "PASS"
    assert p["tiebreak"] is True
    assert p["votes"] == {"pass": 1, "fail": 1, "abstain": 1}


def test_tiebreak_opus_present_breaks_to_fail_with_its_reason():
    p = panel.consensus({"opus": V("FAIL", "ordering violated"), "gem": V("PASS"),
                         "gpt": V("FAIL", "judge_truncated: y")},
                        n_judges=3, tiebreaker="opus")
    assert p["verdict"] == "FAIL"
    assert p["tiebreak"] is True
    assert p["reason"] == "ordering violated"   # the anchor's own reason


# --- consensus: EXCLUDE (§0.3.4) --------------------------------------------

def test_opus_abstains_and_rest_split_excludes():
    # opus refuses; gem/gpt split 1-1; no anchor to break it → EXCLUDE, not FAIL.
    p = panel.consensus({"opus": V("FAIL", "judge_refusal: declined"),
                         "gem": V("PASS"), "gpt": V("FAIL", "missed")},
                        n_judges=3, tiebreaker="opus")
    assert p["verdict"] == "EXCLUDE"
    assert p["reason"] == "panel_undecidable"
    assert p["votes"] == {"pass": 1, "fail": 1, "abstain": 1}


def test_below_quorum_excludes_incomplete():
    # only opus produced a valid verdict; the other two are missing/abstained.
    p = panel.consensus({"opus": V("PASS")}, n_judges=3, tiebreaker="opus")
    assert p["verdict"] == "EXCLUDE"
    assert p["reason"] == "panel_incomplete"
    assert p["votes"] == {"pass": 1, "fail": 0, "abstain": 2}

    # zero valid verdicts (all three refused) → also incomplete, never FAIL.
    z = panel.consensus({"opus": V("FAIL", "judge_refusal: a"),
                         "gem": V("FAIL", "judge_parse_error: b"),
                         "gpt": V("FAIL", "judge_truncated: c")},
                        n_judges=3, tiebreaker="opus")
    assert z["verdict"] == "EXCLUDE"
    assert z["reason"] == "panel_incomplete"


def test_no_tiebreak_flag_on_clean_verdicts():
    p = panel.consensus({"opus": V("PASS"), "gem": V("PASS"), "gpt": V("PASS")},
                        n_judges=3, tiebreaker="opus")
    assert p.get("tiebreak", False) is False


# --- completeness gate (§0.3.5) ---------------------------------------------

def C(prompt_id, index, model, per_judge):
    return {"prompt_id": prompt_id, "criterion_index": index,
            "model": model, "per_judge": per_judge}


def test_n_valid_and_is_complete():
    two = {"opus": V("PASS"), "gem": V("FAIL", "r")}
    assert panel.n_valid(two) == 2 and panel.is_complete(two) is True
    one = {"opus": V("PASS"), "gem": V("FAIL", "judge_refusal: x")}
    assert panel.n_valid(one) == 1 and panel.is_complete(one) is False


def test_completeness_report_flags_undercovered_cells():
    cells = [
        C("P1", 1, "llama-3b", {"opus": V("PASS"), "gem": V("PASS"), "gpt": V("FAIL", "r")}),
        # only 1 valid verdict (gem refused, gpt missing) → incomplete
        C("P1", 2, "llama-3b", {"opus": V("PASS"), "gem": V("FAIL", "judge_refusal: x")}),
    ]
    rep = panel.completeness_report(cells, judges=["opus", "gem", "gpt"])
    assert rep["complete"] is False
    assert rep["n_cells"] == 2 and rep["n_incomplete"] == 1
    bad = rep["incomplete"][0]
    assert (bad["prompt_id"], bad["criterion_index"], bad["model"]) == ("P1", 2, "llama-3b")
    assert bad["n_valid"] == 1
    # the judges needing a re-grade for this cell: gem (refused) + gpt (missing).
    assert sorted(bad["abstaining"]) == ["gem", "gpt"]


def test_dispatch_consensus_switches_on_tiebreaker():
    # a 1-1 tie (gem missing): legacy resolves to FAIL, panel policy lets the
    # anchor break it. This is the single switch aggregate/diagnose flip on.
    per_judge = {"opus": V("PASS"), "gpt": V("FAIL", "r")}
    legacy = panel.dispatch_consensus(None, per_judge, 3)
    assert legacy["verdict"] == "FAIL" and legacy["reason"] == "panel_tie"
    e08 = panel.dispatch_consensus("opus", per_judge, 3)
    assert e08["verdict"] == "PASS" and e08["tiebreak"] is True


def test_injected_malformed_judge_output_becomes_abstain_after_retries(monkeypatch):
    """Gate GE0: a deliberately-malformed judge response is classified ABSTAIN
    after retries — never a FAIL vote. Full path: grade._grade_one retries the
    unparseable output once, records judge_parse_error, and panel.consensus maps
    that reason to ABSTAIN so the two clean PASS votes carry the criterion."""
    from pipeline import grade

    calls = {"n": 0}

    def malformed_call(spec, system, user, label):
        calls["n"] += 1
        return "SORRY — I will not emit JSON. <<not a verdict array>>", "stop"
    monkeypatch.setattr(grade, "call_json", malformed_call)

    opus_spec = grade.JudgeSpec("opus", "openrouter", "anthropic/claude-opus-4.8")
    verdicts = grade._grade_one(opus_spec, "prompt", "response", ["c1"])
    assert calls["n"] == 2                                   # retried once before giving up
    assert verdicts[0]["reason"].startswith("judge_parse_error")

    p = panel.consensus({"opus": verdicts[0], "gem": V("PASS"), "gpt": V("PASS")},
                        n_judges=3, tiebreaker="opus")
    assert p["verdict"] == "PASS"
    assert p["votes"] == {"pass": 2, "fail": 0, "abstain": 1}


def test_completeness_report_all_complete():
    cells = [
        C("P1", 1, "llama-3b", {"opus": V("PASS"), "gem": V("PASS"), "gpt": V("PASS")}),
        C("P1", 2, "llama-3b", {"opus": V("FAIL", "r"), "gem": V("PASS")}),
    ]
    rep = panel.completeness_report(cells, judges=["opus", "gem", "gpt"])
    assert rep["complete"] is True
    assert rep["n_incomplete"] == 0 and rep["incomplete"] == []
