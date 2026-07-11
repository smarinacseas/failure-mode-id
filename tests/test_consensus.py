import pytest

from pipeline._consensus import (
    agreement_stats, consensus_verdict, vote_of,
)


def V(verdict, reason=""):
    return {"index": 1, "verdict": verdict, "reason": reason}


def test_vote_of_artifacts_abstain_loops_vote():
    assert vote_of(V("FAIL", "judge_refusal: x")) == "ABSTAIN"
    assert vote_of(V("FAIL", "judge_parse_error: x")) == "ABSTAIN"
    assert vote_of(V("FAIL", "judge_truncated: x")) == "ABSTAIN"
    assert vote_of(V("FAIL", "missing_in_judge_output")) == "ABSTAIN"
    assert vote_of(V("FAIL", "loop_failure: no answer")) == "FAIL"   # real failure, real vote
    assert vote_of(V("PASS")) == "PASS"


def test_majority_pass_and_fail():
    c = consensus_verdict({"a": V("PASS"), "b": V("PASS"), "c": V("FAIL", "missed X")}, 3)
    assert (c["verdict"], c["reason"], c["votes"]) == ("PASS", "", {"pass": 2, "fail": 1, "abstain": 0})
    c2 = consensus_verdict({"a": V("FAIL", "missed X"), "b": V("FAIL", "missed X"),
                            "c": V("PASS")}, 3)
    assert c2["verdict"] == "FAIL"
    assert c2["reason"] == "missed X"      # most common FAIL-voter reason survives


def test_tie_and_quorum_resolve_to_fail():
    tie = consensus_verdict({"a": V("PASS"), "b": V("FAIL", "r")}, 2)
    assert (tie["verdict"], tie["reason"]) == ("FAIL", "panel_tie")
    nq = consensus_verdict({"a": V("PASS"), "b": V("FAIL", "judge_refusal: x"),
                            "c": V("FAIL", "judge_parse_error: y")}, 3)
    assert (nq["verdict"], nq["reason"]) == ("FAIL", "panel_no_quorum")
    assert nq["votes"] == {"pass": 1, "fail": 0, "abstain": 2}


def test_missing_judge_counts_as_abstain():
    c = consensus_verdict({"a": V("PASS"), "b": V("PASS")}, 5)
    assert c["votes"]["abstain"] == 3 and c["verdict"] == "PASS"


def test_single_judge_run_quorum_is_one():
    # quorum = min(2, n_judges): a lone real verdict stands; a lone artifact doesn't.
    assert consensus_verdict({"a": V("FAIL", "missed")}, 1)["verdict"] == "FAIL"
    assert consensus_verdict({"a": V("FAIL", "judge_refusal: x")}, 1)["reason"] == "panel_no_quorum"


def test_agreement_stats_hand_computed():
    cells = [
        {"a": "PASS", "b": "PASS"},
        {"a": "PASS", "b": "FAIL"},
        {"a": "FAIL", "b": "FAIL"},
        {"a": "PASS", "b": "ABSTAIN"},
    ]
    s = agreement_stats(cells, ["a", "b"])
    assert s["pairwise"]["a"]["b"] == pytest.approx(66.67, abs=0.01)  # 2/3 co-voted cells
    assert s["abstentions"] == {"a": 0, "b": 1}
    # Fleiss (variable-rater form) over the 3 cells with >=2 votes:
    # P_i per cell: 1, 0, 1 -> P_bar = 2/3; votes: 3 PASS, 3 FAIL -> P_e = 0.5
    # (implementation rounds to 4 decimals, hence the loose tolerance)
    assert s["fleiss_kappa"] == pytest.approx((2/3 - 0.5) / 0.5, abs=1e-4)
    # with_consensus counts only quorum'd, untied cells
    assert s["with_consensus"]["a"] == pytest.approx(100.0)


def test_agreement_stats_degenerate_returns_none():
    s = agreement_stats([{"a": "PASS"}], ["a", "b"])
    assert s["fleiss_kappa"] is None and s["pairwise"]["a"]["b"] is None
