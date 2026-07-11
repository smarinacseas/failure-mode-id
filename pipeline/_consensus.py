"""Panel consensus + agreement math (derived, never stored in grades/).

Shared by aggregate.py and diagnose.py so both use identical vote rules
(spec §3): grading-artifact FAILs abstain; loop_failure all-FAILs are genuine
FAIL votes; quorum = min(2, n_judges); ties and below-quorum cells resolve to
FAIL with a distinguishing reason (missing evidence = FAIL, never dropped).
"""

from __future__ import annotations

from collections import Counter

ARTIFACT_PREFIXES = ("judge_refusal", "judge_parse_error", "judge_truncated",
                     "missing_in_judge_output")


def vote_of(verdict_rec: dict) -> str:
    reason = str(verdict_rec.get("reason", ""))
    if reason.startswith(ARTIFACT_PREFIXES):
        return "ABSTAIN"
    return "PASS" if verdict_rec.get("verdict") == "PASS" else "FAIL"


def consensus_verdict(per_judge: dict[str, dict], n_judges: int) -> dict:
    """Majority vote for ONE criterion. per_judge: judge_key -> verdict record;
    judges missing from the map abstain (no grade record = no opinion)."""
    votes = {j: vote_of(v) for j, v in per_judge.items()}
    cast = {j: v for j, v in votes.items() if v != "ABSTAIN"}
    n_pass = sum(1 for v in cast.values() if v == "PASS")
    n_fail = len(cast) - n_pass
    split = {"pass": n_pass, "fail": n_fail, "abstain": n_judges - len(cast)}
    quorum = min(2, n_judges)
    if len(cast) < quorum:
        return {"verdict": "FAIL", "reason": "panel_no_quorum", "votes": split}
    if n_pass == n_fail:
        return {"verdict": "FAIL", "reason": "panel_tie", "votes": split}
    if n_pass > n_fail:
        return {"verdict": "PASS", "reason": "", "votes": split}
    reasons = Counter(str(per_judge[j].get("reason", ""))
                      for j, v in cast.items() if v == "FAIL")
    return {"verdict": "FAIL", "reason": reasons.most_common(1)[0][0], "votes": split}


def _pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 2) if den else None


def agreement_stats(cells: list[dict[str, str]], judges: list[str]) -> dict:
    """cells: one judge_key->vote map per (candidate, prompt, criterion)."""
    pairwise: dict[str, dict[str, float | None]] = {}
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            both = [(c[a], c[b]) for c in cells
                    if c.get(a) in ("PASS", "FAIL") and c.get(b) in ("PASS", "FAIL")]
            pct = _pct(sum(1 for x, y in both if x == y), len(both))
            pairwise.setdefault(a, {})[b] = pct
            pairwise.setdefault(b, {})[a] = pct

    abstentions = {j: sum(1 for c in cells if c.get(j, "ABSTAIN") == "ABSTAIN")
                   for j in judges}

    # Fleiss' kappa, variable-rater generalization: only cells with >=2 cast
    # votes; P_i = sum_j n_ij(n_ij-1) / (n_i(n_i-1)); P_e from overall vote mix.
    eligible = []
    for c in cells:
        cast = [v for v in c.values() if v in ("PASS", "FAIL")]
        if len(cast) >= 2:
            eligible.append(cast)
    kappa = None
    if eligible:
        total = Counter(v for cast in eligible for v in cast)
        n_votes = sum(total.values())
        p_e = sum((n / n_votes) ** 2 for n in total.values())
        p_bar = sum(
            sum(n * (n - 1) for n in Counter(cast).values()) / (len(cast) * (len(cast) - 1))
            for cast in eligible
        ) / len(eligible)
        kappa = round((p_bar - p_e) / (1 - p_e), 4) if p_e < 1 else None

    with_consensus: dict[str, float | None] = {}
    decided = []
    for c in cells:
        cv = consensus_verdict({j: {"verdict": v, "reason": ""} for j, v in c.items()
                                if v in ("PASS", "FAIL")}, len(judges))
        if cv["reason"] in ("", ) or (cv["verdict"] == "FAIL" and cv["reason"] not in
                                      ("panel_tie", "panel_no_quorum")):
            decided.append((c, cv["verdict"]))
    for j in judges:
        voted = [(c[j], verdict) for c, verdict in decided if c.get(j) in ("PASS", "FAIL")]
        with_consensus[j] = _pct(sum(1 for v, cv in voted if v == cv), len(voted))

    return {"pairwise": pairwise, "fleiss_kappa": kappa,
            "with_consensus": with_consensus, "abstentions": abstentions}
