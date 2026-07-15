"""E08 judge-panel policy (plan §0.3.3-5).

Majority consensus over a diverse three-provider panel, with three rules that
deliberately diverge from pipeline/_consensus.py (which resolves ties and
below-quorum cells to FAIL):

  1. A refusal / malformed grade is ABSTAIN, never a FAIL vote. (Reused wholesale
     from _consensus.vote_of — the grade layer already tags artifact reasons.)
  2. A 1-1 tie among the non-abstaining judges is broken by a designated anchor
     judge (Opus 4.8). If the anchor itself abstained, the criterion is EXCLUDED
     from consensus metrics — it is NOT counted against the model as a FAIL.
  3. Every criterion needs >=2 non-abstaining verdicts (the completeness gate).
     Cells below that are EXCLUDE/panel_incomplete and drive targeted re-grades;
     completeness_report() lists them and names the judges to re-run.

EXCLUDE is a third disposition alongside PASS/FAIL: excluded criteria are dropped
from pass-rate denominators and never fed to the cause classifier. This keeps
the census clean at the source rather than laundering judge-pipeline noise and
genuine rubric ambiguity into phantom failures (the E07 contamination §0.3 fixes).
"""

from __future__ import annotations

from collections import Counter

from pipeline._consensus import consensus_verdict, vote_of

# A criterion needs at least this many non-abstaining verdicts to be scored
# (plan §0.3.5). Below it, the cell is incomplete and gets re-graded.
QUORUM = 2

EXCLUDE = "EXCLUDE"


def n_valid(per_judge: dict[str, dict]) -> int:
    """Number of judges that cast a real PASS/FAIL vote (abstentions excluded)."""
    return sum(1 for rec in per_judge.values() if vote_of(rec) != "ABSTAIN")


def is_complete(per_judge: dict[str, dict]) -> bool:
    """True once the criterion has the >=2 non-abstaining verdicts §0.3.5 requires."""
    return n_valid(per_judge) >= QUORUM


def consensus(per_judge: dict[str, dict], n_judges: int, tiebreaker: str) -> dict:
    """Panel verdict for ONE (prompt, criterion, model) cell.

    `per_judge`: judge_key -> that judge's verdict record ({index, verdict,
    reason}); a judge absent from the map abstains (no record = no opinion).
    `n_judges`: full panel size (for the abstain tally). `tiebreaker`: the
    anchor judge's key (Opus).

    Returns {verdict: PASS|FAIL|EXCLUDE, reason, votes:{pass,fail,abstain},
    tiebreak: bool}. FAIL carries the most-common FAIL-voter reason (or, on a
    tie-break, the anchor's own reason) so the cause classifier sees a real
    explanation. EXCLUDE reasons: panel_incomplete (<2 valid) / panel_undecidable
    (tie the anchor could not break).
    """
    votes = {j: vote_of(rec) for j, rec in per_judge.items()}
    cast = {j: v for j, v in votes.items() if v != "ABSTAIN"}
    n_pass = sum(1 for v in cast.values() if v == "PASS")
    n_fail = len(cast) - n_pass
    split = {"pass": n_pass, "fail": n_fail, "abstain": n_judges - len(cast)}

    if len(cast) < QUORUM:
        return {"verdict": EXCLUDE, "reason": "panel_incomplete",
                "votes": split, "tiebreak": False}

    if n_pass != n_fail:
        if n_pass > n_fail:
            return {"verdict": "PASS", "reason": "", "votes": split, "tiebreak": False}
        reasons = Counter(str(per_judge[j].get("reason", ""))
                          for j, v in cast.items() if v == "FAIL")
        return {"verdict": "FAIL", "reason": reasons.most_common(1)[0][0],
                "votes": split, "tiebreak": False}

    # Tie among the non-abstaining judges (only possible with an even cast count).
    # The anchor breaks it with its own vote; if the anchor abstained, exclude.
    anchor = cast.get(tiebreaker)
    if anchor is None:
        return {"verdict": EXCLUDE, "reason": "panel_undecidable",
                "votes": split, "tiebreak": False}
    if anchor == "PASS":
        return {"verdict": "PASS", "reason": "", "votes": split, "tiebreak": True}
    reason = str(per_judge[tiebreaker].get("reason", "")) or "panel_tiebreak_fail"
    return {"verdict": "FAIL", "reason": reason, "votes": split, "tiebreak": True}


def dispatch_consensus(tiebreaker_judge: str | None, per_judge: dict[str, dict],
                       n_judges: int) -> dict:
    """The consensus policy for a run: the E08 panel policy (Opus tie-break +
    EXCLUDE) when a tiebreaker anchor is configured, else legacy majority
    (ties / below-quorum → FAIL; E01-E07 behavior, byte-identical). This is the
    single switch that aggregate and diagnose flip on cfg.tiebreaker_judge, so
    both stages agree on how a criterion resolves.
    """
    if tiebreaker_judge:
        return consensus(per_judge, n_judges, tiebreaker_judge)
    return consensus_verdict(per_judge, n_judges)


def completeness_report(cells, judges: list[str]) -> dict:
    """The completeness gate (§0.3.5). `cells`: iterable of
    {prompt_id, criterion_index, model, per_judge}. `judges`: full panel keys.

    Returns {complete, n_cells, n_incomplete, incomplete:[...]} where each
    incomplete entry names the cell and the judges to re-grade (those that
    abstained OR never produced a record) — the targeted-re-run worklist.
    """
    incomplete: list[dict] = []
    n_cells = 0
    for cell in cells:
        n_cells += 1
        per_judge = cell["per_judge"]
        if is_complete(per_judge):
            continue
        votes = {j: vote_of(rec) for j, rec in per_judge.items()}
        abstaining = [j for j in judges if votes.get(j, "ABSTAIN") == "ABSTAIN"]
        incomplete.append({
            "prompt_id": cell["prompt_id"],
            "criterion_index": cell["criterion_index"],
            "model": cell["model"],
            "n_valid": n_valid(per_judge),
            "abstaining": abstaining,
        })
    return {
        "complete": not incomplete,
        "n_cells": n_cells,
        "n_incomplete": len(incomplete),
        "incomplete": incomplete,
    }
