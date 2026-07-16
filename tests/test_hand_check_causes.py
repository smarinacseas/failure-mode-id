"""Unit tests for the blinded root-cause hand-check tool.

All fixtures are synthetic — these tests never read real runs/ or the real
dashboard JSON. A mini dashboard dict and a matching mini responses.jsonl
exercise: deterministic sampling, stratum counts, the pool-size drift guard,
the integrity cross-check, worksheet blinding, the string->int / 1-based
criterion join, overwrite refusal, vocabulary rejection, and the score math
(confusion matrix incl. the A<->B cell).
"""
from __future__ import annotations

import json
from collections import Counter

import pytest

import scripts.hand_check_causes as hc


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

# Sentinels planted in the FORBIDDEN row fields. None of these may ever appear
# in the worksheet body (blinding). Chosen so they cannot collide with any
# prompt/criterion/response text below.
FORBIDDEN_SENTINELS = {
    "secondary": "ZZSECONDARYZZ",
    "confidence": "ZZCONFIDENCEZZ",
    "evidence": "ZZEVIDENCEZZ",
    "rationale": "ZZRATIONALEZZ",
    "analyst": "ZZANALYSTZZ",
    "judge_concurrence": "ZZCONCURZZ",
}

TAXONOMY = [
    {"key": "constraint_never_surfaced"},
    {"key": "constraint_misread"},
    {"key": "input_misread"},
    {"key": "execution_slip"},
    {"key": "degenerate_output"},
    {"key": "constraint_unaddressed"},
    {"key": "judge_suspect"},
    {"key": "other"},
]

# pool -> how many synthetic rows to mint (all >= the sample strata so the
# fixed-seed sample is a genuine subset, not a degenerate full-pool shuffle).
FIXTURE_POOLS = {
    "constraint_unaddressed": 20,
    "execution_slip": 20,
    "constraint_misread": 15,
    "degenerate_output": 5,  # a 4th pool to prove the sampler filters it out
}

FIXTURE_EXPECTED = {
    "execution_slip": 20,
    "constraint_unaddressed": 20,
    "constraint_misread": 15,
}


def make_dashboard():
    """Mini dashboard: one prompt per row, 8 criteria each, unique response."""
    rows = []
    prompts = []
    n = 0
    for cause, count in FIXTURE_POOLS.items():
        for _ in range(count):
            pid = f"P{n:03d}"
            # 8 criteria; the 3rd (1-based) carries a distinctive marker so the
            # 1-based join can be pinned down.
            criteria = [{"text": f"CRIT-{pid}-c{j}"} for j in range(1, 9)]
            prompts.append({
                "id": pid,
                "prompt_text": f"PROMPT-TEXT-{pid} please schedule the shifts.",
                "responses": {"llama-3b": f"RESPONSE-BODY-{pid} here is the plan."},
                "criteria": criteria,
            })
            rows.append({
                "id": pid,
                "model": "llama-3b",
                "criterion_index": "3",  # STRING on purpose -> tests str->int
                "root_cause": cause,
                "secondary": FORBIDDEN_SENTINELS["secondary"],
                "confidence": FORBIDDEN_SENTINELS["confidence"],
                "evidence": FORBIDDEN_SENTINELS["evidence"],
                "rationale": FORBIDDEN_SENTINELS["rationale"],
                "trace_status": "no_trace",
                "analyst": FORBIDDEN_SENTINELS["analyst"],
                "judge_concurrence": FORBIDDEN_SENTINELS["judge_concurrence"],
            })
            n += 1
    return {
        "prompts": prompts,
        "failure_analysis": {"taxonomy": TAXONOMY, "rows": rows},
    }


def make_responses(dashboard):
    """responses.jsonl content matching the dashboard, as a list of dicts."""
    return [
        {"id": p["id"], "response": p["responses"]["llama-3b"]}
        for p in dashboard["prompts"]
    ]


# ---------------------------------------------------------------------------
# Guard: pool-size / schema-drift
# ---------------------------------------------------------------------------

def test_default_expected_pool_sizes_match_contract():
    # The stamped contract numbers for the real 882-row run.
    assert hc.EXPECTED_POOL_SIZES == {
        "execution_slip": 321,
        "constraint_unaddressed": 254,
        "constraint_misread": 127,
    }


def test_pool_size_guard_passes_on_matching_counts():
    rows = hc.get_rows(make_dashboard())
    hc.assert_pool_sizes(rows, expected=FIXTURE_EXPECTED)  # must not raise


def test_pool_size_guard_aborts_on_drift():
    rows = hc.get_rows(make_dashboard())
    drifted = dict(FIXTURE_EXPECTED, execution_slip=999)
    with pytest.raises(hc.HandCheckError):
        hc.assert_pool_sizes(rows, expected=drifted)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_stratified_sample_counts():
    rows = hc.get_rows(make_dashboard())
    sample = hc.stratified_sample(rows, seed=20260715)
    assert len(sample) == 40
    counts = Counter(r["root_cause"] for r in sample)
    assert counts == {
        "constraint_unaddressed": 15,
        "execution_slip": 15,
        "constraint_misread": 10,
    }


def test_stratified_sample_is_deterministic():
    rows = hc.get_rows(make_dashboard())
    a = hc.stratified_sample(rows, seed=20260715)
    b = hc.stratified_sample(rows, seed=20260715)
    key = lambda s: [(r["id"], r["criterion_index"]) for r in s]
    assert key(a) == key(b)


def test_stratified_sample_differs_by_seed():
    rows = hc.get_rows(make_dashboard())
    a = hc.stratified_sample(rows, seed=1)
    b = hc.stratified_sample(rows, seed=2)
    key = lambda s: [(r["id"], r["criterion_index"]) for r in s]
    assert key(a) != key(b)


# ---------------------------------------------------------------------------
# Criterion join: string index, 1-based
# ---------------------------------------------------------------------------

def test_criterion_join_is_string_to_int_and_one_based():
    dash = make_dashboard()
    pidx = hc.prompts_index(dash)
    row = hc.get_rows(dash)[0]  # criterion_index == "3"
    text = hc.criterion_text_for(row, pidx)
    # 1-based: index "3" -> criteria[2] -> "...-c3"
    assert text.endswith("-c3")
    assert "-c4" not in text  # would be the 0-based (wrong) reading


# ---------------------------------------------------------------------------
# Integrity cross-check
# ---------------------------------------------------------------------------

def test_integrity_check_passes_on_matching_fixtures():
    dash = make_dashboard()
    pidx = hc.prompts_index(dash)
    responses = {r["id"]: r["response"] for r in make_responses(dash)}
    frame = hc.frame_rows(hc.get_rows(dash))
    ids = hc.integrity_check_ids(frame, seed=20260715)
    assert len(ids) == 3
    hc.run_integrity_check(pidx, responses, ids)  # must not raise


def test_integrity_check_aborts_on_mutated_response():
    dash = make_dashboard()
    pidx = hc.prompts_index(dash)
    responses = {r["id"]: r["response"] for r in make_responses(dash)}
    # Corrupt every response so any checked id mismatches.
    responses = {k: v + " TAMPERED" for k, v in responses.items()}
    frame = hc.frame_rows(hc.get_rows(dash))
    ids = hc.integrity_check_ids(frame, seed=20260715)
    with pytest.raises(hc.HandCheckError):
        hc.run_integrity_check(pidx, responses, ids)


def test_integrity_check_aborts_on_missing_id():
    dash = make_dashboard()
    pidx = hc.prompts_index(dash)
    frame = hc.frame_rows(hc.get_rows(dash))
    ids = hc.integrity_check_ids(frame, seed=20260715)
    with pytest.raises(hc.HandCheckError):
        hc.run_integrity_check(pidx, {}, ids)  # empty responses -> missing


# ---------------------------------------------------------------------------
# Worksheet: blinding + build
# ---------------------------------------------------------------------------

def _build(tmp_path, seed=20260715, rule="RULE-TEXT-VERBATIM"):
    dash = make_dashboard()
    responses = {r["id"]: r["response"] for r in make_responses(dash)}
    out = tmp_path / "hand_check"
    hc.build_worksheet(
        dash, responses, rule=rule, seed=seed, out_dir=out,
        expected_pool_sizes=FIXTURE_EXPECTED,
        timestamp="2026-07-15T00:00:00+00:00",
    )
    return out


def test_worksheet_body_contains_no_forbidden_field_values(tmp_path):
    out = _build(tmp_path)
    text = (out / "worksheet.md").read_text(encoding="utf-8")
    assert hc.BODY_DELIMITER in text
    body = text.split(hc.BODY_DELIMITER, 1)[1]
    for sentinel in FORBIDDEN_SENTINELS.values():
        assert sentinel not in body, f"{sentinel} leaked into worksheet body"
    # The per-row root_cause answer must also never appear in the body.
    key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    for info in key["rows"].values():
        assert info["root_cause"] not in body


def test_worksheet_header_stamps_rule_and_vocab(tmp_path):
    out = _build(tmp_path, rule="DECIDE-LIKE-THIS")
    text = (out / "worksheet.md").read_text(encoding="utf-8")
    header = text.split(hc.BODY_DELIMITER, 1)[0]
    assert "DECIDE-LIKE-THIS" in header
    assert "20260715" in header
    for key in [t["key"] for t in TAXONOMY]:
        assert key in header  # full permitted vocabulary listed


def test_worksheet_has_forty_rows_and_blank_labels(tmp_path):
    out = _build(tmp_path)
    text = (out / "worksheet.md").read_text(encoding="utf-8")
    body = text.split(hc.BODY_DELIMITER, 1)[1]
    assert body.count("human_label:") == 40


def test_answer_key_shape(tmp_path):
    out = _build(tmp_path, seed=20260715, rule="R")
    key = json.loads((out / "answer_key.json").read_text(encoding="utf-8"))
    assert key["rule"] == "R"
    assert key["seed"] == 20260715
    assert key["timestamp"] == "2026-07-15T00:00:00+00:00"
    assert len(key["rows"]) == 40
    assert set(key["rows"].keys()) == {str(i) for i in range(1, 41)}
    sample_row = key["rows"]["1"]
    assert set(sample_row) == {"id", "criterion_index", "root_cause"}


def test_build_refuses_to_overwrite(tmp_path):
    out = _build(tmp_path)
    dash = make_dashboard()
    responses = {r["id"]: r["response"] for r in make_responses(dash)}
    with pytest.raises(hc.HandCheckError):
        hc.build_worksheet(
            dash, responses, rule="again", seed=1, out_dir=out,
            expected_pool_sizes=FIXTURE_EXPECTED,
            timestamp="2026-07-15T00:00:00+00:00",
        )


# ---------------------------------------------------------------------------
# Score: parsing + math
# ---------------------------------------------------------------------------

VOCAB = [t["key"] for t in TAXONOMY]


def _worksheet_from_labels(labels):
    """Minimal worksheet text with given per-row human_label values."""
    lines = ["header stuff", hc.BODY_DELIMITER]
    for i, lab in enumerate(labels, start=1):
        lines += [f"## Row {i}", "### Response", f"body-{i}", f"human_label: {lab}", ""]
    return "\n".join(lines)


def test_parse_human_labels_ok():
    text = _worksheet_from_labels(["execution_slip", "other"])
    got = hc.parse_human_labels(text, VOCAB)
    assert got == {1: "execution_slip", 2: "other"}


def test_parse_rejects_out_of_vocab():
    text = _worksheet_from_labels(["execution_slip", "banana"])
    with pytest.raises(hc.HandCheckError):
        hc.parse_human_labels(text, VOCAB)


def test_parse_rejects_blanks():
    text = _worksheet_from_labels(["execution_slip", ""])
    with pytest.raises(hc.HandCheckError):
        hc.parse_human_labels(text, VOCAB)


def _hand_answer_key():
    cu, es, cm = "constraint_unaddressed", "execution_slip", "constraint_misread"
    classifiers = [cu, cu, es, es, cm, cm]
    return {
        "rule": "R", "seed": 1, "timestamp": "T", "vocab": VOCAB,
        "rows": {
            str(i + 1): {"id": f"P{i}", "criterion_index": 3, "root_cause": c}
            for i, c in enumerate(classifiers)
        },
    }


def test_compute_scores_confusion_and_ab_cell():
    cu, es, cm = "constraint_unaddressed", "execution_slip", "constraint_misread"
    key = _hand_answer_key()
    # human labels:  agree, A->B, agree, B->A, agree, cm->other
    human = {1: cu, 2: es, 3: es, 4: cu, 5: cm, 6: "other"}
    s = hc.compute_scores(key, human)

    assert s["n"] == 6
    assert s["overall_agreement"] == pytest.approx(3 / 6)

    assert s["per_cause"][cu]["n"] == 2
    assert s["per_cause"][cu]["agree"] == 1
    assert s["per_cause"][es]["agree"] == 1
    assert s["per_cause"][cm]["agree"] == 1

    # A<->B cell: classifier=cu/human=es (row2) and classifier=es/human=cu (row4)
    assert s["ab_cell"]["classifier_constraint_unaddressed__human_execution_slip"] == 1
    assert s["ab_cell"]["classifier_execution_slip__human_constraint_unaddressed"] == 1
    assert s["ab_cell"]["total"] == 2

    # confusion matrix keyed (human, classifier)
    conf = s["confusion"]
    assert conf[(cu, cu)] == 1
    assert conf[(es, cu)] == 1
    assert conf[(es, es)] == 1
    assert conf[(cu, es)] == 1
    assert conf[(cm, cm)] == 1
    assert conf[("other", cm)] == 1

    dis_rows = {d["row"] for d in s["disagreements"]}
    assert dis_rows == {2, 4, 6}


def test_score_dir_end_to_end(tmp_path):
    out = tmp_path / "hand_check"
    out.mkdir()
    key = _hand_answer_key()
    (out / "answer_key.json").write_text(json.dumps(key), encoding="utf-8")
    cu, es, cm = "constraint_unaddressed", "execution_slip", "constraint_misread"
    labels = [cu, es, es, cu, cm, "other"]
    (out / "worksheet.md").write_text(_worksheet_from_labels(labels), encoding="utf-8")

    summary = hc.score_dir(out)
    assert summary["overall_agreement"] == pytest.approx(3 / 6)
    written = json.loads((out / "score_summary.json").read_text(encoding="utf-8"))
    assert written["n"] == 6
    assert written["rule"] == "R"
