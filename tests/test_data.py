import json

from training.data import load_verih, to_sample


def test_to_sample_builds_system_user_messages_and_constraints():
    # Real VerIH row shape (skai-research/VerIH, RLVR/dataset/verih/train.json):
    # sys_prompt/user_prompt strings + gt = JSON-encoded verifier spec.
    row = {
        "sys_prompt": "Do not include keywords anticor, frolicness, necessarianism in the response.",
        "user_prompt": "Translate this to English?",
        "explain": "",
        "raw_messages": [{"role": "user", "content": "..."}],
        "gt": json.dumps({
            "func_name": "validate_forbidden_words",
            "forbidden_words": ["anticor", "frolicness", "necessarianism"],
        }),
        "type": "forbidden words:aligned",
    }
    s = to_sample(row)
    assert s["messages"][0] == {
        "role": "system",
        "content": "Do not include keywords anticor, frolicness, necessarianism in the response.",
    }
    assert s["messages"][1] == {"role": "user", "content": "Translate this to English?"}
    assert s["constraints"] == [{
        "func_name": "validate_forbidden_words",
        "forbidden_words": ["anticor", "frolicness", "necessarianism"],
    }]
    assert s["type"] == "forbidden words:aligned"


def test_to_sample_omits_system_message_when_sys_prompt_blank():
    row = {"sys_prompt": "", "user_prompt": "hi", "gt": json.dumps({"func_name": "validate_no_commas"})}
    s = to_sample(row)
    assert s["messages"] == [{"role": "user", "content": "hi"}]


def test_to_sample_preserves_conflict_tag_and_defaults_to_none():
    conflict_row = {
        "sys_prompt": "", "user_prompt": "hi",
        "gt": json.dumps({"func_name": "validate_no_commas"}),
        "type": "conflict",
    }
    assert to_sample(conflict_row)["type"] == "conflict"

    row_without_type = {"sys_prompt": "", "user_prompt": "hi", "gt": json.dumps({"func_name": "validate_no_commas"})}
    assert to_sample(row_without_type)["type"] is None


def test_load_verih_reads_json_array_and_applies_limit(tmp_path):
    rows = [
        {"sys_prompt": "", "user_prompt": f"q{i}", "gt": json.dumps({"func_name": "validate_no_commas"})}
        for i in range(3)
    ]
    p = tmp_path / "train.json"
    p.write_text(json.dumps(rows), encoding="utf-8")

    all_samples = load_verih(str(p))
    assert len(all_samples) == 3
    assert all_samples[1]["messages"] == [{"role": "user", "content": "q1"}]

    limited = load_verih(str(p), limit=2)
    assert len(limited) == 2


def test_load_verih_reads_jsonl(tmp_path):
    p = tmp_path / "train.jsonl"
    lines = [
        json.dumps({"sys_prompt": "", "user_prompt": "a", "gt": json.dumps({"func_name": "validate_no_commas"})}),
        json.dumps({"sys_prompt": "", "user_prompt": "b", "gt": json.dumps({"func_name": "validate_no_commas"})}),
    ]
    p.write_text("\n".join(lines), encoding="utf-8")

    samples = load_verih(str(p))
    assert [s["messages"][0]["content"] for s in samples] == ["a", "b"]
