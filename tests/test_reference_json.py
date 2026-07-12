"""dashboard/reference.json — schema + taxonomy-parity checks (spec 2026-07-09 §3)."""
import json
from pathlib import Path

import pytest

from pipeline import _taxonomy

REF_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "reference.json"
TRAINABLE_KEYS = {c["key"] for c in _taxonomy.DERIVED}
DIAGNOSTIC_KEYS = {_taxonomy.COLLAPSED["key"]} | {c["key"] for c in _taxonomy.RESERVED}
BY_KEY = {c["key"]: c
          for c in _taxonomy.DERIVED + [_taxonomy.COLLAPSED] + _taxonomy.RESERVED}


@pytest.fixture(scope="module")
def ref():
    assert REF_PATH.exists(), "dashboard/reference.json missing"
    return json.loads(REF_PATH.read_text())


def test_version_and_stamp(ref):
    assert ref["taxonomy_version"] == _taxonomy.TAXONOMY_VERSION
    assert ref["updated_at"]


def test_covers_every_taxonomy_key_exactly_once(ref):
    keys = [m["key"] for m in ref["modes"]]
    assert sorted(keys) == sorted(TRAINABLE_KEYS | DIAGNOSTIC_KEYS)


def test_kind_partition(ref):
    for m in ref["modes"]:
        expected = "trainable" if m["key"] in TRAINABLE_KEYS else "diagnostic"
        assert m["kind"] == expected, m["key"]


def test_labels_and_descriptions_match_pipeline(ref):
    for m in ref["modes"]:
        assert m["label"] == BY_KEY[m["key"]]["label"], m["key"]
        assert m["description"] == BY_KEY[m["key"]]["description"], m["key"]


def test_trainable_dossiers_complete(ref):
    for m in (x for x in ref["modes"] if x["kind"] == "trainable"):
        words = len(" ".join(m["strategy"]).split())
        assert 250 <= words <= 600, f"{m['key']}: strategy is {words} words"
        assert m["dataset_taxonomy"], m["key"]
        for d in m["dataset_taxonomy"]:
            for f in ("name", "structure", "annotation", "scale", "verifiability"):
                assert d.get(f), (m["key"], f)
        assert m["commissioning"].get("solution"), m["key"]
        assert m["commissioning"].get("commission"), m["key"]
        assert len(m["citations"]) >= 3, m["key"]
        for c in m["citations"]:
            assert c.get("title") and str(c.get("url", "")).startswith("http"), m["key"]
            assert c.get("venue") and c.get("year") and c.get("relevance"), m["key"]


def test_diagnostic_entries_explain_no_dataset(ref):
    for m in (x for x in ref["modes"] if x["kind"] == "diagnostic"):
        assert m["strategy"] and all(p.strip() for p in m["strategy"]), m["key"]
