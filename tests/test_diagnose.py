"""Tests for the diagnose stage (pipeline/_taxonomy.py + pipeline/diagnose.py).

Self-contained: fixtures build a tiny run directory under tmp_path; the
Anthropic Message Batches endpoint is a scripted fake (mirrors
tests/test_concurrency.py's harness — duplicated deliberately so each test
file stands alone)."""

import json
from types import SimpleNamespace

import config
import pipeline._taxonomy as taxonomy


def test_taxonomy_reserved_labels_always_present():
    """judge_suspect and other are a-priori labels (spec §3): they must be
    offered in BOTH trace modes, before and after Pass-1 population."""
    for trace_present in (True, False):
        keys = taxonomy.allowed_keys(trace_present)
        assert "judge_suspect" in keys
        assert "other" in keys


def test_taxonomy_no_trace_mode_collapses_trace_dependent_categories():
    """Without a trace, 'never noticed' vs 'noticed-but-dropped' are
    indistinguishable (spec §3): trace-requiring categories are withheld and
    the collapse category constraint_unaddressed is offered instead."""
    with_trace = taxonomy.allowed_keys(True)
    without = taxonomy.allowed_keys(False)
    assert "constraint_unaddressed" in without
    assert "constraint_unaddressed" not in with_trace
    for cat in taxonomy.DERIVED:
        if cat["requires_trace"]:
            assert cat["key"] not in without


def test_diagnose_system_prompt_is_analyst_not_grader():
    """Spec §4 blinding rule 4: distinct analyst role, no JUDGE_SYSTEM reuse,
    evidence-first instruction, and every offered category documented."""
    text = taxonomy.diagnose_system(True)
    assert "PASS" not in text and "FAIL" not in text.replace("failed", "")
    assert "evidence" in text.lower()
    for key in taxonomy.allowed_keys(True):
        assert key in text
