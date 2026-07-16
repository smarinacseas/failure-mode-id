"""T1.1 coverage-pool verifier tests (CAUSE_A archetype: noticing/tracking
independent requirements). Written test-first — coverage_pool.py does not exist
yet, so this fails RED until it's implemented.

Each verifier gets positive / negative / near-miss coverage (Gate GT1).
"""
import pytest

import coverage_pool  # noqa: F401 — importing registers the coverage verifiers
from base import CheckResult, check


def spec(t, **kw):
    return {"type": t, **kw}


# --- base dispatch -----------------------------------------------------------

def test_check_returns_checkresult():
    r = check("apple", spec("keyword_include", keywords=["apple"]))
    assert isinstance(r, CheckResult)
    assert isinstance(r.passed, bool)
    assert isinstance(r.detail, str)


def test_check_unknown_type_raises():
    with pytest.raises(KeyError):
        check("x", spec("no_such_verifier"))


# --- keyword_include (must-include) ------------------------------------------

def test_keyword_include_all_present_passes():
    assert check("I have an apple and a banana.",
                 spec("keyword_include", keywords=["apple", "banana"])).passed is True


def test_keyword_include_missing_one_fails_and_names_it():
    r = check("I have an apple.", spec("keyword_include", keywords=["apple", "banana"]))
    assert r.passed is False
    assert "banana" in r.detail


def test_keyword_include_case_insensitive_by_default():
    assert check("APPLE and Banana here",
                 spec("keyword_include", keywords=["apple", "banana"])).passed is True


def test_keyword_include_case_sensitive_when_requested():
    assert check("APPLE",
                 spec("keyword_include", keywords=["apple"], case_sensitive=True)).passed is False


# --- keyword_exclude (must-avoid) --------------------------------------------

def test_keyword_exclude_none_present_passes():
    assert check("All good here.",
                 spec("keyword_exclude", keywords=["error", "fail"])).passed is True


def test_keyword_exclude_present_fails_and_names_it():
    r = check("There was an error.", spec("keyword_exclude", keywords=["error", "warn"]))
    assert r.passed is False
    assert "error" in r.detail


# --- required_sections -------------------------------------------------------

def test_required_sections_all_headers_pass():
    body = "## Summary\nfoo bar\n\n## Next Steps\n- do a thing"
    assert check(body, spec("required_sections", sections=["Summary", "Next Steps"])).passed is True


def test_required_sections_plain_and_bold_headers_pass():
    body = "Summary:\nfoo\n\n**Next Steps**\nbar"
    assert check(body, spec("required_sections", sections=["Summary", "Next Steps"])).passed is True


def test_required_sections_missing_section_fails():
    r = check("## Summary\nfoo", spec("required_sections", sections=["Summary", "Next Steps"]))
    assert r.passed is False
    assert "Next Steps" in r.detail


def test_required_sections_inline_mention_is_not_a_header():
    # near-miss: the word appears in prose but never as a section header
    r = check("In the summary we describe the next steps in one paragraph.",
              spec("required_sections", sections=["Summary"]))
    assert r.passed is False


# --- no_placeholders ---------------------------------------------------------

def test_no_placeholders_clean_text_passes():
    assert check("Dear John, your order ships Monday.", spec("no_placeholders")).passed is True


def test_no_placeholders_allcaps_bracket_template_fails():
    r = check("Dear [NAME], your order ships on [DATE].", spec("no_placeholders"))
    assert r.passed is False
    assert "[NAME]" in r.detail


def test_no_placeholders_mustache_and_angle_templates_fail():
    assert check("Hello {{name}}", spec("no_placeholders")).passed is False
    assert check("Status: <TODO>", spec("no_placeholders")).passed is False


def test_no_placeholders_lowercase_bracket_is_not_a_placeholder():
    # near-miss: bracketed lowercase / numeric text is legitimate, not a template slot
    assert check("See item [3] and the aside [details below].",
                 spec("no_placeholders")).passed is True
