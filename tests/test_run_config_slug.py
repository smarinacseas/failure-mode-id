import pytest
from pipeline.run_config import parse_slug, track_for_slug, InvalidSlugError


def test_parse_slug_accepts_e_series():
    assert parse_slug("E07-reasoning-full75") == (7, "reasoning-full75")


def test_parse_slug_accepts_t_series():
    assert parse_slug("T01-ihrlvr") == (1, "ihrlvr")


def test_parse_slug_rejects_other_prefix():
    with pytest.raises(InvalidSlugError):
        parse_slug("X01-nope")


def test_track_for_slug():
    assert track_for_slug("T01-ihrlvr") == "training"
    assert track_for_slug("E07-reasoning-full75") == "analysis"
    assert track_for_slug("t02-lower") == "training"  # case-insensitive prefix
