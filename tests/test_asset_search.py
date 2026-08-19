from dataclasses import replace

import pytest

from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.models import MetadataConfig
from negpy.services.assets.search import Term, facts_for, match, parse_query


def _facts(**overrides):
    base = {
        "name": "img_0042.nef",
        "path": "/photos/roll12/img_0042.nef",
        "ext": "nef",
        "date": "2024-03-15",
        "keeper": False,
        "rejected": False,
        "edited": True,
        "film": "kodak portra 400",
        "camera": "nikon f3",
        "lens": "nikkor 50mm",
        "developer": "c-41",
        "format": "35mm",
        "scanning": "",
        "roll": "roll12",
        "frame": 7,
        "iso": 400,
        "push": 0,
        "shot": "",
        "place": "",
    }
    base.update(overrides)
    return base


def _hits(query: str, **overrides) -> bool:
    return match(parse_query(query), _facts(**overrides))


def test_empty_query_matches_everything():
    assert parse_query("") == []
    assert parse_query("   ") == []
    assert match([], {}) is True


@pytest.mark.parametrize(
    "query, expected",
    [
        ("img_0042", True),
        ("IMG_0042", True),
        ("0043", False),
        ("film:portra", True),
        ("film:velvia", False),
        ("camera:f3", True),
        ("lens:nikkor", True),
        ("developer:c-41", True),
        ("format:35mm", True),
        ("roll:roll12", True),
        ("ext:nef", True),
        ("ext:tif", False),
    ],
)
def test_text_fields(query, expected):
    assert _hits(query) is expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("iso:400", True),
        ("iso:200", False),
        ("iso:>=400", True),
        ("iso:>400", False),
        ("iso:<=400", True),
        ("iso:<400", False),
        ("frame:7", True),
        ("push:0", True),
    ],
)
def test_numeric_fields(query, expected):
    assert _hits(query) is expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("date:2024", True),
        ("date:2024-03", True),
        ("date:2023", False),
        ("date:>=2024-03", True),
        ("date:>=2024-04", False),
        ("date:<2024-04", True),
    ],
)
def test_date_prefix_comparison(query, expected):
    assert _hits(query) is expected


def test_flag_fields_ignore_their_value():
    assert _hits("keeper:", keeper=True) is True
    assert _hits("keeper:", keeper=False) is False
    assert _hits("rejected:", rejected=True) is True
    assert _hits("edited:", edited=False) is False


def test_negation():
    assert _hits("-film:velvia") is True
    assert _hits("-film:portra") is False
    assert _hits("-keeper:", keeper=False) is True


def test_missing_fact_never_matches_but_negation_does():
    assert _hits("film:portra", film="") is False
    assert _hits("-film:portra", film="") is True


def test_terms_are_anded():
    assert _hits("film:portra iso:>=400 -rejected:") is True
    assert _hits("film:portra iso:>=800") is False


def test_quoted_phrase_is_one_term():
    assert parse_query('camera:"Nikon F3"') == [Term("camera", ":", "nikon f3")]
    assert _hits('camera:"nikon f3"') is True


def test_unbalanced_quote_falls_back_to_whitespace_split():
    assert _hits('film:portra "unclosed') is False  # the stray token is a bare word
    assert parse_query('film:portra "unclosed') != []


def test_unknown_field_is_a_bare_word():
    assert parse_query("bogus:thing") == [Term("", ":", "bogus:thing")]
    assert _hits("bogus:thing") is False
    assert _hits("nef") is True  # bare word still matches the filename


def test_lone_dash_is_dropped():
    assert parse_query("-") == []


def test_facts_for_unedited_asset_has_no_metadata():
    asset = {"name": "IMG_1.NEF", "path": "/a/IMG_1.NEF", "mtime": 0.0}
    facts = facts_for(asset, None)
    assert facts["edited"] is False
    assert facts["name"] == "img_1.nef"
    assert facts["ext"] == "nef"
    assert "film" not in facts


def test_facts_for_edited_asset_reads_metadata_and_roll():
    config = replace(
        WorkspaceConfig(),
        metadata=MetadataConfig(
            film="Portra 400",
            film_manufacturer="Kodak",
            camera_make="Nikon",
            camera_model="F3",
            film_iso=400,
            capture_roll="Roll12",
            capture_frame=7,
            format="Other",
            format_other="6x17",
        ),
    )
    config = replace(config, process=replace(config.process, roll_name="Summer"))
    facts = facts_for({"name": "a.nef", "keeper": True}, config)

    assert facts["film"] == "kodak portra 400"
    assert facts["camera"] == "nikon f3"
    assert facts["iso"] == 400
    assert facts["frame"] == 7
    assert facts["format"] == "6x17"
    assert facts["roll"] == "roll12 summer"
    assert facts["keeper"] is True
    assert facts["edited"] is True


def test_facts_for_bad_mtime_yields_empty_date():
    assert facts_for({"name": "a.nef", "mtime": None}, None)["date"] == ""
    assert _hits("date:2024", date="") is False


@pytest.mark.parametrize(
    "query, expected",
    [
        ("shot:1998", True),
        ("shot:1998-07", True),
        ("shot:1999", False),
        ("shot:>=1998-07", True),
        ("shot:>=1998-08", False),
        ("shot:<2000", True),
        ("shot:>1998-07-14", False),
        ("-shot:1998", False),
    ],
)
def test_capture_date_orders_without_parsing_a_date(query: str, expected: bool):
    """`shot` is truncated ISO-8601, so a prefix comparison is the whole ordering."""
    assert _hits(query, shot="1998-07-14 16:30") is expected


def test_capture_date_is_not_the_file_date():
    assert _hits("shot:2024", shot="1998") is False
    assert _hits("date:2024") is True


def test_a_year_only_capture_date_is_not_claimed_to_be_in_any_month():
    assert _hits("shot:1998", shot="1998") is True
    assert _hits("shot:>=1998-07", shot="1998") is False


@pytest.mark.parametrize("query, expected", [("place:tokyo", True), ("place:japan", True), ("place:paris", False)])
def test_place_matches_any_of_city_state_country(query: str, expected: bool):
    assert _hits(query, place="tokyo tokyo japan") is expected


def test_unset_capture_date_and_place_never_match():
    assert _hits("shot:1998") is False
    assert _hits("place:tokyo") is False


def test_facts_for_edited_asset_reads_capture_date_and_place():
    config = replace(
        WorkspaceConfig(),
        metadata=MetadataConfig(
            capture_date="1998-07-14 16:30",
            location_city="Tokyo",
            location_state="Tokyo",
            location_country="Japan",
        ),
    )
    facts = facts_for({"name": "a.nef", "mtime": 0.0}, config)
    assert facts["shot"] == "1998-07-14 16:30"
    assert facts["place"] == "tokyo tokyo japan"
