"""Folder-name matching against the gear catalog, plus the standalone ISO and capture-date
readers -- these tests exercise the matching and inference directly, no Qt."""

from negpy.features.metadata.gear_models import Camera, FilmStock, GearLibrary
from negpy.services.assets.gear_match import match_gear_for_folder


def _library(cameras=(), film_stocks=()) -> GearLibrary:
    return GearLibrary(cameras=list(cameras), film_stocks=list(film_stocks))


def test_a_shared_word_matches_a_film_stock():
    gold = FilmStock(manufacturer="Kodak", stock_name="Gold 200")
    library = _library(film_stocks=[gold])

    match = match_gear_for_folder("08_penf_gold_marbella", library)

    assert match.film_stock_id == gold.id


def test_a_squashed_abbreviation_matches_a_camera():
    pen_f = Camera(make="Olympus", model="Pen F")
    library = _library(cameras=[pen_f])

    match = match_gear_for_folder("08_penf_gold_marbella", library)

    assert match.camera_id == pen_f.id


def test_camera_and_film_stock_match_independently():
    pen_f = Camera(make="Olympus", model="Pen F")
    gold = FilmStock(manufacturer="Kodak", stock_name="Gold 200")
    library = _library(cameras=[pen_f], film_stocks=[gold])

    match = match_gear_for_folder("08_penf_gold_marbella", library)

    assert match.camera_id == pen_f.id
    assert match.film_stock_id == gold.id
    assert match.any()


def test_an_unrelated_folder_name_matches_nothing():
    library = _library(
        cameras=[Camera(make="Olympus", model="Pen F")],
        film_stocks=[FilmStock(manufacturer="Kodak", stock_name="Gold 200")],
    )

    match = match_gear_for_folder("family_trip_photos", library)

    assert not match.any()


def test_two_candidates_sharing_a_word_is_treated_as_no_match():
    plus_a = FilmStock(manufacturer="Ilford", stock_name="HP5 Plus 400")
    plus_b = FilmStock(manufacturer="Ilford", stock_name="FP4 Plus 125")
    library = _library(film_stocks=[plus_a, plus_b])

    match = match_gear_for_folder("roll_plus_test", library)

    assert match.film_stock_id == ""


def test_matching_is_case_insensitive():
    portra = FilmStock(manufacturer="Kodak", stock_name="Portra 400")
    library = _library(film_stocks=[portra])

    match = match_gear_for_folder("2024_PORTRA_leica", library)

    assert match.film_stock_id == portra.id


def test_empty_library_matches_nothing():
    match = match_gear_for_folder("08_penf_gold_marbella", GearLibrary())

    assert not match.any()


def test_display_name_is_matched_when_set():
    custom = Camera(make="", model="", display_name="Widelux F7")
    library = _library(cameras=[custom])

    match = match_gear_for_folder("widelux_panorama_roll", library)

    assert match.camera_id == custom.id


def test_a_short_unrelated_substring_does_not_spuriously_match():
    """ "marbella" should not accidentally match a stock whose squashed name happens to
    share a short run with it."""
    library = _library(film_stocks=[FilmStock(manufacturer="Kodak", stock_name="Ektar 100")])

    match = match_gear_for_folder("08_penf_gold_marbella", library)

    assert match.film_stock_id == ""


def test_a_plausible_number_infers_an_iso_when_no_stock_matches():
    match = match_gear_for_folder("06_scala_50_vietnam", GearLibrary())

    assert match.iso == 50


def test_a_film_format_number_is_not_treated_as_an_iso():
    match = match_gear_for_folder("08_bronica_ilford_120_marbella", GearLibrary())

    assert match.iso is None


def test_a_number_embedded_in_a_letter_prefixed_token_is_read_as_an_iso():
    """The roll-sequence prefix ("05") and the camera token ("om1hd") both carry digits too
    short to be a real ISO, so only "e100" resolves."""
    match = match_gear_for_folder("05_om1hd_e100_tailandia", GearLibrary())

    assert match.iso == 100


def test_two_plausible_iso_candidates_is_treated_as_no_match():
    match = match_gear_for_folder("50_roll_400_iso", GearLibrary())

    assert match.iso is None


def test_iso_is_not_inferred_standalone_once_a_stock_already_matched():
    """The matched stock's own ISO reaches MetadataConfig through metadata_from_gear, so a
    standalone guess here would be redundant at best and conflicting at worst."""
    gold = FilmStock(manufacturer="Kodak", stock_name="Gold 200")
    library = _library(film_stocks=[gold])

    match = match_gear_for_folder("05_penees_kodakgold200_tailandia", library)

    assert match.film_stock_id == gold.id
    assert match.iso is None


def test_a_date_is_inferred_from_the_folder_name():
    match = match_gear_for_folder("2024-10-17_family_trip", GearLibrary())

    assert match.capture_date == "2024-10-17"


def test_a_date_with_underscores_or_no_separator_is_still_inferred():
    assert match_gear_for_folder("2024_10_17_family_trip", GearLibrary()).capture_date == "2024-10-17"
    assert match_gear_for_folder("20241017_family_trip", GearLibrary()).capture_date == "2024-10-17"


def test_two_distinct_dates_is_treated_as_no_match():
    match = match_gear_for_folder("2024-10-17_backup_of_2023-01-05", GearLibrary())

    assert match.capture_date == ""


def test_an_impossible_date_is_not_inferred():
    match = match_gear_for_folder("2024-13-40_family_trip", GearLibrary())

    assert match.capture_date == ""
