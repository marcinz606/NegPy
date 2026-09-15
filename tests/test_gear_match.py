"""Folder-name matching only ever points at gear already in the user's own library --
these tests exercise the word and squashed-abbreviation matching directly, no Qt."""

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

    match = match_gear_for_folder("2024-10-17_family_trip", library)

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
