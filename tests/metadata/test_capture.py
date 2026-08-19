"""Tests for capture date/place parsing and conversion."""

import piexif
import pytest

from negpy.features.metadata.exif_read import gps_decimal
from negpy.features.metadata.capture import (
    deg2tile,
    exif_gps_rationals,
    format_coords,
    parse_capture_date,
    parse_coords,
    place_summary,
    tile2deg,
    xmp_gps,
)


class TestParseCaptureDate:
    @pytest.mark.parametrize(
        "text, normalized, precision",
        [
            ("1998", "1998", "year"),
            ("1998-07", "1998-07", "month"),
            ("1998-7", "1998-07", "month"),
            ("1998/07/14", "1998-07-14", "day"),
            ("1998-07-14 16:30", "1998-07-14 16:30", "minute"),
            ("1998-07-14T16:30:05", "1998-07-14 16:30:05", "second"),
            ("  2024-02-29  ", "2024-02-29", "day"),
        ],
    )
    def test_accepts_partial_forms(self, text: str, normalized: str, precision: str) -> None:
        parsed = parse_capture_date(text)
        assert parsed is not None
        assert (parsed.text, parsed.precision) == (normalized, precision)

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "98", "1998-13", "1998-02-30", "2026-02-29", "1998-07-14 25:00", "yesterday", "1700", "3200"],
    )
    def test_rejects_impossible_instants(self, text: str) -> None:
        assert parse_capture_date(text) is None

    def test_offset_needs_a_time(self) -> None:
        assert parse_capture_date("1998-07-14 16:30+02:00").tz_offset == "+02:00"
        assert parse_capture_date("1998-07-14T16:30Z").tz_offset == "+00:00"
        assert parse_capture_date("1998-07-14").tz_offset == ""

    def test_exif_pads_the_unknown_parts(self) -> None:
        assert parse_capture_date("1998").exif_text() == "1998:01:01 00:00:00"
        assert parse_capture_date("1998-07").exif_text() == "1998:07:01 00:00:00"
        assert parse_capture_date("1998-07-14 16:30").exif_text() == "1998:07:14 16:30:00"

    def test_xmp_keeps_the_truncation(self) -> None:
        assert parse_capture_date("1998-07").xmp_text() == "1998-07"
        assert parse_capture_date("1998-07-14 16:30+02:00").xmp_text() == "1998-07-14 16:30+02:00"

    def test_compact_and_year_for_filenames(self) -> None:
        parsed = parse_capture_date("1998-07")
        assert (parsed.compact(), parsed.year) == ("19980701", 1998)


class TestParseCoords:
    @pytest.mark.parametrize(
        "text",
        [
            "35.6762, 139.6503",
            "35.6762 139.6503",
            "https://www.openstreetmap.org/#map=13/35.6762/139.6503",
            "https://www.openstreetmap.org/?mlat=35.6762&mlon=139.6503#map=16/35.6762/139.6503",
            "https://www.google.com/maps/@35.6762,139.6503,15z",
            "https://maps.google.com/?q=35.6762,139.6503",
        ],
    )
    def test_accepts_pairs_and_map_links(self, text: str) -> None:
        lat, lon = parse_coords(text)
        assert lat == pytest.approx(35.6762)
        assert lon == pytest.approx(139.6503)

    @pytest.mark.parametrize("text", ["", "Tokyo", "91.0, 139.0", "35.0, 200.0", "35.0"])
    def test_rejects_non_positions(self, text: str) -> None:
        assert parse_coords(text) is None

    def test_negative_pair(self) -> None:
        assert parse_coords("-33.8688, -151.2093") == (-33.8688, -151.2093)


class TestPlaceSummary:
    def test_names_win_over_coordinates(self) -> None:
        assert place_summary("Tokyo", "", "Japan", 35.0, 139.0) == "Tokyo, Japan"

    def test_coordinates_when_no_names(self) -> None:
        assert place_summary("", "", "", 35.0, 139.0) == format_coords(35.0, 139.0)

    def test_empty_when_nothing_is_set(self) -> None:
        assert place_summary("", "", "", None, None) == ""


class TestGps:
    def test_exif_rationals_and_hemispheres(self) -> None:
        gps = exif_gps_rationals(-33.8688, 151.2093)
        assert gps[piexif.GPSIFD.GPSLatitudeRef] == b"S"
        assert gps[piexif.GPSIFD.GPSLongitudeRef] == b"E"
        assert gps[piexif.GPSIFD.GPSMapDatum] == b"WGS-84"
        degrees, minutes, seconds = gps[piexif.GPSIFD.GPSLatitude]
        decimal = degrees[0] + minutes[0] / 60.0 + (seconds[0] / seconds[1]) / 3600.0
        assert decimal == pytest.approx(33.8688, abs=1e-4)

    def test_xmp_form(self) -> None:
        lat, lon = xmp_gps(35.6762, -139.6503)
        assert lat.startswith("35,40.57") and lat.endswith("N")
        assert lon.startswith("139,39.01") and lon.endswith("W")


class TestTileMath:
    @pytest.mark.parametrize("zoom", [2, 8, 18])
    @pytest.mark.parametrize("lat, lon", [(0.0, 0.0), (35.6762, 139.6503), (-33.8688, 151.2093)])
    def test_round_trip(self, lat: float, lon: float, zoom: int) -> None:
        back_lat, back_lon = tile2deg(*deg2tile(lat, lon, zoom), zoom)
        assert back_lat == pytest.approx(lat, abs=1e-6)
        assert back_lon == pytest.approx(lon, abs=1e-6)

    def test_zoom_zero_centre_is_the_tile_middle(self) -> None:
        assert deg2tile(0.0, 0.0, 1) == (1.0, 1.0)

    def test_latitude_clamps_at_the_mercator_limit(self) -> None:
        _x, y = deg2tile(89.9, 0.0, 4)
        assert 0.0 <= y <= 16.0


class TestGpsDecimal:
    @pytest.mark.parametrize(
        "dms, ref, expected",
        [
            (((51, 1), (30, 1), (0, 100)), b"N", 51.5),
            (((0, 1), (7, 1), (3900, 100)), b"W", -0.1275),
            (((33, 1), (52, 1), (768, 100)), b"S", -33.8688),
        ],
    )
    def test_hemisphere_signs_the_value(self, dms, ref, expected: float) -> None:
        assert gps_decimal(dms, ref) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("dms", [None, (), ((1, 1), (2, 1)), ((1, 0), (2, 1), (3, 1)), "50"])
    def test_malformed_triplets_are_none(self, dms) -> None:
        assert gps_decimal(dms, b"N") is None

    def test_round_trips_the_exif_writer(self) -> None:
        rationals = exif_gps_rationals(35.6762, -139.6503)
        lat = gps_decimal(rationals[piexif.GPSIFD.GPSLatitude], rationals[piexif.GPSIFD.GPSLatitudeRef])
        lon = gps_decimal(rationals[piexif.GPSIFD.GPSLongitude], rationals[piexif.GPSIFD.GPSLongitudeRef])
        assert (lat, lon) == pytest.approx((35.6762, -139.6503), abs=1e-4)
