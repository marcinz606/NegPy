import io
from datetime import datetime, timedelta

import piexif
from PIL import Image

from negpy.features.metadata.capture import parse_capture_date
from negpy.features.metadata.models import MetadataConfig
from negpy.features.metadata.timezone import (
    capture_zone,
    is_zone,
    offset_text,
    resolve_local,
    zone_at,
    zone_names,
    zones_at,
)
from negpy.features.metadata.writer import embed_metadata

HURGHADA = (27.2574, 33.8128)
BERLIN = (52.52, 13.405)


class TestZoneAtPlace:
    def test_land_places(self) -> None:
        assert zone_at(*HURGHADA) == "Africa/Cairo"
        assert zone_at(*BERLIN) == "Europe/Berlin"
        assert zone_at(-33.8568, 151.2153) == "Australia/Sydney"

    def test_open_sea_gives_a_nautical_zone(self) -> None:
        assert zone_at(30.0, -40.0).startswith("Etc/GMT")

    def test_all_zones_include_the_first(self) -> None:
        assert zone_at(*HURGHADA) in zones_at(*HURGHADA)

    def test_capture_zone_prefers_a_valid_choice(self) -> None:
        assert capture_zone("Europe/Berlin", *HURGHADA) == "Europe/Berlin"
        assert capture_zone("Not/AZone", *HURGHADA) == "Africa/Cairo"
        assert capture_zone("", *HURGHADA) == "Africa/Cairo"
        assert capture_zone("", None, None) == ""

    def test_zone_names(self) -> None:
        assert is_zone("Africa/Cairo") and not is_zone("") and not is_zone("Mars/Olympus")
        assert "Africa/Cairo" in zone_names()


class TestLocalTimeInZone:
    def test_daylight_saving_follows_the_date(self) -> None:
        assert resolve_local(datetime(2026, 10, 25, 12, 38), "Africa/Cairo").utcoffset() == timedelta(hours=3)
        assert resolve_local(datetime(2026, 10, 30, 12, 0), "Africa/Cairo").utcoffset() == timedelta(hours=2)

    def test_a_repeated_hour_takes_its_first_occurrence(self) -> None:
        local = resolve_local(datetime(2026, 10, 25, 2, 30), "Europe/Berlin")
        assert (local.hour, local.utcoffset()) == (2, timedelta(hours=2))

    def test_a_skipped_hour_moves_forward(self) -> None:
        local = resolve_local(datetime(2026, 3, 29, 2, 30), "Europe/Berlin")
        assert (local.hour, local.minute, local.utcoffset()) == (3, 30, timedelta(hours=2))

    def test_offset_text(self) -> None:
        assert offset_text(timedelta(hours=3)) == "+03:00"
        assert offset_text(timedelta(hours=-3, minutes=-30)) == "-03:30"
        assert offset_text(timedelta(hours=5, minutes=45)) == "+05:45"
        assert offset_text(timedelta(0)) == "+00:00"


class TestCaptureDateInZone:
    def test_a_time_gets_the_offset_of_the_zone(self) -> None:
        cd = parse_capture_date("2026-10-25 10:12").in_zone("Africa/Cairo")
        assert (cd.text, cd.tz_offset) == ("2026-10-25 10:12", "+03:00")
        assert cd.utc().isoformat() == "2026-10-25T07:12:00+00:00"

    def test_seconds_are_kept(self) -> None:
        assert parse_capture_date("2026-10-25 10:12:09").in_zone("Africa/Cairo").text == "2026-10-25 10:12:09"

    def test_a_typed_offset_wins(self) -> None:
        assert parse_capture_date("2026-10-25 10:12+05:00").in_zone("Africa/Cairo").tz_offset == "+05:00"

    def test_a_date_without_a_time_gets_no_offset(self) -> None:
        cd = parse_capture_date("2026-10-25").in_zone("Africa/Cairo")
        assert cd.tz_offset == "" and cd.local() is None and cd.utc() is None

    def test_a_skipped_hour_moves_the_written_time(self) -> None:
        cd = parse_capture_date("2026-03-29 02:30").in_zone("Europe/Berlin")
        assert (cd.text, cd.tz_offset) == ("2026-03-29 03:30", "+02:00")


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (10, 20, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _export(config: MetadataConfig) -> tuple[dict, bytes]:
    out = embed_metadata(_jpeg_bytes(), config, {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}})
    return piexif.load(out), out


class TestExportWithZone:
    def test_the_zone_of_the_place_gives_offset_and_gps_time(self) -> None:
        exif, out = _export(MetadataConfig(capture_date="2026-10-25 10:12", gps_latitude=HURGHADA[0], gps_longitude=HURGHADA[1]))
        assert exif["Exif"][piexif.ExifIFD.DateTimeOriginal] == b"2026:10:25 10:12:00"
        assert exif["Exif"][piexif.ExifIFD.OffsetTimeOriginal] == b"+03:00"
        assert exif["GPS"][piexif.GPSIFD.GPSDateStamp] == b"2026:10:25"
        assert exif["GPS"][piexif.GPSIFD.GPSTimeStamp] == ((7, 1), (12, 1), (0, 1))
        assert b"<negpy:CaptureTimeZone>Africa/Cairo</negpy:CaptureTimeZone>" in out
        assert b"<photoshop:DateCreated>2026-10-25 10:12+03:00</photoshop:DateCreated>" in out

    def test_a_chosen_zone_overrides_the_place(self) -> None:
        config = MetadataConfig(
            capture_date="2026-10-25 10:12", gps_latitude=HURGHADA[0], gps_longitude=HURGHADA[1], capture_timezone="Europe/Berlin"
        )
        exif, out = _export(config)
        assert exif["Exif"][piexif.ExifIFD.OffsetTimeOriginal] == b"+01:00"
        assert b"<negpy:CaptureTimeZone>Europe/Berlin</negpy:CaptureTimeZone>" in out

    def test_a_chosen_zone_works_without_a_place(self) -> None:
        exif, _ = _export(MetadataConfig(capture_date="2026-10-25 10:12", capture_timezone="Africa/Cairo"))
        assert exif["Exif"][piexif.ExifIFD.OffsetTimeOriginal] == b"+03:00"
        assert piexif.GPSIFD.GPSDateStamp not in exif["GPS"]

    def test_a_typed_offset_wins_over_the_zone(self) -> None:
        exif, _ = _export(MetadataConfig(capture_date="2026-10-25 10:12+05:00", gps_latitude=HURGHADA[0], gps_longitude=HURGHADA[1]))
        assert exif["Exif"][piexif.ExifIFD.OffsetTimeOriginal] == b"+05:00"
        assert exif["GPS"][piexif.GPSIFD.GPSTimeStamp] == ((5, 1), (12, 1), (0, 1))

    def test_no_time_or_no_zone_means_no_offset(self) -> None:
        exif, _ = _export(MetadataConfig(capture_date="2026-10-25", gps_latitude=HURGHADA[0], gps_longitude=HURGHADA[1]))
        assert piexif.ExifIFD.OffsetTimeOriginal not in exif["Exif"]
        assert piexif.GPSIFD.GPSTimeStamp not in exif["GPS"]
        exif, _ = _export(MetadataConfig(capture_date="2026-10-25 10:12"))
        assert piexif.ExifIFD.OffsetTimeOriginal not in exif["Exif"]

    def test_time_source_and_anchor_go_to_xmp(self) -> None:
        config = MetadataConfig(capture_date="2026-10-25 10:12", capture_time_source="anchor", capture_anchor_ref="IMG_2514.HEIC")
        _, out = _export(config)
        assert b"<negpy:CaptureTimeSource>anchor</negpy:CaptureTimeSource>" in out
        assert b"<negpy:CaptureAnchorRef>IMG_2514.HEIC</negpy:CaptureAnchorRef>" in out

    def test_no_iptc_block_is_written(self) -> None:
        _, out = _export(MetadataConfig(capture_date="2026-10-25 10:12", gps_latitude=HURGHADA[0], gps_longitude=HURGHADA[1]))
        assert b"Photoshop 3.0" not in out
