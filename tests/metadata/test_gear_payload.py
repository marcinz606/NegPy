"""Tests for gear library and metadata payload resolution."""

import os


import piexif

import pytest


from negpy.features.metadata.gear_models import Camera, FilmStock, GearLibrary, GearPreset, Lens

from negpy.features.metadata.gear_logic import metadata_from_gear

from negpy.features.metadata.models import MetadataConfig

from negpy.features.metadata.payload import build_metadata_payload, build_image_description, has_capture_gear

from negpy.features.metadata.xmp import build_xmp_xml

from negpy.features.metadata.writer import embed_metadata

from negpy.services.assets.gear import GearProfiles


@pytest.fixture
def gear_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.services.assets.gear.APP_CONFIG.gear_dir", str(tmp_path))

    return tmp_path


def test_seed_example_copies_bundled_files(gear_dir):
    GearProfiles.seed_example()

    assert os.path.isfile(os.path.join(gear_dir, "cameras.json"))


def test_load_and_save_library(gear_dir):
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Canon", model="AE-1")],
        lenses=[Lens(id="l1", lens_model="50mm", make="Canon")],
        film_stocks=[FilmStock(id="f1", manufacturer="Kodak", stock_name="Portra 400", iso=400)],
        gear_presets=[GearPreset(id="p1", display_name="Test", camera_id="c1", lens_id="l1", film_stock_id="f1")],
    )

    GearProfiles.save_library(library)

    loaded = GearProfiles.load_library()

    assert len(loaded.cameras) == 1

    assert loaded.cameras[0].make == "Canon"

    assert loaded.gear_presets[0].display_name == "Test"


def test_metadata_from_gear_preset():
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Canon", model="AE-1 Program")],
        lenses=[Lens(id="l1", lens_model="FD 50mm f/1.4", make="Canon", focal_length_mm=50, max_aperture=1.4)],
        film_stocks=[FilmStock(id="f1", manufacturer="Kodak", stock_name="Portra 400", iso=400)],
        gear_presets=[GearPreset(id="p1", display_name="Combo", camera_id="c1", lens_id="l1", film_stock_id="f1")],
    )

    config = metadata_from_gear(MetadataConfig(), library, gear_preset_id="p1")

    assert config.camera_make == "Canon"

    assert config.camera_model == "AE-1 Program"

    assert config.film == "Kodak Portra 400"

    assert config.film_iso == 400


def test_build_image_description():
    from negpy.features.metadata.payload import MetadataPayload

    payload = MetadataPayload(
        camera_make="Canon",
        camera_model="AE-1",
        lens_model="50mm f/1.4",
        film_stock="Portra 400",
        iso=400,
    )

    assert build_image_description(payload) == "Canon AE-1 • 50mm f/1.4 • Portra 400 • ISO 400"


def test_build_metadata_payload_preview_pairs():
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Canon", model="AE-1")],
        lenses=[],
        film_stocks=[],
        gear_presets=[],
    )

    config = MetadataConfig(camera_id="c1", developer="D-76 1+1")

    payload = build_metadata_payload(config, library)

    pairs = dict(payload.to_preview_pairs())

    assert pairs["Camera make"] == "Canon"

    assert pairs["Developer"] == "D-76 1+1"

    assert payload.exif_flags.camera is True


def test_developer_only_does_not_trigger_capture_exif():
    assert has_capture_gear(MetadataConfig(developer="D-76")) is False


def test_xmp_contains_negpy_capture_namespace():
    from negpy.features.metadata.payload import MetadataPayload

    payload = MetadataPayload(
        film_stock="Portra 400",
        film_manufacturer="Kodak",
        film_format="35mm",
        developer="D-76",
    )

    xml = build_xmp_xml(payload)

    assert "negpy:CaptureFilmStock" in xml

    assert "negpy:CaptureFilmManufacturer" in xml

    assert "negpy:Developer" in xml

    assert "tiff:Make" not in xml


def test_scan_rig_preserved_in_xmp_while_exif_shows_analog():
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Nikon", model="FM2")],
        lenses=[Lens(id="l1", lens_model="Nikkor 28mm f/2.8 AIS", make="Nikkor", focal_length_mm=28, max_aperture=2.8)],
        film_stocks=[],
        gear_presets=[],
    )

    source_exif = {
        "0th": {
            piexif.ImageIFD.Make: b"NIKON CORPORATION",
            piexif.ImageIFD.Model: b"NIKON D750",
        },
        "Exif": {
            piexif.ExifIFD.LensMake: b"NIKON",
            piexif.ExifIFD.LensModel: b"AF-S 60mm f/2.8G",
            piexif.ExifIFD.FocalLength: (600, 10),
            piexif.ExifIFD.FocalLengthIn35mmFilm: 60,
            piexif.ExifIFD.ExposureTime: (1, 640),
            piexif.ExifIFD.FNumber: (56, 10),
            piexif.ExifIFD.ISOSpeedRatings: 100,
        },
        "GPS": {},
        "Interop": {},
        "1st": {},
    }

    config = MetadataConfig(camera_id="c1", lens_id="l1", scanning="DSLR copy-stand")

    payload = build_metadata_payload(config, library, source_exif)

    assert payload.camera_model == "FM2"

    assert payload.scan_camera_make == "NIKON CORPORATION"

    assert payload.exif_flags.camera is True

    assert payload.exif_flags.lens is True

    xml = build_xmp_xml(payload)

    assert "negpy:ScanCameraMake" in xml

    assert "negpy:CaptureCameraModel" in xml

    assert "NIKON CORPORATION" in xml


def test_embed_jpeg_analog_exif_and_scan_xmp():
    from PIL import Image

    buf = __import__("io").BytesIO()

    Image.new("RGB", (8, 8), (128, 64, 32)).save(buf, format="JPEG")

    jpeg = buf.getvalue()

    source_exif = {
        "0th": {
            piexif.ImageIFD.Make: b"NIKON CORPORATION",
            piexif.ImageIFD.Model: b"NIKON D750",
        },
        "Exif": {
            piexif.ExifIFD.LensMake: b"NIKON",
            piexif.ExifIFD.LensModel: b"AF-S 60mm f/2.8G",
            piexif.ExifIFD.FocalLength: (600, 10),
            piexif.ExifIFD.FocalLengthIn35mmFilm: 60,
            piexif.ExifIFD.ISOSpeedRatings: 100,
        },
        "GPS": {},
        "Interop": {},
        "1st": {},
    }

    library = GearLibrary(
        cameras=[Camera(id="c1", make="Nikon", model="FM2")],
        lenses=[Lens(id="l1", lens_model="Nikkor 28mm f/2.8 AIS", make="Nikkor", focal_length_mm=28, max_aperture=2.8)],
        film_stocks=[FilmStock(id="f1", manufacturer="Kodak", stock_name="Portra 400", iso=400)],
        gear_presets=[],
    )

    config = MetadataConfig(
        camera_id="c1",
        lens_id="l1",
        film_stock_id="f1",
        film="Portra 400",
        scanning="DSLR scan",
    )

    out = embed_metadata(jpeg, config, source_exif, gear=library)

    assert b"http://ns.adobe.com/xap/1.0/" in out

    assert b"negpy:ScanCameraMake" in out

    assert b"negpy:CaptureCameraModel" in out

    assert b"NIKON CORPORATION" in out

    loaded = piexif.load(out)

    assert loaded["0th"][piexif.ImageIFD.Make] == b"Nikon"

    assert loaded["0th"][piexif.ImageIFD.Model] == b"FM2"

    assert loaded["Exif"][piexif.ExifIFD.LensModel] == b"Nikkor 28mm f/2.8 AIS"

    assert loaded["Exif"][piexif.ExifIFD.FocalLength] == (28, 1)

    assert piexif.ExifIFD.FocalLengthIn35mmFilm not in loaded["Exif"]

    assert loaded["Exif"][piexif.ExifIFD.ISOSpeedRatings] == 400

    assert loaded["0th"][piexif.ImageIFD.Software] == b"NegPy"


def test_embed_keeps_scan_exif_when_capture_not_set():
    from PIL import Image

    buf = __import__("io").BytesIO()

    Image.new("RGB", (8, 8), (128, 64, 32)).save(buf, format="JPEG")

    jpeg = buf.getvalue()

    source_exif = {
        "0th": {
            piexif.ImageIFD.Make: b"Plustek",
            piexif.ImageIFD.Model: b"OpticFilm 8200",
        },
        "Exif": {
            piexif.ExifIFD.LensModel: b"",
            piexif.ExifIFD.ISOSpeedRatings: 200,
        },
        "GPS": {},
        "Interop": {},
        "1st": {},
    }

    out = embed_metadata(jpeg, MetadataConfig(developer="HC-110"), source_exif)

    loaded = piexif.load(out)

    assert loaded["0th"][piexif.ImageIFD.Make] == b"Plustek"

    assert loaded["0th"][piexif.ImageIFD.Model] == b"OpticFilm 8200"

    assert loaded["Exif"][piexif.ExifIFD.ISOSpeedRatings] == 200
