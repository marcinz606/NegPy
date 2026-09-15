"""Tests for gear library and metadata payload resolution."""

import json
import os
from unittest.mock import MagicMock, patch


import piexif

import pytest

from PyQt6.QtWidgets import QDialog

from negpy.features.metadata.gear_models import Camera, FilmStock, GearLibrary, Lens

from negpy.features.metadata.gear_logic import metadata_from_gear, matches_gear_filter

from negpy.features.metadata.models import MetadataConfig

from negpy.features.metadata.payload import build_metadata_payload, build_image_description, has_capture_gear

from negpy.features.metadata.xmp import build_xmp_xml

from negpy.features.metadata.writer import embed_metadata

from negpy.services.assets.gear import GearProfiles


@pytest.fixture
def gear_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("negpy.services.assets.gear.APP_CONFIG.gear_dir", str(tmp_path))
    monkeypatch.setattr("negpy.services.assets.gear.get_resource_path", lambda _: str(tmp_path / "_no_bundled"))

    return tmp_path


def test_ensure_user_dir_creates_directory(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "gear"
    monkeypatch.setattr("negpy.services.assets.gear.APP_CONFIG.gear_dir", str(target))

    assert not target.exists()

    GearProfiles.ensure_user_dir()

    assert target.is_dir()


def test_load_library_merges_bundled_and_user_deduped(tmp_path, monkeypatch):
    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    monkeypatch.setattr("negpy.services.assets.gear.APP_CONFIG.gear_dir", str(user_dir))
    monkeypatch.setattr("negpy.services.assets.gear.get_resource_path", lambda _: str(bundled_dir))

    with open(bundled_dir / "cameras.json", "w", encoding="utf-8") as f:
        json.dump([{"id": "cam-shared", "make": "Leica", "model": "M6"}], f)
    with open(user_dir / "cameras.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {"id": "cam-shared", "make": "Leica", "model": "M6-EDITED"},
                {"id": "user-1", "make": "Pentax", "model": "K1000"},
            ],
            f,
        )

    library = GearProfiles.load_library()

    assert [c.id for c in library.cameras] == ["cam-shared", "user-1"]
    assert library.cameras[0].model == "M6"
    assert library.cameras[0].is_bundled is True
    assert library.cameras[1].is_bundled is False


def test_save_library_excludes_bundled_items(gear_dir):
    library = GearLibrary(
        cameras=[
            Camera(id="cam-bundled", make="Leica", is_bundled=True),
            Camera(id="user-1", make="Pentax"),
        ]
    )

    GearProfiles.save_library(library)

    on_disk = GearProfiles._read_list(os.path.join(gear_dir, "cameras.json"), Camera)
    assert [c.id for c in on_disk] == ["user-1"]


def test_duplicate_bundled_item_is_editable_and_persistable(gear_dir):
    from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel

    library = GearLibrary(cameras=[Camera(id="cam-bundled", make="Leica", model="M6", is_bundled=True)])
    dlg = GearLibraryPanel(library)

    # No personal cameras yet, so the default (personal-only) list is empty; the
    # bundled item is reached through the catalog toggle, the behavior under test.
    dlg.items.show_catalog_btn.setChecked(True)

    assert dlg.items.display_name_edit.isEnabled() is False
    assert dlg.items.del_btn.isEnabled() is False

    dlg.items._duplicate_item()

    dup = dlg.library().cameras[-1]
    assert dup.is_bundled is False
    assert dup.id != "cam-bundled"
    assert dlg.items.display_name_edit.isEnabled() is True

    on_disk = GearProfiles._read_list(os.path.join(gear_dir, "cameras.json"), Camera)
    assert [c.id for c in on_disk] == [dup.id]


def test_default_list_shows_personal_gear_only(gear_dir):
    from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel

    library = GearLibrary(
        cameras=[
            Camera(id="cam-bundled", make="Leica", model="M6", is_bundled=True),
            Camera(id="cam-mine", make="Pentax", model="K1000"),
        ]
    )
    dlg = GearLibraryPanel(library)

    labels = [dlg.items.item_list.item(i).text() for i in range(dlg.items.item_list.count())]
    assert labels == ["Pentax K1000"]

    dlg.items.show_catalog_btn.setChecked(True)
    labels = {dlg.items.item_list.item(i).text() for i in range(dlg.items.item_list.count())}
    assert labels == {"Pentax K1000", "Leica M6"}


def test_empty_personal_list_shows_a_hint_not_a_blank_box(gear_dir):
    from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel

    library = GearLibrary(cameras=[Camera(id="cam-bundled", make="Leica", model="M6", is_bundled=True)])
    dlg = GearLibraryPanel(library)

    assert dlg.items.empty_hint.isVisibleTo(dlg) is True
    assert "cameras" in dlg.items.empty_hint.text()
    assert dlg.items.item_list.isVisibleTo(dlg) is False

    dlg.items.show_catalog_btn.setChecked(True)
    assert dlg.items.empty_hint.isVisibleTo(dlg) is False
    assert dlg.items.item_list.isVisibleTo(dlg) is True


def test_add_item_opens_the_catalog_and_clones_the_pick(gear_dir):
    from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel

    library = GearLibrary(cameras=[Camera(id="cam-bundled", make="Leica", model="M6", is_bundled=True)])
    dlg = GearLibraryPanel(library)

    fake_dlg = MagicMock()
    fake_dlg.exec.return_value = QDialog.DialogCode.Accepted
    fake_dlg.wants_custom.return_value = False
    fake_dlg.selected_id.return_value = "cam-bundled"
    with patch("negpy.desktop.view.widgets.gear_library_panel.GearCatalogDialog", return_value=fake_dlg):
        dlg.items._add_item()

    assert len(dlg.library().cameras) == 2
    added = dlg.library().cameras[-1]
    assert added.is_bundled is False
    assert added.make == "Leica"
    assert added.id != "cam-bundled"


def test_add_item_custom_fallback_creates_a_blank_record(gear_dir):
    from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel

    library = GearLibrary(cameras=[Camera(id="cam-bundled", make="Leica", model="M6", is_bundled=True)])
    dlg = GearLibraryPanel(library)

    fake_dlg = MagicMock()
    fake_dlg.exec.return_value = QDialog.DialogCode.Accepted
    fake_dlg.wants_custom.return_value = True
    with patch("negpy.desktop.view.widgets.gear_library_panel.GearCatalogDialog", return_value=fake_dlg):
        dlg.items._add_item()

    added = dlg.library().cameras[-1]
    assert added.is_bundled is False
    # Real fields (make, model, ...) ride into EXIF verbatim, so they stay blank; only the
    # UI-only display_name carries a placeholder, pre-selected for the name the user types.
    assert added.make == ""
    assert added.model == ""
    assert added.display_name == "New Camera"
    assert dlg.items.display_name_edit.text() == "New Camera"


def test_delete_item_does_not_act_on_a_bundled_selection(gear_dir):
    from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel

    library = GearLibrary(cameras=[Camera(id="cam-bundled", make="Leica", model="M6", is_bundled=True)])
    dlg = GearLibraryPanel(library)
    dlg.items.show_catalog_btn.setChecked(True)

    dlg.items._delete_item()

    assert [c.id for c in dlg.library().cameras] == ["cam-bundled"]


def test_load_and_save_library(gear_dir):
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Canon", model="AE-1")],
        lenses=[Lens(id="l1", lens_model="50mm", make="Canon")],
        film_stocks=[FilmStock(id="f1", manufacturer="Kodak", stock_name="Portra 400", iso=400)],
    )

    GearProfiles.save_library(library)

    loaded = GearProfiles.load_library()

    assert len(loaded.cameras) == 1

    assert loaded.cameras[0].make == "Canon"


def test_matches_gear_filter_substring_case_insensitive():
    camera = Camera(id="c1", make="Nikon", model="FM2")
    lens = Lens(id="l1", lens_model="Nikkor 28mm f/2.8 AI-S", make="Nikkor", focal_length_mm=28)
    film = FilmStock(id="f1", manufacturer="Kodak", stock_name="Portra 400", iso=400)
    assert matches_gear_filter(camera, "fm2")
    assert matches_gear_filter(lens, "28")
    assert matches_gear_filter(film, "portra")
    assert not matches_gear_filter(camera, "canon")
    assert matches_gear_filter(camera, "")


def test_searchable_gear_combo_empty_selection_shows_placeholder():
    from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo

    combo = SearchableGearCombo(placeholder="Search cameras…")
    library = GearLibrary(cameras=[Camera(id="c1", make="Nikon", model="FM2")])
    combo.set_gear_items(library.cameras, "", lambda camera: camera.resolved_display_name)

    assert combo.selected_id() == ""
    assert combo.line_edit().text() == ""

    combo.set_gear_items(library.cameras, "c1", lambda camera: camera.resolved_display_name)
    assert combo.line_edit().text() == "Nikon FM2"
    assert combo.selected_id() == "c1"

    combo.set_selected_id("")
    assert combo.line_edit().text() == ""
    assert combo.selected_id() == ""


def test_searchable_gear_combo_reverts_partial_search_on_blur():
    from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo

    combo = SearchableGearCombo(placeholder="Search cameras…")
    library = GearLibrary(cameras=[Camera(id="c1", make="Nikon", model="FM2")])
    combo.set_gear_items(library.cameras, "c1", lambda camera: camera.resolved_display_name)

    combo.line_edit().setText("nik")
    combo._on_text_edited("nik")
    combo._finalize()

    assert combo.line_edit().text() == "Nikon FM2"
    assert combo.selected_id() == "c1"


def test_searchable_gear_combo_clearing_field_clears_selection():
    from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo

    combo = SearchableGearCombo(placeholder="Search cameras…")
    library = GearLibrary(cameras=[Camera(id="c1", make="Nikon", model="FM2")])
    combo.set_gear_items(library.cameras, "c1", lambda camera: camera.resolved_display_name)

    combo.line_edit().clear()
    combo._on_text_edited("")
    # Selection stays committed until the user finalizes (blur / Enter)…
    assert combo.selected_id() == "c1"
    assert combo.line_edit().text() == ""

    combo._finalize()
    assert combo.selected_id() == ""
    assert combo.line_edit().text() == ""


def test_searchable_gear_combo_replace_selection_after_search():
    from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo

    events: list[str] = []
    combo = SearchableGearCombo(placeholder="Search cameras…")
    combo.selection_changed.connect(events.append)
    library = GearLibrary(
        cameras=[
            Camera(id="leica", make="Leica", model="M6"),
            Camera(id="nikon", make="Nikon", model="FM2"),
        ]
    )
    combo.set_gear_items(library.cameras, "leica", lambda camera: camera.resolved_display_name)

    # Typing to search does NOT mutate the committed selection or emit.
    combo.line_edit().setText("Leica M")
    combo._on_text_edited("Leica M")
    combo.line_edit().setText("nik")
    combo._on_text_edited("nik")
    assert combo.selected_id() == "leica"
    assert events == []

    # Picking Nikon commits exactly once, and it sticks.
    combo._commit_id("nikon")
    assert combo.selected_id() == "nikon"
    assert combo.line_edit().text() == "Nikon FM2"
    assert events == ["nikon"]

    combo._finalize()
    assert combo.selected_id() == "nikon"
    assert combo.line_edit().text() == "Nikon FM2"
    assert events == ["nikon"]


def test_gear_library_dialog_item_search_hides_non_matching_selection():
    from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel

    library = GearLibrary(
        lenses=[
            Lens(id="l-canon", lens_model="FD 50mm f/1.4", make="Canon"),
            Lens(id="l-nikon", lens_model="Nikkor 50mm f/1.8 AI-S", make="Nikkor"),
        ]
    )
    dlg = GearLibraryPanel(library)
    dlg.items._select_category("lenses")
    dlg.items.item_list.setCurrentRow(0)  # Canon selected

    dlg.items.item_search.setText("nik")
    dlg.items._rebuild_item_list()

    labels = [dlg.items.item_list.item(i).text() for i in range(dlg.items.item_list.count())]
    assert labels == ["Nikkor 50mm f/1.8 AI-S"]
    assert dlg.items.item_list.currentRow() == -1
    assert dlg.items.lens_model_edit.text() == "FD 50mm f/1.4"


def test_metadata_from_gear_clearing_camera_id():
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Leica", model="M6")],
        lenses=[],
        film_stocks=[],
    )
    base = MetadataConfig(camera_id="c1", camera_make="Leica", camera_model="M6")

    cleared = metadata_from_gear(base, library, camera_id="")

    assert cleared.camera_id == ""
    assert cleared.camera_make == ""
    assert cleared.camera_model == ""


def test_metadata_from_gear_ids():
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Canon", model="AE-1 Program")],
        lenses=[Lens(id="l1", lens_model="FD 50mm f/1.4", make="Canon", focal_length_mm=50, max_aperture=1.4)],
        film_stocks=[FilmStock(id="f1", manufacturer="Kodak", stock_name="Portra 400", iso=400)],
    )

    config = metadata_from_gear(MetadataConfig(), library, camera_id="c1", lens_id="l1", film_stock_id="f1")

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
        film_format="35mm",
        developer="D-76 1+1",
        push_pull="Push +1",
        scan_method="DSLR copy-stand",
    )

    assert build_image_description(payload) == "Canon AE-1 • 50mm f/1.4 • Portra 400 • ISO 400"
    assert (
        build_image_description(
            payload,
            ("camera", "lens", "film", "iso", "format", "developer", "push_pull", "scanning"),
        )
        == "Canon AE-1 • 50mm f/1.4 • Portra 400 • ISO 400 • 35mm • D-76 1+1 • Push +1 • DSLR copy-stand"
    )


def test_build_image_description_omits_normal_push_pull():
    from negpy.features.metadata.payload import MetadataPayload

    payload = MetadataPayload(developer="D-76", push_pull="Normal", film_format="35mm")
    assert build_image_description(payload, ("format", "developer", "push_pull")) == "35mm • D-76"


def test_build_image_description_process_only():
    from negpy.features.metadata.payload import MetadataPayload

    payload = MetadataPayload(film_format="120", developer="HC-110", scan_method="flatbed")
    assert build_image_description(payload, ("format", "developer", "scanning")) == "120 • HC-110 • flatbed"


def test_build_metadata_payload_respects_description_fields():
    config = MetadataConfig(
        camera_make="Nikon",
        camera_model="FM2",
        film="Tri-X",
        film_iso=400,
        developer="D-76",
        description_fields=("camera", "film", "developer"),
    )
    payload = build_metadata_payload(config)
    assert payload.image_description == "Nikon FM2 • Tri-X • D-76"


def test_normalize_description_fields_orders_and_filters():
    from negpy.features.metadata.models import normalize_description_fields

    assert normalize_description_fields(["iso", "camera", "nope"]) == ("camera", "iso")
    assert MetadataConfig(description_fields=["scanning", "film"]).description_fields == ("film", "scanning")
    assert MetadataConfig().description_fields is None


def test_build_metadata_payload_preview_pairs():
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Canon", model="AE-1")],
        lenses=[],
        film_stocks=[],
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
    assert "negpy:CaptureRoll" not in xml
    assert "negpy:CaptureFrame" not in xml


def test_xmp_includes_capture_roll_and_frame_when_set():
    from negpy.features.metadata.payload import MetadataPayload

    xml = build_xmp_xml(MetadataPayload(capture_roll="Summer24", capture_frame=12))
    assert "negpy:CaptureRoll" in xml
    assert ">Summer24<" in xml
    assert "negpy:CaptureFrame" in xml
    assert ">12<" in xml


def test_xmp_omits_empty_capture_roll_and_unset_frame():
    from negpy.features.metadata.payload import MetadataPayload

    xml = build_xmp_xml(MetadataPayload(capture_roll="", capture_frame=None, film_stock="HP5"))
    assert "negpy:CaptureRoll" not in xml
    assert "negpy:CaptureFrame" not in xml
    assert "negpy:CaptureFilmStock" in xml


def test_build_metadata_payload_passes_capture_roll_frame():
    config = MetadataConfig(capture_roll="Roll001", capture_frame=3)
    payload = build_metadata_payload(config)
    assert payload.capture_roll == "Roll001"
    assert payload.capture_frame == 3
    pairs = dict(payload.to_preview_pairs())
    assert pairs["Roll"] == "Roll001"
    assert pairs["Frame"] == "3"


def test_scan_rig_preserved_in_xmp_while_exif_shows_analog():
    library = GearLibrary(
        cameras=[Camera(id="c1", make="Nikon", model="FM2")],
        lenses=[Lens(id="l1", lens_model="Nikkor 28mm f/2.8 AIS", make="Nikkor", focal_length_mm=28, max_aperture=2.8)],
        film_stocks=[],
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
