"""Offline tests for the Metadata panel's Capture card."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

from dataclasses import replace

import piexif
import pytest
from PyQt6.QtWidgets import QApplication, QCheckBox, QLabel

from conftest import FakeController
from negpy.desktop.view.sidebar import metadata as metadata_module
from negpy.desktop.view.sidebar.metadata import MetadataSidebar
from negpy.features.metadata.gear_models import (
    DeveloperRecipe,
    GearLibrary,
    ProcessScanPreset,
    ScanSetup,
)

if not QApplication.instance():
    _app = QApplication(sys.argv)


@pytest.fixture
def sidebar(monkeypatch) -> MetadataSidebar:
    monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(GearLibrary))
    controller = FakeController()
    controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
    return MetadataSidebar(controller)


def _set_metadata(sidebar: MetadataSidebar, **changes) -> None:
    state = sidebar.state
    state.config = replace(state.config, metadata=replace(state.config.metadata, **changes))


class TestCaptureDate:
    def test_invalid_date_is_flagged_and_not_persisted(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(sidebar, capture_date="1998-07")
        sidebar.capture_date_edit.setText("1998-13")
        assert sidebar.capture_date_edit.styleSheet() != ""
        sidebar._persist_all_metadata_settings()
        assert sidebar.state.config.metadata.capture_date == "1998-07"

    def test_valid_date_is_normalized_on_persist(self, sidebar: MetadataSidebar) -> None:
        sidebar.capture_date_edit.setText("1998/7/4 16:30")
        assert sidebar.capture_date_edit.styleSheet() == ""
        sidebar._persist_all_metadata_settings()
        assert sidebar.state.config.metadata.capture_date == "1998-07-04 16:30"

    def test_cleared_date_persists_as_unset(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(sidebar, capture_date="1998")
        sidebar.sync_ui()
        sidebar.capture_date_edit.setText("")
        sidebar._persist_all_metadata_settings()
        assert sidebar.state.config.metadata.capture_date == ""


class TestCapturePlace:
    def test_place_field_shows_names_then_coordinates(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(sidebar, location_city="Tokyo", location_country="Japan")
        sidebar.sync_ui()
        assert sidebar.place_edit.text() == "Tokyo, Japan"

        _set_metadata(sidebar, location_city="", location_country="", gps_latitude=35.0, gps_longitude=139.0)
        sidebar.sync_ui()
        assert sidebar.place_edit.text() == "35.00000, 139.00000"

    def test_pasted_map_link_sets_coordinates_and_keeps_names(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(sidebar, location_city="Tokyo")
        sidebar.place_edit.setText("https://www.openstreetmap.org/#map=13/49.5/19.5")
        sidebar._on_place_edited()
        conf = sidebar.state.config.metadata
        assert (conf.gps_latitude, conf.gps_longitude) == (49.5, 19.5)
        assert conf.location_city == "Tokyo"

    def test_unparsable_place_text_is_reverted(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(sidebar, location_city="Tokyo", location_country="Japan")
        sidebar.place_edit.setText("somewhere nice")
        sidebar._on_place_edited()
        assert sidebar.place_edit.text() == "Tokyo, Japan"
        assert sidebar.state.config.metadata.gps_latitude is None

    def test_clear_empties_position_and_names(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(sidebar, location_city="Tokyo", gps_latitude=35.0, gps_longitude=139.0)
        sidebar._on_place_clear()
        conf = sidebar.state.config.metadata
        assert (conf.gps_latitude, conf.gps_longitude, conf.location_city) == (None, None, "")
        assert sidebar.place_edit.text() == ""

    def test_picker_result_is_applied(self, sidebar: MetadataSidebar, monkeypatch) -> None:
        class _StubDialog:
            DialogCode = metadata_module.LocationPickerDialog.DialogCode

            def __init__(self, *_args, **_kwargs):
                pass

            def exec(self):
                return self.DialogCode.Accepted

            def location(self):
                return 35.6762, 139.6503, "Tokyo", "Tokyo", "Japan"

        monkeypatch.setattr(metadata_module, "LocationPickerDialog", _StubDialog)
        sidebar._open_location_picker()
        conf = sidebar.state.config.metadata
        assert (conf.gps_latitude, conf.location_city, conf.location_country) == (
            35.6762,
            "Tokyo",
            "Japan",
        )
        assert sidebar.place_edit.text() == "Tokyo, Tokyo, Japan"


class TestSourceGpsPrefill:
    _SCAN_GPS = {
        "GPS": {
            piexif.GPSIFD.GPSLatitude: ((35, 1), (40, 1), (3432, 100)),
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLongitude: ((139, 1), (39, 1), (108, 100)),
            piexif.GPSIFD.GPSLongitudeRef: b"E",
        }
    }

    @staticmethod
    def _capture_picker(monkeypatch) -> dict:
        seen: dict = {}

        class _StubDialog:
            DialogCode = metadata_module.LocationPickerDialog.DialogCode

            def __init__(self, *args, **kwargs):
                seen.update(kwargs)

            def exec(self):
                return self.DialogCode.Rejected

        monkeypatch.setattr(metadata_module, "LocationPickerDialog", _StubDialog)
        return seen

    def _with_scan_exif(self, sidebar: MetadataSidebar) -> None:
        sidebar.state.current_file_hash = "hash1"
        sidebar.state.source_exif["hash1"] = self._SCAN_GPS

    def test_picker_opens_on_the_scan_position(self, sidebar: MetadataSidebar, monkeypatch) -> None:
        self._with_scan_exif(sidebar)
        seen = self._capture_picker(monkeypatch)
        sidebar._open_location_picker()
        assert seen["center"] == pytest.approx((35.6762, 139.6503), abs=1e-4)

    def test_scan_position_is_not_adopted_as_the_capture_place(self, sidebar: MetadataSidebar) -> None:
        self._with_scan_exif(sidebar)
        sidebar.sync_ui()
        assert sidebar.place_edit.text() == ""
        assert sidebar.state.config.metadata.gps_latitude is None

    def test_scan_position_shows_in_the_preview(self, sidebar: MetadataSidebar) -> None:
        self._with_scan_exif(sidebar)
        sidebar._update_preview()
        labels = [sidebar.preview_rows.itemAt(i).widget().findChildren(QLabel) for i in range(sidebar.preview_rows.count())]
        texts = [label.text() for row in labels for label in row]
        assert "Scan place" in texts
        assert "35.67620, 139.65030" in texts

    def test_an_existing_place_needs_no_centre(self, sidebar: MetadataSidebar, monkeypatch) -> None:
        self._with_scan_exif(sidebar)
        _set_metadata(sidebar, gps_latitude=1.0, gps_longitude=2.0)
        seen = self._capture_picker(monkeypatch)
        sidebar._open_location_picker()
        assert seen["center"] is None


class TestSyncCheckbox:
    def test_sits_at_the_top_beside_protect_and_not_inside_a_card(self, sidebar: MetadataSidebar) -> None:
        order = [sidebar.layout.indexOf(w) for w in (sidebar.protect_check, sidebar.sync_check)]
        assert -1 not in order
        assert order[0] < order[1] < sidebar.layout.indexOf(sidebar._metadata_controls)
        assert sidebar._metadata_controls.findChildren(QCheckBox).count(sidebar.sync_check) == 0

    def test_protect_disables_it(self, sidebar: MetadataSidebar) -> None:
        """Protect mode ignores the panel's fields, so syncing them would mean nothing."""
        sidebar._on_protect_toggled(True)
        assert sidebar.sync_check.isEnabled() is False
        sidebar._on_protect_toggled(False)
        assert sidebar.sync_check.isEnabled() is True

    def test_toggle_persists(self, sidebar: MetadataSidebar) -> None:
        sidebar.sync_check.setChecked(True)
        sidebar._persist_all_metadata_settings()
        assert sidebar.state.config.metadata.sync_to_batch is True


class TestPlaceButtons:
    def test_are_icon_only_with_tooltips_carrying_the_meaning(self, sidebar: MetadataSidebar) -> None:
        for button in (sidebar.place_map_btn, sidebar.place_clear_btn):
            assert button.text() == ""
            assert button.icon().isNull() is False
            assert button.toolTip() != ""


class TestDeveloperAndScanCombos:
    @pytest.fixture
    def stocked(self, sidebar: MetadataSidebar, monkeypatch) -> MetadataSidebar:
        library = GearLibrary(
            developers=[DeveloperRecipe(id="d1", developer="D-76", dilution="1+1", time="9:30", temperature_c=20)],
            scan_setups=[ScanSetup(id="s1", device="Sony A7RIV", light_source="Scanlight narrowband")],
            process_scan_presets=[ProcessScanPreset(id="ps1", display_name="Home B&W", developer_id="d1", scan_setup_id="s1")],
        )
        monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
        sidebar._refresh_gear_combos(force=True)
        return sidebar

    def test_picking_a_developer_fills_the_text_field(self, stocked: MetadataSidebar) -> None:
        stocked.developer_combo.set_selected_id("d1")
        stocked.developer_combo.selection_changed.emit("d1")

        assert stocked.state.config.metadata.developer_id == "d1"
        assert stocked.developer_edit.text() == "D-76 1+1, 9:30 @ 20 °C"

    def test_picking_a_scan_setup_fills_the_text_field(self, stocked: MetadataSidebar) -> None:
        stocked.scan_setup_combo.set_selected_id("s1")
        stocked.scan_setup_combo.selection_changed.emit("s1")

        assert stocked.state.config.metadata.scan_setup_id == "s1"
        assert stocked.scanning_edit.text() == "Copy stand — Sony A7RIV, Scanlight narrowband"

    def test_process_scan_preset_sets_both_cards(self, stocked: MetadataSidebar) -> None:
        stocked.process_scan_combo.set_selected_id("ps1")
        stocked.process_scan_combo.selection_changed.emit("ps1")

        assert stocked.developer_edit.text() == "D-76 1+1, 9:30 @ 20 °C"
        assert stocked.scanning_edit.text() == "Copy stand — Sony A7RIV, Scanlight narrowband"

    def test_a_manual_pick_clears_only_the_process_scan_preset(self, stocked: MetadataSidebar) -> None:
        """Chemistry and gear own separate presets, so one must not clear the other."""
        _set_metadata(stocked, gear_preset_id="p1")
        stocked.process_scan_combo.set_selected_id("ps1")
        stocked.process_scan_combo.selection_changed.emit("ps1")

        stocked.developer_combo.set_selected_id("d1")
        stocked.developer_combo.selection_changed.emit("d1")

        assert stocked.state.config.metadata.process_scan_preset_id == ""
        assert stocked.state.config.metadata.gear_preset_id == "p1"

    def test_clear_empties_both_slots(self, stocked: MetadataSidebar) -> None:
        stocked.process_scan_combo.set_selected_id("ps1")
        stocked.process_scan_combo.selection_changed.emit("ps1")

        stocked._on_process_scan_preset_clear()

        conf = stocked.state.config.metadata
        assert (conf.process_scan_preset_id, conf.developer_id, conf.scan_setup_id) == ("", "", "")
        assert conf.developer == ""
        assert conf.scanning == ""

    def test_a_pick_during_the_save_delay_keeps_both_edits(self, stocked: MetadataSidebar) -> None:
        """Typing then picking inside the debounce window must lose neither."""
        stocked.capture_roll_edit.setText("Roll007")

        stocked.developer_combo.set_selected_id("d1")
        stocked.developer_combo.selection_changed.emit("d1")

        conf = stocked.state.config.metadata
        assert conf.capture_roll == "Roll007"
        assert conf.developer == "D-76 1+1, 9:30 @ 20 °C"

    def test_a_typed_override_survives_a_later_camera_pick(self, stocked: MetadataSidebar) -> None:
        stocked.developer_combo.set_selected_id("d1")
        stocked.developer_combo.selection_changed.emit("d1")
        stocked.developer_edit.setText("D-76 1+1, stand 20 min")

        stocked.camera_combo.selection_changed.emit("")

        assert stocked.state.config.metadata.developer == "D-76 1+1, stand 20 min"
