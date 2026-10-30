"""Offline tests for the Metadata panel's Capture card."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

from dataclasses import replace
from unittest.mock import patch

import piexif
import pytest
from PyQt6.QtWidgets import QWIDGETSIZE_MAX, QApplication, QLabel

from conftest import FakeController
from negpy.desktop.view.sidebar import metadata as metadata_module
from negpy.desktop.view.sidebar.metadata import MetadataSidebar
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel
from negpy.features.metadata.gear_models import Camera, FilmStock, GearLibrary
from negpy.kernel.system.config import APP_CONFIG

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


@pytest.fixture
def gear_sidebar(monkeypatch) -> MetadataSidebar:
    library = GearLibrary(
        cameras=[
            Camera(id="cam-bundled", make="Leica", model="M6", is_bundled=True),
            Camera(id="cam-mine", make="Pentax", model="K1000", is_bundled=False),
        ]
    )
    monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
    monkeypatch.setattr(metadata_module.GearProfiles, "save_library", staticmethod(lambda _lib: None))
    controller = FakeController()
    controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
    return MetadataSidebar(controller)


class TestGearCombos:
    def test_camera_combo_defaults_to_personal_gear_only(self, gear_sidebar: MetadataSidebar) -> None:
        ids = {item_id for _label, item_id, _search in gear_sidebar.camera_combo._entries}
        assert ids == {"cam-mine", metadata_module.OTHER_ID}

    def test_a_bundled_selection_made_before_the_filter_stays_visible(self, monkeypatch) -> None:
        library = GearLibrary(cameras=[Camera(id="cam-bundled", make="Leica", model="M6", is_bundled=True)])
        monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
        controller = FakeController()
        controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
        controller.state.config = replace(
            controller.state.config, metadata=replace(controller.state.config.metadata, camera_id="cam-bundled")
        )
        sidebar = MetadataSidebar(controller)
        assert sidebar.camera_combo.selected_id() == "cam-bundled"
        assert sidebar.camera_combo.line_edit().text() == "Leica M6"

    def test_other_pick_clones_the_catalog_camera_into_personal_gear(self, gear_sidebar: MetadataSidebar) -> None:
        cloned = Camera(id="cam-cloned", make="Leica", model="M6", is_bundled=False)
        with patch.object(metadata_module, "resolve_other_gear_pick", return_value=cloned):
            gear_sidebar.camera_combo._commit_id(metadata_module.OTHER_ID)

        new_id = gear_sidebar.state.config.metadata.camera_id
        assert new_id == "cam-cloned"
        assert gear_sidebar.camera_combo.selected_id() == new_id
        assert gear_sidebar.camera_combo.line_edit().text() == "Leica M6"

    def test_other_pick_add_custom_starts_a_blank_personal_camera(self, gear_sidebar: MetadataSidebar) -> None:
        custom = Camera(id="cam-custom", display_name="New Camera")
        with patch.object(metadata_module, "resolve_other_gear_pick", return_value=custom):
            gear_sidebar.camera_combo._commit_id(metadata_module.OTHER_ID)

        new_id = gear_sidebar.state.config.metadata.camera_id
        assert new_id == "cam-custom"
        assert gear_sidebar.camera_combo.line_edit().text() == "New Camera"

    def test_other_pick_cancelled_reverts_to_the_previous_selection(self, gear_sidebar: MetadataSidebar) -> None:
        _set_metadata(gear_sidebar, camera_id="cam-mine")
        gear_sidebar.sync_ui()

        with patch.object(metadata_module, "resolve_other_gear_pick", return_value=None):
            gear_sidebar.camera_combo._commit_id(metadata_module.OTHER_ID)

        assert gear_sidebar.camera_combo.selected_id() == "cam-mine"
        assert gear_sidebar.state.config.metadata.camera_id == "cam-mine"


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


class TestExposureLock:
    """The Exposure field is lock-gated: unlock to type, re-lock to keep the text."""

    def _type_and_relock(self, sidebar: MetadataSidebar) -> None:
        sidebar._toggle_exif_lock("exposure", sidebar.exposure_edit, sidebar._exposure_lock_btn, True)
        sidebar.exposure_edit.setText("1/125s f/2.8 ISO 400 -1/3ev")
        sidebar._toggle_exif_lock("exposure", sidebar.exposure_edit, sidebar._exposure_lock_btn, False)

    def test_relocking_keeps_the_typed_text(self, sidebar: MetadataSidebar) -> None:
        self._type_and_relock(sidebar)

        assert sidebar.exposure_edit.text() == "1/125s f/2.8 ISO 400 -1/3ev"
        assert sidebar.state.config.metadata.exposure_override == "1/125s f/2.8 ISO 400 -1/3ev"

    def test_the_debounce_and_a_frame_switch_keep_the_text(self, sidebar: MetadataSidebar) -> None:
        self._type_and_relock(sidebar)
        sidebar._persist_all_metadata_settings()
        sidebar._on_file_selected("/scans/other/frame002.tif")
        sidebar.sync_ui()

        assert sidebar.exposure_edit.text() == "1/125s f/2.8 ISO 400 -1/3ev"
        assert sidebar.state.config.metadata.exposure_override == "1/125s f/2.8 ISO 400 -1/3ev"


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


class TestCardLockOnEdit:
    """Editing a Capture value locks the card so a frame switch keeps the value."""

    def test_editing_place_locks_the_card(self, sidebar: MetadataSidebar) -> None:
        sidebar.controller.sync_metadata_card_locks.reset_mock()
        sidebar.place_edit.setText("35.68, 139.69")
        sidebar._on_place_edited()

        sidebar.controller.sync_metadata_card_locks.assert_called_once()
        meta = sidebar.state.config.metadata
        assert (meta.gps_latitude, meta.gps_longitude) == (35.68, 139.69)

    def test_persisting_metadata_locks_the_card(self, sidebar: MetadataSidebar) -> None:
        sidebar.controller.sync_metadata_card_locks.reset_mock()
        sidebar._mark_dirty()
        sidebar._persist_all_metadata_settings()

        sidebar.controller.sync_metadata_card_locks.assert_called_once()

    def test_flushing_exposure_override_locks_the_card(self, sidebar: MetadataSidebar) -> None:
        sidebar.controller.sync_metadata_card_locks.reset_mock()
        sidebar.exposure_edit.setText("1/125s f/2.8")
        sidebar._flush_exposure_override()

        sidebar.controller.sync_metadata_card_locks.assert_called_once()


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


def _cards_enabled(sidebar: MetadataSidebar) -> list[bool]:
    return [section.isEnabled() for _key, section in sidebar._scope_sections()]


def _page(sidebar: MetadataSidebar, key: str):
    return sidebar.tabs.stack.widget(sidebar.tabs.keys.index(key))


class TestTabs:
    def test_two_field_tabs_then_the_two_library_tabs_as_icons(self, sidebar: MetadataSidebar) -> None:
        assert [btn.text() for btn in sidebar.tabs.buttons] == ["Shot", "Process", "Presets", "My Gear"]
        assert [btn._icon_only for btn in sidebar.tabs.buttons] == [False, False, True, True]

    def test_each_card_sits_on_its_tab_under_that_tabs_header(self, sidebar: MetadataSidebar) -> None:
        for key, sections in (
            ("shot", (sidebar.gear_section, sidebar.capture_section, sidebar.exposure_section)),
            ("process", (sidebar.process_section, sidebar.scanning_section)),
        ):
            page = _page(sidebar, key)
            assert all(page.isAncestorOf(section) for section in (sidebar.tab_headers[key], *sections))

    def test_the_header_in_front_follows_the_tab(self, sidebar: MetadataSidebar) -> None:
        sidebar.show_tab("process")
        assert sidebar.active_tab_header() is sidebar.tab_headers["process"]
        sidebar.show_tab("gear")
        assert sidebar.active_tab_header() is None

    def test_the_tab_is_kept(self, sidebar: MetadataSidebar) -> None:
        sidebar.show_tab("presets")
        assert sidebar.controller.session.repo.get_global_setting("metadata_tab") == "presets"

    def test_a_tab_is_marked_while_a_card_on_it_is_edited(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(sidebar, developer="D-76")
        sidebar.sync_ui()
        assert [btn.is_marked() for btn in sidebar.tabs.buttons] == [False, True, False, False]

    def test_library_tab_tooltips_carry_their_bound_shortcut(self, sidebar: MetadataSidebar, monkeypatch) -> None:
        import negpy.desktop.view.shortcut_registry as registry

        monkeypatch.setattr(registry, "key_for", lambda action_id, bindings=None: "G" if action_id == "tab_gear_items" else "")
        sidebar.apply_shortcut_tooltips()

        tips = {key: btn.toolTip() for key, btn in zip(sidebar.tabs.keys, sidebar.tabs.buttons)}
        assert "G" in tips["gear"]
        assert "G" not in tips["presets"]


class TestPreviewPinning:
    """The Preview is pinned above the tabs like Analysis, and folds to its header."""

    def test_preview_is_pinned_above_the_tabs(self, sidebar: MetadataSidebar) -> None:
        assert sidebar.preview_split.widget(0) is sidebar.preview_section
        assert sidebar.preview_split.widget(1).isAncestorOf(sidebar.tabs.stack)

    def test_a_folded_preview_keeps_only_its_header(self, sidebar: MetadataSidebar) -> None:
        sidebar.preview_section.set_expanded(False)
        assert sidebar.preview_section.maximumHeight() == sidebar.preview_section.sizeHint().height()

        sidebar.preview_section.set_expanded(True)
        assert sidebar.preview_section.maximumHeight() == QWIDGETSIZE_MAX


class TestProtectGating:
    """Protect Original Metadata is a toggle on the Preview, so it stays live while the cards are off."""

    def test_protect_disables_the_cards(self, sidebar: MetadataSidebar) -> None:
        assert all(_cards_enabled(sidebar))

        _set_metadata(sidebar, protect_original_metadata=True)
        sidebar.sync_ui()
        assert not any(_cards_enabled(sidebar))

        _set_metadata(sidebar, protect_original_metadata=False)
        sidebar.sync_ui()
        assert all(_cards_enabled(sidebar))

    def test_the_toggle_persists_and_grays_out_the_cards(self, sidebar: MetadataSidebar) -> None:
        assert sidebar.preview_section.isAncestorOf(sidebar.protect_btn)
        sidebar.protect_btn.setChecked(True)
        assert sidebar.state.config.metadata.protect_original_metadata is True
        assert not any(_cards_enabled(sidebar))
        assert sidebar.protect_btn.isEnabled()

        _set_metadata(sidebar, protect_original_metadata=False)
        sidebar.sync_ui()
        assert not sidebar.protect_btn.isChecked()


class TestPlaceButtons:
    def test_are_icon_only_with_tooltips_carrying_the_meaning(self, sidebar: MetadataSidebar) -> None:
        for button in (sidebar.place_map_btn, sidebar.place_clear_btn):
            assert button.text() == ""
            assert button.icon().isNull() is False
            assert button.toolTip() != ""


class TestClearButtons:
    """Each Clear empties what its own picker fills, and leaves the rest of the card alone."""

    def test_process_clear_empties_the_recipe_and_its_link(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(
            sidebar,
            process_id="p1",
            developer="HC-110",
            process_dilution="1+31",
            push_pull=1,
            process_time_seconds=570,
            process_temperature_c=20.0,
            format="120",
            film="Ilford HP5+",
        )
        sidebar.sync_ui()

        sidebar.process_clear_btn.click()

        meta = sidebar.state.config.metadata
        assert meta.process_id == ""
        assert meta.developer == ""
        assert meta.process_dilution == ""
        assert meta.push_pull == 0
        assert meta.process_time_seconds is None
        assert meta.process_temperature_c is None
        # Format comes from the film stock, not the process picker.
        assert meta.format == "120"
        assert meta.film == "Ilford HP5+"

    def test_scanning_clear_keeps_roll_and_frame(self, sidebar: MetadataSidebar) -> None:
        _set_metadata(sidebar, scanning_id="s1", scanning="DSLR copy-stand", capture_roll="Roll042", capture_frame=12)
        sidebar.sync_ui()

        sidebar.scan_clear_btn.click()

        meta = sidebar.state.config.metadata
        assert meta.scanning_id == ""
        assert meta.scanning == ""
        # Stamped by the scan itself, not filled by the setup picker.
        assert meta.capture_roll == "Roll042"
        assert meta.capture_frame == 12

    def test_clears_are_bound_actions(self, sidebar: MetadataSidebar) -> None:
        from negpy.desktop.view.shortcut_registry import REGISTRY

        for action_id in ("metadata_clear_gear", "metadata_clear_process", "metadata_clear_scanning"):
            assert action_id in REGISTRY
            assert REGISTRY[action_id].default_key == ""


class TestGearInferFromFolder:
    def _sidebar_with_folder(self, monkeypatch, folder_name: str, **library_kwargs) -> MetadataSidebar:
        library = GearLibrary(**library_kwargs)
        monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
        controller = FakeController()
        controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
        controller.state.uploaded_files = [{"path": f"/scans/{folder_name}/frame001.tif"}]
        controller.state.selected_file_idx = 0
        return MetadataSidebar(controller)

    def test_infers_camera_and_film_stock_from_the_folder_name(self, monkeypatch) -> None:
        sidebar = self._sidebar_with_folder(
            monkeypatch,
            "04_om1_Fuji400_Japon",
            cameras=[Camera(id="c1", make="Olympus", model="OM-1")],
            film_stocks=[FilmStock(id="f1", manufacturer="Fuji", stock_name="400")],
        )
        sidebar.gear_infer_btn.click()
        meta = sidebar.state.config.metadata
        assert meta.camera_id == "c1"
        assert meta.film_stock_id == "f1"

    def test_does_not_overwrite_an_already_set_camera(self, monkeypatch) -> None:
        sidebar = self._sidebar_with_folder(
            monkeypatch,
            "04_om1_Fuji400_Japon",
            cameras=[Camera(id="c1", make="Olympus", model="OM-1"), Camera(id="c2", make="Nikon", model="FM2")],
            film_stocks=[FilmStock(id="f1", manufacturer="Fuji", stock_name="400")],
        )
        _set_metadata(sidebar, camera_id="c2")
        sidebar.gear_infer_btn.click()
        meta = sidebar.state.config.metadata
        assert meta.camera_id == "c2"
        assert meta.film_stock_id == "f1"

    def test_no_match_shows_a_status_warning_and_changes_nothing(self, monkeypatch) -> None:
        sidebar = self._sidebar_with_folder(
            monkeypatch,
            "unrelated_folder_name",
            cameras=[Camera(id="c1", make="Olympus", model="OM-1")],
        )
        sidebar.gear_infer_btn.click()
        assert sidebar.state.config.metadata.camera_id == ""
        sidebar.controller.set_status.assert_called_once()

    def test_no_open_file_shows_a_status_warning(self, monkeypatch) -> None:
        library = GearLibrary()
        monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
        controller = FakeController()
        controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
        sidebar = MetadataSidebar(controller)

        sidebar.gear_infer_btn.click()

        sidebar.controller.set_status.assert_called_once()

    def test_infers_iso_and_capture_date_from_the_folder_name(self, monkeypatch) -> None:
        sidebar = self._sidebar_with_folder(monkeypatch, "06_scala_50_2024-05-12_vietnam")
        sidebar.gear_infer_btn.click()
        meta = sidebar.state.config.metadata
        assert meta.film_iso == 50
        assert meta.capture_date == "2024-05-12"

    def test_does_not_overwrite_an_already_set_iso_or_capture_date(self, monkeypatch) -> None:
        sidebar = self._sidebar_with_folder(monkeypatch, "06_scala_50_2024-05-12_vietnam")
        _set_metadata(sidebar, film_iso=800, capture_date="1999-01-01")
        sidebar.gear_infer_btn.click()
        meta = sidebar.state.config.metadata
        assert meta.film_iso == 800
        assert meta.capture_date == "1999-01-01"

    def test_iso_from_folder_stays_off_once_a_stock_matches(self, monkeypatch) -> None:
        """A matched stock's own ISO reaches the frame through the gear pick, so the
        standalone folder-name guess never overrides it."""
        sidebar = self._sidebar_with_folder(
            monkeypatch,
            "05_penees_kodakgold200_tailandia",
            film_stocks=[FilmStock(id="f1", manufacturer="Kodak", stock_name="Gold 200", iso=200)],
        )
        sidebar.gear_infer_btn.click()
        meta = sidebar.state.config.metadata
        assert meta.film_stock_id == "f1"
        assert meta.film_iso == 200


def _card_titled(sidebar: MetadataSidebar, title: str) -> CollapsibleSection:
    for section in sidebar.findChildren(CollapsibleSection):
        if section.title_label.text() == title:
            return section
    raise AssertionError(f"no card titled {title!r}")


class TestCollapsibleCards:
    def test_every_card_collapses(self, sidebar: MetadataSidebar) -> None:
        for title in ("Metadata Preview", "Analog Gear", "Capture", "Process", "Scanning", "Exposure"):
            section = _card_titled(sidebar, title)
            assert section.collapsible is True
            assert section.chevron_label is not None

    def test_preview_header_click_collapses_it(self, sidebar: MetadataSidebar) -> None:
        sidebar.preview_section.toggle_button.click()
        assert sidebar.preview_section.content_area.isHidden() is True


@pytest.fixture
def gear(monkeypatch, tmp_path) -> GearLibraryPanel:
    monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
    return GearLibraryPanel(GearLibrary())


class TestGearCards:
    def test_gear_cards_get_their_own_tabs_and_stay_live_under_protect(self, sidebar: MetadataSidebar, gear) -> None:
        sidebar.attach_gear_panel(gear)

        _set_metadata(sidebar, protect_original_metadata=True)
        sidebar.sync_ui()

        assert _page(sidebar, "presets").isAncestorOf(gear.sections["presets"])
        assert _page(sidebar, "gear").isAncestorOf(gear.sections["items"])
        assert all(section.isEnabled() for section in gear.sections.values())
        assert not gear.presets.apply_btn.isEnabled()

    def test_apply_on_the_presets_tab_writes_onto_this_frame(self, sidebar: MetadataSidebar, gear, monkeypatch, tmp_path) -> None:
        from negpy.services.assets.presets import MetadataPresets

        monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path / "presets"))
        sidebar.attach_gear_panel(gear)
        MetadataPresets.save_preset("HP5", {"developer": "D-76"})

        gear.presets.apply_requested.emit("HP5")

        assert sidebar.state.config.metadata.developer == "D-76"

    def test_a_sync_never_unblocks_the_gear_cards(self, sidebar: MetadataSidebar, gear) -> None:
        sidebar.attach_gear_panel(gear)
        gear.items.item_list.blockSignals(True)

        sidebar.block_signals(True)
        sidebar.block_signals(False)

        assert gear.items.item_list.signalsBlocked()
