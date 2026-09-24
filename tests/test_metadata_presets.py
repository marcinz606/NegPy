"""Metadata presets: catalog coverage, the JSON namespace, and the panel's load path."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import fields, replace
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QDialog, QMessageBox

from conftest import FakeController
from negpy.desktop.settings_catalog import CATALOG, apply_selected_fields, preset_config, rows_for_keys, selected_flat_dict
from negpy.desktop.view.sidebar import metadata as metadata_module
from negpy.desktop.view.sidebar.metadata import MetadataSidebar
from negpy.desktop.view.widgets.gear_library_panel import GearLibraryPanel
from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.gear_models import Camera, DevelopmentProcess, FilmFormat, FilmStock, GearLibrary, ScanSetup
from negpy.desktop.view.shortcut_registry import REGISTRY
from negpy.features.metadata.capture import parse_dev_time, parse_temperature
from negpy.features.metadata.models import GEAR_FIELDS, MetadataConfig
from negpy.features.metadata.writer import _exif_ascii
from negpy.features.metadata.payload import build_metadata_payload
from negpy.services.assets.search import facts_for, match, parse_query
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.migrations import gear_presets as gear_preset_migration
from negpy.services.assets.gear import GearProfiles
from negpy.services.assets.migrations.gear_presets import migrate_gear_presets
from negpy.services.assets.presets import PRESET_NOTES_KEY, MetadataPresets, Presets, is_valid_preset_name, preset_notes, with_preset_notes

# A frame number belongs to one frame, so it is not offered as a preset field.
_UNPRESETABLE = {"capture_frame"}


class _FakeRepo:
    def __init__(self):
        self._settings: dict = {}

    def get_global_setting(self, key, default=None):
        return self._settings.get(key, default)

    def save_global_setting(self, key, value):
        self._settings[key] = value


def _metadata_rows():
    return [r for title, rows in CATALOG if title == "Metadata" for r in rows]


def test_catalog_covers_every_metadata_field():
    listed = [f for r in _metadata_rows() for f in r.fields]
    assert sorted(listed) == sorted(set(listed)), "a field is listed in two rows"
    assert set(listed) == {f.name for f in fields(MetadataConfig)} - _UNPRESETABLE


def test_gear_row_travels_as_one_unit():
    base = WorkspaceConfig()
    source = replace(
        base,
        metadata=replace(
            base.metadata,
            camera_id="c1",
            camera_make="Nikon",
            camera_model="F3",
            film_stock_id="f1",
            film="Kodak Portra 400",
            film_iso=400,
        ),
    )
    data = selected_flat_dict(source, [r for r in _metadata_rows() if r.label == "Analog Gear"])
    assert data["camera_id"] == "c1" and data["camera_make"] == "Nikon"
    assert data["film_iso"] == 400
    # Nothing outside the gear pick rides along.
    assert "developer" not in data and "scanning" not in data


def test_presets_and_metadata_presets_are_separate_namespaces():
    Presets.save_preset("Portra", {"density": 1.5})
    MetadataPresets.save_preset("Portra", {"developer": "D-76"})

    assert Presets.list_presets() == ["Portra"]
    assert MetadataPresets.list_presets() == ["Portra"]
    assert Presets.load_preset("Portra") == {"density": 1.5}
    assert MetadataPresets.load_preset("Portra") == {"developer": "D-76"}
    assert MetadataPresets.delete_preset("Portra") is True
    assert Presets.load_preset("Portra") == {"density": 1.5}


@pytest.fixture(autouse=True)
def presets_dir(monkeypatch, tmp_path):
    """Never touch the user's own preset store."""
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    return tmp_path


@pytest.fixture
def qapp_dialog_library(monkeypatch, tmp_path):
    """The library dialog on its Process page, over one saved recipe."""
    monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
    library = GearLibrary(
        processes=[DevelopmentProcess(id="p1", display_name="D-76", developer="D-76", time_seconds=570, temperature_c=20.0)]
    )
    dialog = GearLibraryPanel(library)
    dialog.items._select_category("processes")
    return dialog, library


@pytest.fixture
def sidebar(monkeypatch) -> MetadataSidebar:
    monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(GearLibrary))
    controller = FakeController()
    controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
    return MetadataSidebar(controller)


def test_load_writes_only_the_stored_fields(sidebar: MetadataSidebar) -> None:
    state = sidebar.state
    state.config = replace(state.config, metadata=replace(state.config.metadata, scanning="Flextight", developer="Rodinal"))
    MetadataPresets.save_preset("HP5", {"developer": "D-76 1+1", "push_pull": 1})
    sidebar._refresh_metadata_presets()
    sidebar.metadata_preset_combo.set_selected_id("HP5")

    sidebar._on_metadata_preset_load()

    assert state.config.metadata.developer == "D-76 1+1"
    assert state.config.metadata.push_pull == 1
    assert state.config.metadata.scanning == "Flextight"


def test_load_restores_gear_ids_with_resolved_values(sidebar: MetadataSidebar) -> None:
    base = WorkspaceConfig()
    source = replace(base, metadata=replace(base.metadata, camera_id="c1", camera_make="Nikon", camera_model="F3"))
    MetadataPresets.save_preset("F3", selected_flat_dict(source, [r for r in _metadata_rows() if r.label == "Analog Gear"]))
    sidebar._refresh_metadata_presets()
    sidebar.metadata_preset_combo.set_selected_id("F3")

    sidebar._on_metadata_preset_load()

    meta = sidebar.state.config.metadata
    assert (meta.camera_id, meta.camera_make, meta.camera_model) == ("c1", "Nikon", "F3")


def test_manage_saves_the_current_frame_as_a_preset(sidebar: MetadataSidebar) -> None:
    state = sidebar.state
    state.config = replace(state.config, metadata=replace(state.config.metadata, developer="D-76", scanning="Flextight"))
    dlg = MagicMock()
    dlg.exec.return_value = QDialog.DialogCode.Accepted
    dlg.name.return_value = "Dev only"
    dlg.selected.return_value = [r for r in _metadata_rows() if r.label == "Process"]
    library = GearLibraryPanel(GearLibrary(), current_config_fn=lambda: state.config)
    with patch("negpy.desktop.view.widgets.gear_library_panel.GranularSettingsDialog", return_value=dlg):
        library.presets._add_item()

    assert MetadataPresets.load_preset("Dev only") == {
        "developer": "D-76",
        "process_dilution": "",
        "push_pull": 0,
        "process_time_seconds": None,
        "process_temperature_c": None,
        "process_id": "",
    }


def test_manage_edit_renames_and_keeps_values() -> None:
    MetadataPresets.save_preset(
        "Old",
        {
            "developer": "D-76",
            "process_dilution": "",
            "push_pull": 1,
            "process_time_seconds": None,
            "process_temperature_c": None,
            "process_id": "",
        },
    )
    library = GearLibraryPanel(GearLibrary())
    dlg = MagicMock()
    dlg.exec.return_value = QDialog.DialogCode.Accepted
    dlg.name.return_value = "New"
    dlg.selected.return_value = [r for r in _metadata_rows() if r.label == "Process"]
    with patch("negpy.desktop.view.widgets.gear_library_panel.GranularSettingsDialog", return_value=dlg):
        library.presets._edit_preset()

    assert MetadataPresets.list_presets() == ["New"]
    assert MetadataPresets.load_preset("New") == {
        "developer": "D-76",
        "process_dilution": "",
        "push_pull": 1,
        "process_time_seconds": None,
        "process_temperature_c": None,
        "process_id": "",
    }


def test_manage_lists_presets_and_shows_the_selected_one() -> None:
    MetadataPresets.save_preset("HP5", {"developer": "D-76 1+1", "push_pull": 1, "scanning": "DSLR copy-stand"})
    library = GearLibraryPanel(GearLibrary())

    assert [library.presets.item_list.item(i).text() for i in range(library.presets.item_list.count())] == ["HP5"]
    assert library.presets.preset_name_label.text() == "HP5"
    # Stored rows with an editor are shown as fields, filled with the preset's own values.
    assert library.presets.preset_developer_edit.text() == "D-76 1+1"
    assert library.presets.preset_developer_edit.isVisibleTo(library.presets.preset_panel)
    assert library.presets.preset_scanning_edit.text() == "DSLR copy-stand"
    assert library.presets.preset_push_combo.currentText() == "Push +1"
    # Rows it does not store stay hidden.
    assert not library.presets.preset_roll_edit.isVisibleTo(library.presets.preset_panel)


def test_preset_fields_reach_other_frames_unchanged():
    """The rows a preset stores are the rows applied, whatever else the target holds."""
    data = {"developer": "D-76", "scanning": "Flextight"}
    base = WorkspaceConfig()
    target = replace(base, metadata=replace(base.metadata, developer="Rodinal", capture_roll="Roll042"))
    merged = apply_selected_fields(preset_config(data), target, rows_for_keys(data, "metadata"))
    assert merged.metadata.developer == "D-76"
    assert merged.metadata.scanning == "Flextight"
    assert merged.metadata.capture_roll == "Roll042"


def test_gear_presets_migrate_to_metadata_presets(monkeypatch, tmp_path):
    """A gear preset's three ids become the resolved gear fields, once."""
    monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
    os.makedirs(APP_CONFIG.gear_dir, exist_ok=True)
    with open(os.path.join(APP_CONFIG.gear_dir, "gear_presets.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": "p1", "displayName": "FM2 combo", "cameraId": "c1", "filmStockId": "f1"}], f)

    library = GearLibrary(
        cameras=[Camera(id="c1", make="Nikon", model="FM2")],
        film_stocks=[FilmStock(id="f1", manufacturer="Ilford", stock_name="HP5+", iso=400)],
    )
    monkeypatch.setattr(gear_preset_migration.GearProfiles, "load_library", staticmethod(lambda: library))
    monkeypatch.setattr(gear_preset_migration, "get_resource_path", lambda _p: str(tmp_path / "bundled"))

    repo = _FakeRepo()
    migrate_gear_presets(repo)

    stored = MetadataPresets.load_preset("FM2 combo")
    assert stored is not None
    assert stored["camera_id"] == "c1" and stored["camera_make"] == "Nikon"
    assert stored["film"] == "Ilford HP5+" and stored["film_iso"] == 400
    assert set(stored) == set(GEAR_FIELDS)

    # Second run is a no-op, and never overwrites a preset the user has since edited.
    MetadataPresets.save_preset("FM2 combo", {"developer": "D-76"})
    migrate_gear_presets(repo)
    assert MetadataPresets.load_preset("FM2 combo") == {"developer": "D-76"}


def test_migration_skips_a_name_already_taken(monkeypatch, tmp_path):
    monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
    os.makedirs(APP_CONFIG.gear_dir, exist_ok=True)
    with open(os.path.join(APP_CONFIG.gear_dir, "gear_presets.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": "p1", "displayName": "Mine", "cameraId": "c1"}], f)
    monkeypatch.setattr(gear_preset_migration.GearProfiles, "load_library", staticmethod(GearLibrary))
    monkeypatch.setattr(gear_preset_migration, "get_resource_path", lambda _p: str(tmp_path / "bundled"))
    MetadataPresets.save_preset("Mine", {"scanning": "Flextight"})

    migrate_gear_presets(_FakeRepo())

    assert MetadataPresets.load_preset("Mine") == {"scanning": "Flextight"}


def test_process_and_scan_setups_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
    GearProfiles._library_cache = None
    GearProfiles.save_library(
        GearLibrary(
            processes=[DevelopmentProcess(id="p1", display_name="D-76 1+1, push +1", developer="D-76 1+1", push_pull=1)],
            scan_setups=[ScanSetup(id="s1", display_name="Copy-stand · Z7", scanning="DSLR copy-stand")],
        )
    )
    loaded = GearProfiles.load_library()

    assert loaded.get_process("p1").developer == "D-76 1+1"
    assert loaded.get_process("p1").push_pull == 1
    assert loaded.get_scan_setup("s1").scanning == "DSLR copy-stand"


def _library_with_process() -> GearLibrary:
    return GearLibrary(
        processes=[DevelopmentProcess(id="p1", display_name="D-76 1+1, push +1", developer="D-76 1+1", push_pull=1)],
        scan_setups=[ScanSetup(id="s1", display_name="Copy-stand · Z7", scanning="DSLR copy-stand")],
    )


def test_picking_a_process_fills_developer_and_push(monkeypatch) -> None:
    library = _library_with_process()
    monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
    controller = FakeController()
    controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
    sidebar = MetadataSidebar(controller)

    sidebar.process_combo.set_selected_id("p1")
    sidebar._on_process_selected()

    meta = sidebar.state.config.metadata
    assert (meta.process_id, meta.developer, meta.push_pull) == ("p1", "D-76 1+1", 1)

    sidebar.scan_setup_combo.set_selected_id("s1")
    sidebar._on_scan_setup_selected()
    assert sidebar.state.config.metadata.scanning_id == "s1"
    assert sidebar.state.config.metadata.scanning == "DSLR copy-stand"


def test_typing_over_a_filled_value_unlinks_the_saved_entry(monkeypatch) -> None:
    library = _library_with_process()
    monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
    controller = FakeController()
    controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
    sidebar = MetadataSidebar(controller)
    sidebar.process_combo.set_selected_id("p1")
    sidebar._on_process_selected()

    sidebar.developer_edit.setText("Rodinal 1+50")
    sidebar._persist_all_metadata_settings()

    meta = sidebar.state.config.metadata
    assert meta.process_id == ""
    assert meta.developer == "Rodinal 1+50"


class TestDevelopmentTime:
    def test_picking_a_process_fills_time_and_temperature(self, monkeypatch) -> None:
        library = GearLibrary(
            processes=[
                DevelopmentProcess(id="p1", display_name="D-76 1+1", developer="D-76", dilution="1+1", time_seconds=570, temperature_c=20.0)
            ]
        )
        monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
        controller = FakeController()
        controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
        sidebar = MetadataSidebar(controller)

        sidebar.process_combo.set_selected_id("p1")
        sidebar._on_process_selected()

        assert sidebar.state.config.metadata.process_time_seconds == 570
        assert sidebar.state.config.metadata.process_temperature_c == 20.0
        assert sidebar.state.config.metadata.process_dilution == "1+1"
        assert sidebar.dev_time_edit.text() == "9:30"
        assert sidebar.dilution_edit.text() == "1+1"

    def test_typed_time_persists_as_seconds(self, sidebar: MetadataSidebar) -> None:
        sidebar.dev_time_edit.setText("11:15")
        sidebar.dev_temp_edit.setText("24.5")
        sidebar._persist_all_metadata_settings()

        assert sidebar.state.config.metadata.process_time_seconds == 675
        assert sidebar.state.config.metadata.process_temperature_c == 24.5

    def test_unreadable_time_is_flagged_and_not_persisted(self, sidebar: MetadataSidebar) -> None:
        sidebar.state.config = replace(
            sidebar.state.config,
            metadata=replace(sidebar.state.config.metadata, process_time_seconds=570),
        )
        sidebar.dev_time_edit.setText("1:75")
        assert sidebar.dev_time_edit.styleSheet() != ""
        sidebar._persist_all_metadata_settings()

        assert sidebar.state.config.metadata.process_time_seconds == 570

    def test_search_matches_a_time_range(self):
        base = WorkspaceConfig()
        cfg = replace(
            base,
            metadata=replace(
                base.metadata,
                developer="D-76",
                process_dilution="1+1",
                process_time_seconds=570,
                process_temperature_c=20.0,
            ),
        )
        facts = facts_for({"name": "roll1-04.tif", "path": "/x/roll1-04.tif"}, cfg)

        assert match(parse_query("devtime:>=9"), facts)
        assert match(parse_query("devtime:<=10 temp:20"), facts)
        assert not match(parse_query("devtime:>12"), facts)
        assert match(parse_query("developer:d-76 devtime:>=9"), facts)
        assert match(parse_query("dilution:1+1"), facts)
        assert not match(parse_query("dilution:1+50"), facts)

    def test_search_skips_a_frame_with_no_time(self):
        facts = facts_for({"name": "x.tif", "path": "/x.tif"}, WorkspaceConfig())
        assert not match(parse_query("devtime:>=1"), facts)
        assert not match(parse_query("temp:20"), facts)

    def test_time_reaches_the_export_payload(self):
        base = WorkspaceConfig()
        meta = replace(
            base.metadata,
            developer="D-76",
            process_dilution="1+1",
            process_time_seconds=570,
            process_temperature_c=20.0,
        )
        payload = build_metadata_payload(meta, GearLibrary(), None)

        assert payload.development_time == "9:30"
        assert payload.development_temperature == "20 °C"
        rows = dict(next(rows for title, rows in payload.to_preview_sections() if title == "Process"))
        assert rows["Development time"] == "9:30"
        assert rows["Temperature"] == "20 °C"
        assert rows["Dilution"] == "1+1"

    def test_dilution_joins_the_developer_in_the_image_description(self):
        base = WorkspaceConfig()
        meta = replace(
            base.metadata,
            developer="D-76",
            process_dilution="1+1",
            description_fields=("developer",),
        )
        payload = build_metadata_payload(meta, GearLibrary(), None)

        assert payload.developer_display() == "D-76 1+1"
        assert "D-76 1+1" in payload.image_description


class TestDilution:
    def test_picking_a_process_fills_dilution(self, monkeypatch) -> None:
        library = GearLibrary(processes=[DevelopmentProcess(id="p1", display_name="HC-110 B", developer="HC-110", dilution="1+31")])
        monkeypatch.setattr(metadata_module.GearProfiles, "load_library", staticmethod(lambda: library))
        controller = FakeController()
        controller.session.update_config = lambda config, **_kwargs: setattr(controller.state, "config", config)
        sidebar = MetadataSidebar(controller)

        sidebar.process_combo.set_selected_id("p1")
        sidebar._on_process_selected()

        assert sidebar.state.config.metadata.process_dilution == "1+31"
        assert sidebar.dilution_edit.text() == "1+31"

    def test_typed_dilution_persists_and_unlinks(self, sidebar: MetadataSidebar) -> None:
        sidebar.dilution_edit.setText("1+50")
        sidebar._persist_all_metadata_settings()

        assert sidebar.state.config.metadata.process_dilution == "1+50"
        assert sidebar.state.config.metadata.process_id == ""

    def test_search_matches_a_dilution(self):
        base = WorkspaceConfig()
        cfg = replace(base, metadata=replace(base.metadata, developer="HC-110", process_dilution="1+31"))
        facts = facts_for({"name": "a.tif", "path": "/a.tif"}, cfg)

        assert match(parse_query("dilution:1+31"), facts)
        assert match(parse_query("developer:hc-110 dilution:1+31"), facts)
        assert not match(parse_query("dilution:1+50"), facts)

    def test_dilution_reaches_the_export_payload(self):
        base = WorkspaceConfig()
        meta = replace(base.metadata, developer="D-76", process_dilution="1+1")
        payload = build_metadata_payload(meta, GearLibrary(), None)

        assert payload.dilution == "1+1"
        assert payload.developer_display() == "D-76 1+1"
        # The joined form is for the prose description; the table gives the dilution its own row.
        rows = dict(next(rows for title, rows in payload.to_preview_sections() if title == "Process"))
        assert rows["Developer"] == "D-76"


class TestReviewFixes:
    def test_migration_stays_pending_when_a_source_is_damaged(self, monkeypatch, tmp_path):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        os.makedirs(APP_CONFIG.gear_dir, exist_ok=True)
        path = os.path.join(APP_CONFIG.gear_dir, "gear_presets.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ this is not json")
        monkeypatch.setattr(gear_preset_migration.GearProfiles, "load_library", staticmethod(GearLibrary))
        monkeypatch.setattr(gear_preset_migration, "get_resource_path", lambda _p: str(tmp_path / "bundled"))
        repo = _FakeRepo()

        migrate_gear_presets(repo)
        assert repo.get_global_setting("gear_presets_migrated") is None
        assert MetadataPresets.list_presets() == []

        # Repaired on a later launch: the presets still convert.
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"id": "p1", "displayName": "FM2 combo", "cameraId": "c1"}], f)
        migrate_gear_presets(repo)

        assert repo.get_global_setting("gear_presets_migrated") is True
        assert MetadataPresets.list_presets() == ["FM2 combo"]

    def test_missing_source_still_completes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        monkeypatch.setattr(gear_preset_migration.GearProfiles, "load_library", staticmethod(GearLibrary))
        monkeypatch.setattr(gear_preset_migration, "get_resource_path", lambda _p: str(tmp_path / "bundled"))
        repo = _FakeRepo()

        migrate_gear_presets(repo)

        assert repo.get_global_setting("gear_presets_migrated") is True

    def test_parsers_reject_non_finite_input(self):
        for text in ("inf", "-inf", "nan", "1e400"):
            assert parse_dev_time(text) is None
            assert parse_temperature(text) is None
        assert parse_dev_time("9:30") == 570
        assert parse_temperature("20") == 20.0

    def test_exif_transliterates_the_degree_sign(self):
        assert _exif_ascii("Development Temperature: 20 °C") == b"Development Temperature: 20 C"

    def test_library_editor_keeps_a_value_while_the_time_is_half_typed(self, qapp_dialog_library):
        dialog, library = qapp_dialog_library

        dialog.items.dev_time_edit.setText("9:")
        assert library.processes[0].time_seconds == 570
        assert dialog.items.dev_time_edit.styleSheet() != ""

        dialog.items.dev_time_edit.setText("11:15")
        assert library.processes[0].time_seconds == 675
        assert dialog.items.dev_time_edit.styleSheet() == ""

    def test_library_editor_treats_blank_as_an_explicit_clear(self, qapp_dialog_library):
        dialog, library = qapp_dialog_library

        dialog.items.dev_time_edit.setText("")

        assert library.processes[0].time_seconds is None
        assert dialog.items.dev_time_edit.styleSheet() == ""

    def test_load_action_is_registered_and_dispatches(self):
        assert "metadata_preset_load" in REGISTRY
        assert REGISTRY["metadata_preset_load"].default_key == ""
        source = (Path(__file__).parent.parent / "negpy/desktop/view/keyboard_shortcuts.py").read_text()
        assert '"metadata_preset_load": lambda: right.metadata_sidebar.metadata_preset_load_btn.click()' in source


class TestFilmFormatTravelsWithGear:
    def test_gear_row_carries_the_stock_format(self):
        base = WorkspaceConfig()
        source = replace(base, metadata=replace(base.metadata, film_stock_id="f1", film="Kodak Portra 400", format="120"))
        data = selected_flat_dict(source, [r for r in _metadata_rows() if r.label == "Analog Gear"])

        assert data["format"] == "120"
        assert "format_other" in data

    def test_applying_a_120_preset_to_a_35mm_frame_updates_the_format(self):
        base = WorkspaceConfig()
        data = selected_flat_dict(
            replace(base, metadata=replace(base.metadata, film_stock_id="f120", film="Portra 400", format="120")),
            [r for r in _metadata_rows() if r.label == "Analog Gear"],
        )
        target = replace(base, metadata=replace(base.metadata, format="35mm", film="HP5+"))

        merged = apply_selected_fields(preset_config(data), target, rows_for_keys(data, "metadata"))

        assert merged.metadata.format == "120"
        assert merged.metadata.film == "Portra 400"

    def test_format_is_no_longer_a_row_of_its_own(self):
        assert [r.label for r in _metadata_rows() if r.label == "Format"] == []


class TestPresetNotes:
    def test_notes_round_trip_without_reaching_the_config(self):
        MetadataPresets.save_preset("HP5", with_preset_notes({"developer": "D-76"}, "  9 min, agitate 10s  "))
        data = MetadataPresets.load_preset("HP5")

        assert data[PRESET_NOTES_KEY] == "9 min, agitate 10s"
        assert preset_notes(data) == "9 min, agitate 10s"
        # The reserved key never reaches from_flat_dict, and never looks like a stored field.
        assert preset_config(data).metadata.developer == "D-76"
        assert [r.label for r in rows_for_keys(data, "metadata")] == ["Process"]

    def test_empty_notes_are_not_stored(self):
        MetadataPresets.save_preset("Bare", with_preset_notes({"developer": "D-76"}, "   "))
        assert MetadataPresets.load_preset("Bare") == {"developer": "D-76"}

    def test_migration_carries_gear_preset_notes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        os.makedirs(APP_CONFIG.gear_dir, exist_ok=True)
        with open(os.path.join(APP_CONFIG.gear_dir, "gear_presets.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": "p1", "displayName": "Street combo", "cameraId": "c1", "notes": "the beater body"}], f)
        monkeypatch.setattr(gear_preset_migration.GearProfiles, "load_library", staticmethod(GearLibrary))
        monkeypatch.setattr(gear_preset_migration, "get_resource_path", lambda _p: str(tmp_path / "bundled"))

        migrate_gear_presets(_FakeRepo())

        assert preset_notes(MetadataPresets.load_preset("Street combo")) == "the beater body"

    def test_manage_pane_edits_notes_in_place(self, qapp_dialog_library):
        dialog, _library = qapp_dialog_library
        MetadataPresets.save_preset("HP5", {"developer": "D-76"})
        dialog.presets._rebuild_item_list(select_id="HP5")

        dialog.presets.preset_notes_edit.setText("stand development")

        assert preset_notes(MetadataPresets.load_preset("HP5")) == "stand development"
        assert MetadataPresets.load_preset("HP5")["developer"] == "D-76"

    def test_changing_which_fields_a_preset_stores_keeps_its_notes(self, qapp_dialog_library):
        dialog, _library = qapp_dialog_library
        MetadataPresets.save_preset("HP5", with_preset_notes({"developer": "D-76"}, "keep me"))
        dialog.presets._rebuild_item_list(select_id="HP5")
        dlg = MagicMock()
        dlg.exec.return_value = QDialog.DialogCode.Accepted
        dlg.name.return_value = "HP5"
        dlg.selected.return_value = [r for r in _metadata_rows() if r.label == "Process"]
        with patch("negpy.desktop.view.widgets.gear_library_panel.GranularSettingsDialog", return_value=dlg):
            dialog.presets._edit_preset()

        assert preset_notes(MetadataPresets.load_preset("HP5")) == "keep me"


class TestSecondReviewRound:
    def test_editing_keeps_a_row_that_stores_a_default_value(self, qapp_dialog_library):
        dialog, _library = qapp_dialog_library
        # push_pull 0 is the default, and storing it deliberately means "develop normally".
        MetadataPresets.save_preset("Normal dev", {"developer": "D-76", "push_pull": 0, "process_id": ""})
        dialog.presets._rebuild_item_list(select_id="Normal dev")

        captured = {}

        def _capture(parent, cfg, name, **kwargs):
            dlg = GranularSettingsDialog(parent, cfg, name, **kwargs)
            captured["dlg"] = dlg
            dlg.exec = lambda: QDialog.DialogCode.Accepted
            return dlg

        with patch("negpy.desktop.view.widgets.gear_library_panel.GranularSettingsDialog", _capture):
            dialog.presets._edit_preset()

        assert "Process" in [r.label for r in captured["dlg"].selected()]
        assert captured["dlg"]._show_unchanged.isChecked(), "editing lists every row, not only edited ones"
        assert MetadataPresets.load_preset("Normal dev")["push_pull"] == 0
        assert MetadataPresets.load_preset("Normal dev")["developer"] == "D-76"

    def test_panel_keeps_the_stored_temperature_while_input_is_invalid(self, sidebar: MetadataSidebar) -> None:
        sidebar.state.config = replace(
            sidebar.state.config,
            metadata=replace(sidebar.state.config.metadata, process_temperature_c=20.0),
        )
        sidebar.dev_temp_edit.setText("twenty")
        assert sidebar.dev_temp_edit.styleSheet() != ""
        sidebar._persist_all_metadata_settings()

        assert sidebar.state.config.metadata.process_temperature_c == 20.0

    def test_panel_clears_the_temperature_when_blanked(self, sidebar: MetadataSidebar) -> None:
        sidebar.state.config = replace(
            sidebar.state.config,
            metadata=replace(sidebar.state.config.metadata, process_temperature_c=20.0),
        )
        sidebar.sync_ui()
        assert sidebar.dev_temp_edit.text() == "20"

        sidebar.dev_temp_edit.setText("")
        sidebar._persist_all_metadata_settings()

        assert sidebar.state.config.metadata.process_temperature_c is None

    def test_migration_will_not_overwrite_a_name_differing_only_in_case(self, monkeypatch, tmp_path):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        os.makedirs(APP_CONFIG.gear_dir, exist_ok=True)
        with open(os.path.join(APP_CONFIG.gear_dir, "gear_presets.json"), "w", encoding="utf-8") as f:
            json.dump([{"id": "p1", "displayName": "portra", "cameraId": "c1"}], f)
        monkeypatch.setattr(gear_preset_migration.GearProfiles, "load_library", staticmethod(GearLibrary))
        monkeypatch.setattr(gear_preset_migration, "get_resource_path", lambda _p: str(tmp_path / "bundled"))
        MetadataPresets.save_preset("Portra", {"scanning": "mine"})

        migrate_gear_presets(_FakeRepo())

        assert MetadataPresets.load_preset("Portra") == {"scanning": "mine"}
        assert sorted(n.casefold() for n in MetadataPresets.list_presets()) == ["portra"]

    def test_preset_writes_land_atomically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
        real_replace = os.replace
        seen = {}

        def _spy(src, dst):
            seen["tmp"] = src
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", _spy)
        MetadataPresets.save_preset("HP5", {"developer": "D-76"})

        assert seen["tmp"].endswith(".tmp")
        assert MetadataPresets.load_preset("HP5") == {"developer": "D-76"}
        assert [f for f in os.listdir(tmp_path / "metadata") if f.endswith(".tmp")] == []

    def test_load_tooltip_carries_its_binding(self, sidebar: MetadataSidebar) -> None:
        from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut

        expected = tooltip_with_shortcut("Write the selected preset's fields onto this frame", "metadata_preset_load")
        assert sidebar.metadata_preset_load_btn.toolTip() == expected


class TestPresetNames:
    def test_a_name_that_escapes_the_namespace_is_refused(self):
        for name in ("../escaped", "..", "a/b", "", "   ", ".hidden", "trailing."):
            assert not is_valid_preset_name(name)
            with pytest.raises(ValueError):
                MetadataPresets.save_preset(name, {"developer": "D-76"})

    def test_names_users_actually_type_are_accepted(self):
        for name in ("HP5 @ 800 · D-76", "Portra 400 (lab)", "6×7 120", "D-76 1+1"):
            assert is_valid_preset_name(name)
            MetadataPresets.save_preset(name, {"developer": "D-76"})
            assert MetadataPresets.load_preset(name) == {"developer": "D-76"}

    def test_a_case_only_rename_keeps_the_preset(self, presets_dir):
        MetadataPresets.save_preset("hp5", {"developer": "D-76"})

        assert MetadataPresets.rename_preset("hp5", "HP5") is True

        assert MetadataPresets.list_presets() == ["HP5"]
        assert MetadataPresets.load_preset("HP5") == {"developer": "D-76"}

    def test_exists_is_case_folded(self):
        MetadataPresets.save_preset("Portra", {"developer": "C-41"})
        assert MetadataPresets.exists("portra")
        assert MetadataPresets.exists("PORTRA")
        assert not MetadataPresets.exists("Velvia")

    def test_rename_refuses_an_unusable_target(self):
        MetadataPresets.save_preset("HP5", {"developer": "D-76"})
        assert MetadataPresets.rename_preset("HP5", "../evil") is False
        assert MetadataPresets.list_presets() == ["HP5"]

    def test_renaming_onto_another_preset_asks_first(self, qapp_dialog_library, monkeypatch):
        dialog, _library = qapp_dialog_library
        MetadataPresets.save_preset("Source", {"developer": "D-76"})
        MetadataPresets.save_preset("Existing", {"scanning": "Flextight"})
        dialog.presets._rebuild_item_list(select_id="Source")

        dlg = MagicMock()
        dlg.exec.return_value = QDialog.DialogCode.Accepted
        dlg.name.return_value = "Existing"
        dlg.selected.return_value = [r for r in _metadata_rows() if r.label == "Process"]
        monkeypatch.setattr(
            "negpy.desktop.view.widgets.gear_library_panel.QMessageBox.question",
            lambda *_a, **_k: QMessageBox.StandardButton.No,
        )
        with patch("negpy.desktop.view.widgets.gear_library_panel.GranularSettingsDialog", return_value=dlg):
            dialog.presets._edit_preset()

        # Declined: both presets survive untouched.
        assert sorted(MetadataPresets.list_presets()) == ["Existing", "Source"]
        assert MetadataPresets.load_preset("Existing") == {"scanning": "Flextight"}

    def test_load_tooltip_follows_a_rebinding(self, sidebar: MetadataSidebar, monkeypatch) -> None:
        import negpy.desktop.view.shortcut_registry as registry

        assert "Ctrl+Shift+L" not in sidebar.metadata_preset_load_btn.toolTip()
        monkeypatch.setattr(
            registry,
            "key_for",
            lambda action_id, bindings=None: "Ctrl+Shift+L" if action_id == "metadata_preset_load" else "",
        )

        sidebar.apply_shortcut_tooltips()

        assert "Ctrl+Shift+L" in sidebar.metadata_preset_load_btn.toolTip()


class TestEditingPresetValuesInTheLibrary:
    """Swapping a camera or a developer without opening a frame."""

    @pytest.fixture
    def dialog(self, monkeypatch, tmp_path):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        library = GearLibrary(
            cameras=[Camera(id="c1", make="Nikon", model="FM2"), Camera(id="c2", make="Canon", model="AE-1")],
            film_stocks=[FilmStock(id="f1", manufacturer="Ilford", stock_name="HP5+", iso=400, format=FilmFormat.FORMAT_120)],
            processes=[DevelopmentProcess(id="p1", display_name="HC-110 B", developer="HC-110", dilution="1+31", time_seconds=390)],
        )
        dlg = GearLibraryPanel(library)
        return dlg, library

    def _select(self, dlg, name):
        dlg.presets._rebuild_item_list(select_id=name)

    def test_swapping_the_camera_rewrites_the_resolved_values(self, dialog):
        dlg, _library = dialog
        MetadataPresets.save_preset("Kit", selected_flat_dict(WorkspaceConfig(), [r for r in _metadata_rows() if r.label == "Analog Gear"]))
        self._select(dlg, "Kit")

        dlg.presets.preset_camera_combo.set_selected_id("c2")
        dlg.presets._on_preset_gear_changed()

        stored = MetadataPresets.load_preset("Kit")
        assert stored["camera_id"] == "c2"
        assert stored["camera_make"] == "Canon"
        assert stored["camera_model"] == "AE-1"

    def test_picking_a_film_stock_carries_its_format(self, dialog):
        dlg, _library = dialog
        MetadataPresets.save_preset("Kit", selected_flat_dict(WorkspaceConfig(), [r for r in _metadata_rows() if r.label == "Analog Gear"]))
        self._select(dlg, "Kit")

        dlg.presets.preset_film_combo.set_selected_id("f1")
        dlg.presets._on_preset_gear_changed()

        stored = MetadataPresets.load_preset("Kit")
        assert stored["film"] == "Ilford HP5+"
        assert stored["film_iso"] == 400
        assert stored["format"] == "120"

    def test_picking_a_saved_process_fills_the_recipe(self, dialog):
        dlg, _library = dialog
        MetadataPresets.save_preset("Dev", selected_flat_dict(WorkspaceConfig(), [r for r in _metadata_rows() if r.label == "Process"]))
        self._select(dlg, "Dev")

        dlg.presets.preset_process_combo.set_selected_id("p1")
        dlg.presets._on_preset_process_picked()

        stored = MetadataPresets.load_preset("Dev")
        assert (stored["developer"], stored["process_dilution"]) == ("HC-110", "1+31")
        assert stored["process_time_seconds"] == 390
        assert stored["process_id"] == "p1"

    def test_typing_a_developer_unlinks_the_saved_process(self, dialog):
        dlg, _library = dialog
        MetadataPresets.save_preset(
            "Dev",
            {
                "developer": "HC-110",
                "process_dilution": "1+31",
                "push_pull": 0,
                "process_time_seconds": 390,
                "process_temperature_c": None,
                "process_id": "p1",
            },
        )
        self._select(dlg, "Dev")

        dlg.presets.preset_developer_edit.setText("Rodinal")

        stored = MetadataPresets.load_preset("Dev")
        assert stored["developer"] == "Rodinal"
        assert stored["process_id"] == ""

    def test_editing_never_adds_a_row_the_preset_does_not_store(self, dialog):
        dlg, _library = dialog
        MetadataPresets.save_preset("Dev", {"developer": "HC-110", "push_pull": 0, "process_id": ""})
        self._select(dlg, "Dev")

        dlg.presets.preset_developer_edit.setText("Rodinal")

        stored = MetadataPresets.load_preset("Dev")
        assert stored["developer"] == "Rodinal"
        assert "camera_id" not in stored
        assert "scanning" not in stored

    def test_a_half_typed_time_does_not_erase_the_stored_one(self, dialog):
        dlg, _library = dialog
        MetadataPresets.save_preset(
            "Dev",
            {
                "developer": "HC-110",
                "process_dilution": "",
                "push_pull": 0,
                "process_time_seconds": 390,
                "process_temperature_c": None,
                "process_id": "",
            },
        )
        self._select(dlg, "Dev")

        dlg.presets.preset_time_edit.setText("6:")

        assert MetadataPresets.load_preset("Dev")["process_time_seconds"] == 390
        assert dlg.presets.preset_time_edit.styleSheet() != ""


class TestPresetGearCombosDefaultToOwnGear:
    """Same mine-vs-catalog default as the Metadata tab's own combos, since this form
    edits the same camera_id / lens_id / ... fields."""

    @pytest.fixture
    def dialog(self, monkeypatch, tmp_path):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        library = GearLibrary(
            cameras=[
                Camera(id="c-bundled", make="Leica", model="M6", is_bundled=True),
                Camera(id="c-mine", make="Nikon", model="FM2", is_bundled=False),
            ]
        )
        dlg = GearLibraryPanel(library)
        MetadataPresets.save_preset("Kit", selected_flat_dict(WorkspaceConfig(), [r for r in _metadata_rows() if r.label == "Analog Gear"]))
        dlg.presets._rebuild_item_list(select_id="Kit")
        return dlg, library

    def test_camera_combo_defaults_to_personal_gear_only(self, dialog):
        dlg, _library = dialog
        from negpy.desktop.view.widgets.gear_library_panel import OTHER_ID

        ids = {item_id for _label, item_id, _search in dlg.presets.preset_camera_combo._entries}
        assert ids == {"c-mine", OTHER_ID}

    def test_other_pick_clones_the_catalog_camera_and_writes_through(self, dialog):
        dlg, _library = dialog
        from negpy.desktop.view.widgets.gear_library_panel import OTHER_ID

        cloned = Camera(id="c-cloned", make="Leica", model="M6", is_bundled=False)
        with patch("negpy.desktop.view.widgets.gear_library_panel.resolve_other_gear_pick", return_value=cloned):
            dlg.presets.preset_camera_combo._commit_id(OTHER_ID)

        stored = MetadataPresets.load_preset("Kit")
        assert stored["camera_id"] == "c-cloned"
        assert stored["camera_make"] == "Leica"
        assert any(c.id == "c-cloned" for c in _library.cameras)

    def test_other_pick_cancelled_reverts_to_no_selection(self, dialog):
        dlg, _library = dialog
        from negpy.desktop.view.widgets.gear_library_panel import OTHER_ID

        with patch("negpy.desktop.view.widgets.gear_library_panel.resolve_other_gear_pick", return_value=None):
            dlg.presets.preset_camera_combo._commit_id(OTHER_ID)

        assert dlg.presets.preset_camera_combo.selected_id() == ""


class TestMigrationNaming:
    """A name the store cannot take costs that one preset, never the presets after it."""

    @pytest.fixture
    def sources(self, monkeypatch, tmp_path):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        os.makedirs(APP_CONFIG.gear_dir, exist_ok=True)
        monkeypatch.setattr(gear_preset_migration.GearProfiles, "load_library", staticmethod(GearLibrary))
        monkeypatch.setattr(gear_preset_migration, "get_resource_path", lambda _p: str(tmp_path / "bundled"))

        def write(presets):
            with open(os.path.join(APP_CONFIG.gear_dir, "gear_presets.json"), "w", encoding="utf-8") as f:
                json.dump(presets, f)

        return write

    @pytest.mark.parametrize("display", ["Nikon F2.", ".portra", "...", "Two\nlines"])
    def test_an_awkward_name_does_not_strand_the_next_preset(self, sources, display):
        sources(
            [
                {"id": "p1", "displayName": display, "cameraId": "c1"},
                {"id": "p2", "displayName": "Later preset", "cameraId": "c1"},
            ]
        )
        repo = _FakeRepo()

        migrate_gear_presets(repo)

        # Whatever the first one is called, the one after it is never lost.
        assert "Later preset" in MetadataPresets.list_presets()
        assert len(MetadataPresets.list_presets()) == 2
        assert repo.get_global_setting("gear_presets_migrated") is True

    def test_a_separator_leaves_one_space_not_three(self, sources):
        sources([{"id": "p1", "displayName": "AE-1P / FD 50 f/1.4 / Portra 400"}])

        migrate_gear_presets(_FakeRepo())

        assert MetadataPresets.list_presets() == ["AE-1P FD 50 f 1.4 Portra 400"]

    def test_unnamed_presets_are_kept_and_numbered(self, sources):
        sources([{"id": "p1", "cameraId": "c1"}, {"id": "p2", "cameraId": "c2"}])

        migrate_gear_presets(_FakeRepo())

        assert sorted(MetadataPresets.list_presets()) == ["Unnamed preset", "Unnamed preset 2"]

    def test_a_refused_write_leaves_the_migration_pending(self, sources, monkeypatch):
        sources([{"id": "p1", "displayName": "Kit", "cameraId": "c1"}])
        monkeypatch.setattr(
            gear_preset_migration.MetadataPresets,
            "save_preset",
            staticmethod(lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full"))),
        )
        repo = _FakeRepo()

        migrate_gear_presets(repo)

        assert repo.get_global_setting("gear_presets_migrated") is None


class TestUnsetFormat:
    """An unset format is a row of its own, and must not read back as "Other"."""

    def test_the_panel_leaves_an_unset_format_alone(self, sidebar: MetadataSidebar) -> None:
        assert sidebar.state.config.metadata.format == ""

        sidebar.capture_roll_edit.setText("Roll042")
        sidebar._persist_all_metadata_settings()

        assert sidebar.state.config.metadata.format == ""

    def test_editing_a_preset_leaves_an_unset_format_alone(self, monkeypatch, tmp_path):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        gear_row = [r for r in _metadata_rows() if r.label == "Analog Gear"]
        MetadataPresets.save_preset("Body only", selected_flat_dict(WorkspaceConfig(), gear_row))
        assert MetadataPresets.load_preset("Body only")["format"] == ""

        dlg = GearLibraryPanel(GearLibrary(cameras=[Camera(id="c1", make="Nikon", model="FM2")]))
        dlg.presets._rebuild_item_list(select_id="Body only")

        assert dlg.presets.preset_format_combo.currentText() != "Other"
        dlg.presets._on_preset_value_changed()

        assert MetadataPresets.load_preset("Body only")["format"] == ""

    def test_a_real_format_still_round_trips(self, sidebar: MetadataSidebar) -> None:
        sidebar.format_combo.setCurrentText("120")
        sidebar._persist_all_metadata_settings()

        assert sidebar.state.config.metadata.format == "120"


class TestDeveloperNotes:
    def test_the_dilution_is_not_printed_twice(self):
        meta = replace(WorkspaceConfig().metadata, developer="D-76", process_dilution="1+50")
        payload = build_metadata_payload(meta, GearLibrary(), None)

        rows = dict(next(rows for title, rows in payload.to_preview_sections() if title == "Process"))

        assert rows["Developer"] == "D-76"
        assert rows["Dilution"] == "1+50"


class TestNewPresetWindow:
    def test_it_lists_every_row_not_only_the_edited_ones(self, monkeypatch, tmp_path):
        """A new preset is a choice of fields, so an unset one must be on offer too."""
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        base = WorkspaceConfig()
        cfg = replace(base, metadata=replace(base.metadata, developer="D-76"))
        library = GearLibraryPanel(GearLibrary(), current_config_fn=lambda: cfg)

        captured = {}

        def _capture(parent, config, name, **kwargs):
            dlg = GranularSettingsDialog(parent, config, name, **kwargs)
            captured["dlg"] = dlg
            dlg.exec = lambda: QDialog.DialogCode.Rejected
            return dlg

        with patch("negpy.desktop.view.widgets.gear_library_panel.GranularSettingsDialog", _capture):
            library.presets._add_item()

        dlg = captured["dlg"]
        assert dlg._show_unchanged.isChecked(), "the new-preset window lists every row"
        # Revealed, not ticked: only what the frame actually sets arrives selected.
        assert [r.label for r in dlg.selected()] == ["Process"]


class TestItemsPresetsSplit:
    """The Gear tab's My Gear/Presets sections, and what each side does and does not have."""

    def _panel(self, monkeypatch, tmp_path, **kwargs):
        monkeypatch.setattr(APP_CONFIG, "gear_dir", str(tmp_path / "gear"))
        return GearLibraryPanel(GearLibrary(), **kwargs)

    def test_both_sections_start_expanded(self, monkeypatch, tmp_path):
        panel = self._panel(monkeypatch, tmp_path)

        assert panel._sections["items"].content_area.isVisibleTo(panel) is True
        assert panel._sections["presets"].content_area.isVisibleTo(panel) is True

    def test_collapsing_one_section_leaves_the_other_open(self, monkeypatch, tmp_path):
        panel = self._panel(monkeypatch, tmp_path)

        panel._sections["items"].toggle_button.setChecked(False)

        assert panel._sections["items"].content_area.isVisibleTo(panel) is False
        assert panel._sections["presets"].content_area.isVisibleTo(panel) is True

    def test_items_category_picker_has_no_presets_entry(self, monkeypatch, tmp_path):
        panel = self._panel(monkeypatch, tmp_path)

        categories = [panel.items.category_list.itemData(i) for i in range(panel.items.category_list.count())]
        assert categories == ["cameras", "lenses", "film_stocks", "processes", "scan_setups"]

    def test_presets_has_no_category_picker_or_catalog_toggle(self, monkeypatch, tmp_path):
        panel = self._panel(monkeypatch, tmp_path)

        assert not hasattr(panel.presets, "category_list")
        assert not hasattr(panel.presets, "show_catalog_btn")

    def test_catalog_toggle_is_items_only(self, monkeypatch, tmp_path):
        panel = self._panel(monkeypatch, tmp_path)

        assert hasattr(panel.items, "show_catalog_btn")
        assert not hasattr(panel.presets, "show_catalog_btn")

    def test_personal_gear_added_on_items_reaches_an_open_presets_combo(self, monkeypatch, tmp_path):
        panel = self._panel(monkeypatch, tmp_path)
        MetadataPresets.save_preset("Kit", {})
        panel.presets._rebuild_item_list(select_id="Kit")

        panel.items._add_custom_item()
        added = panel.items._current_items()[-1]

        ids = {item_id for _label, item_id, _search in panel.presets.preset_camera_combo._entries}
        assert added.id in ids

    def test_show_section_by_key_expands_a_collapsed_section(self, monkeypatch, tmp_path):
        panel = self._panel(monkeypatch, tmp_path)
        panel._sections["presets"].toggle_button.setChecked(False)

        panel.show_section_by_key("presets")

        assert panel._sections["presets"].content_area.isVisibleTo(panel) is True

    def test_show_section_by_key_ignores_an_unknown_key(self, monkeypatch, tmp_path):
        panel = self._panel(monkeypatch, tmp_path)

        panel.show_section_by_key("bogus")

        assert panel._sections["items"].content_area.isVisibleTo(panel) is True

    def test_section_tooltips_carry_their_bound_shortcut(self, monkeypatch, tmp_path):
        # A bare, unmodified key: Ctrl/Shift combos render as platform symbols (e.g. macOS
        # shows ⇧⌘ glyphs), which a literal string match can't see through.
        import negpy.desktop.view.shortcut_registry as registry

        panel = self._panel(monkeypatch, tmp_path)
        monkeypatch.setattr(
            registry,
            "key_for",
            lambda action_id, bindings=None: "G" if action_id == "tab_gear_items" else "",
        )

        panel.apply_shortcut_tooltips()

        assert "G" in panel._sections["items"].toggle_button.toolTip()
        assert "G" not in panel._sections["presets"].toggle_button.toolTip()
