from dataclasses import replace

from negpy.desktop.settings_catalog import all_rows, selected_flat_dict
from negpy.desktop.view.widgets.roll_settings_dialog import RollSettingsDialog
from negpy.domain.models import WorkspaceConfig
from negpy.features.metadata.gear_models import Camera, FilmStock, GearLibrary
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.presets import MetadataPresets


def _cfg(**metadata_fields) -> WorkspaceConfig:
    base = WorkspaceConfig()
    return replace(base, metadata=replace(base.metadata, **metadata_fields))


def _dialog(cfg: WorkspaceConfig, sel_count: int = 2, roll_count: int = 5) -> RollSettingsDialog:
    return RollSettingsDialog(None, cfg, GearLibrary(), sel_count=sel_count, roll_count=roll_count)


def test_groups_with_data_start_checked_others_do_not(qapp):
    dlg = _dialog(_cfg(capture_roll="Roll007", developer="D-76"))
    assert dlg._checks["Roll"].isChecked()
    assert dlg._checks["Process"].isChecked()
    assert not dlg._checks["Analog Gear"].isChecked()
    assert not dlg._checks["Capture Date"].isChecked()


def test_scope_defaults_to_whole_roll_and_reads_the_radios(qapp):
    dlg = _dialog(_cfg(), sel_count=3, roll_count=8)
    assert dlg.scope() == "roll"
    dlg._scope_radios.sel.setChecked(True)
    assert dlg.scope() == "selection"
    dlg._scope_radios.current.setChecked(True)
    assert dlg.scope() == "current"


def test_scope_falls_back_to_current_when_nothing_else_is_loaded(qapp):
    dlg = _dialog(_cfg(), sel_count=0, roll_count=0)
    assert dlg.scope() == "current"


def test_selected_rows_only_returns_checked_groups(qapp):
    dlg = _dialog(_cfg(capture_roll="Roll007"))
    assert [r.label for r in dlg.selected_rows()] == ["Roll"]
    dlg._checks["Capture Date"].setChecked(True)
    assert {r.label for r in dlg.selected_rows()} == {"Roll", "Capture Date"}


def test_selected_config_carries_edited_field_values(qapp):
    dlg = _dialog(_cfg())
    dlg.capture_roll_edit.setText("Roll042")
    dlg.developer_edit.setText("HC-110")
    out = dlg.selected_config()
    assert out.metadata.capture_roll == "Roll042"
    assert out.metadata.developer == "HC-110"


def test_selected_config_keeps_stored_value_for_unreadable_dev_time(qapp):
    dlg = _dialog(_cfg(process_time_seconds=390))
    dlg.dev_time_edit.setText("not a time")
    assert dlg.selected_config().metadata.process_time_seconds == 390


def test_load_preset_fills_fields_and_checks_its_rows(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "presets_dir", str(tmp_path))
    preset_cfg = _cfg(capture_roll="RollX", capture_date="2020-01-01")
    rows = [r for r in all_rows() if r.label in ("Roll", "Capture Date")]
    MetadataPresets.save_preset("smoke", selected_flat_dict(preset_cfg, rows))

    dlg = _dialog(_cfg())
    dlg.preset_combo.set_selected_id("smoke")
    dlg._on_load_preset()

    assert dlg._checks["Roll"].isChecked()
    assert dlg._checks["Capture Date"].isChecked()
    assert dlg.capture_roll_edit.text() == "RollX"
    assert dlg.selected_config().metadata.capture_roll == "RollX"


def test_load_preset_is_a_noop_when_nothing_is_selected(qapp):
    dlg = _dialog(_cfg(capture_roll="Roll007"))
    dlg._on_load_preset()
    assert dlg.capture_roll_edit.text() == "Roll007"


def test_apply_detected_gear_ticks_gear_and_fills_the_combos(qapp):
    camera = Camera(make="Olympus", model="Pen F")
    stock = FilmStock(manufacturer="Kodak", stock_name="Gold 200")
    dlg = RollSettingsDialog(None, _cfg(), GearLibrary(cameras=[camera], film_stocks=[stock]), sel_count=0, roll_count=5)

    dlg.apply_detected_gear(camera_id=camera.id, film_stock_id=stock.id)

    assert dlg._checks["Analog Gear"].isChecked()
    assert dlg.camera_combo.selected_id() == camera.id
    assert dlg.film_stock_combo.selected_id() == stock.id
    assert dlg.selected_config().metadata.camera_id == camera.id


def test_apply_detected_gear_with_nothing_detected_is_a_noop(qapp):
    dlg = _dialog(_cfg())
    dlg.apply_detected_gear(camera_id="", film_stock_id="")
    assert not dlg._checks["Analog Gear"].isChecked()
