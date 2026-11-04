"""The Film Strip's Apply starts a frame's placement (crop, rotation, flips) off."""

from dataclasses import replace
from unittest.mock import MagicMock, patch

from negpy.desktop.view.widgets import granular_settings_dialog as gsd
from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def _dialog():
    cfg = DEFAULT_WORKSPACE_CONFIG
    cfg = replace(cfg, geometry=replace(cfg.geometry, fine_rotation=1.2, crop_rect=(0.1, 0.1, 0.9, 0.9)))
    cfg = replace(cfg, exposure=replace(cfg.exposure, density=cfg.exposure.density + 0.1))
    dlg = gsd.GranularSettingsDialog(None, cfg, "a.tif", show_scope=True, sel_count=0, roll_count=3)
    return dlg


def _checked(dlg) -> dict:
    return {row.id: box.isChecked() for box, row, _edited, _line in dlg._checks}


def test_placement_rows_start_off_and_the_look_stays_on(qapp):
    dlg = _dialog()

    dlg.uncheck_rows(gsd._PLACEMENT_ROW_IDS)

    checked = _checked(dlg)
    assert not checked["geometry.crop_rect"] and not checked["geometry.fine_rotation"]
    assert checked["exposure.density"]


def test_only_the_full_picker_starts_placement_off():
    """A card's own Apply to Frames (Crop, Straighten) copies what the user asked for."""
    session = MagicMock()
    with (
        patch.object(gsd, "_apply_targets", return_value=(0, 0, 3)),
        patch.object(gsd, "GranularSettingsDialog") as dialog,
        patch.object(gsd, "_source_name", return_value="a.tif"),
    ):
        dialog.return_value.exec.return_value = 0
        gsd.open_apply_dialog(None, session)
        dialog.return_value.uncheck_rows.assert_called_once_with(gsd._PLACEMENT_ROW_IDS)

        dialog.reset_mock()
        dialog.return_value.exec.return_value = 0
        gsd.open_apply_dialog(None, session, rows=[MagicMock(id="geometry.crop_rect")])
        dialog.return_value.uncheck_rows.assert_not_called()
