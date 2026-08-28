"""Paste picker's normalization-bounds row: shown only for a clipboard that carries
per-frame bounds, and opt-out."""

from dataclasses import replace

from PyQt6.QtWidgets import QLabel

from negpy.desktop.view.widgets.granular_settings_dialog import GranularSettingsDialog
from negpy.domain.models import WorkspaceConfig

_FLOORS = (0.1, 0.2, 0.3)
_CEILS = (0.8, 0.85, 0.9)


def _cfg(**process_kwargs) -> WorkspaceConfig:
    c = WorkspaceConfig()
    return replace(c, process=replace(c.process, **process_kwargs))


def _bounds_section(dlg) -> bool:
    return dlg._bounds_local is not None


def test_no_bounds_row_without_bounds_mode(qapp):
    dlg = GranularSettingsDialog(None, _cfg(local_floors=_FLOORS, local_ceils=_CEILS), "clipboard")
    assert not _bounds_section(dlg)
    assert not dlg.paste_bounds()


def test_bounds_row_is_shown_and_ticked(qapp):
    cfg = _cfg(local_floors=_FLOORS, local_ceils=_CEILS, lock_bounds=True)
    dlg = GranularSettingsDialog(None, cfg, "clipboard", bounds_mode="local")
    assert _bounds_section(dlg)
    assert dlg.paste_bounds()
    assert "locked" in dlg._bounds_local.text()


def test_bounds_row_reports_the_values(qapp):
    cfg = _cfg(local_floors=_FLOORS, local_ceils=_CEILS)
    dlg = GranularSettingsDialog(None, cfg, "clipboard", bounds_mode="local")
    labels = [w.text() for w in dlg.findChildren(QLabel)]
    assert "0.1 / 0.2 / 0.3 → 0.8 / 0.85 / 0.9" in labels
    assert "locked" not in dlg._bounds_local.text()


def test_bounds_row_can_be_unticked(qapp):
    cfg = _cfg(local_floors=_FLOORS, local_ceils=_CEILS)
    dlg = GranularSettingsDialog(None, cfg, "clipboard", bounds_mode="local")
    dlg._bounds_local.setChecked(False)
    assert not dlg.paste_bounds()


def test_bounds_row_alone_keeps_apply_enabled(qapp):
    cfg = _cfg(local_floors=_FLOORS, local_ceils=_CEILS)
    dlg = GranularSettingsDialog(None, cfg, "clipboard", bounds_mode="local")
    dlg._set_all_checked(False)
    assert not dlg.apply_btn.isEnabled()
    dlg._bounds_local.setChecked(True)
    assert dlg.apply_btn.isEnabled()


def test_axes_mode_keeps_the_roll_baseline_row(qapp):
    cfg = _cfg(local_floors=_FLOORS, local_ceils=_CEILS)
    dlg = GranularSettingsDialog(None, cfg, "current", show_scope=True, bounds_mode="axes", sel_count=1)
    assert dlg._bounds_luma is not None and dlg._bounds_color is not None
    assert dlg._bounds_local is None
    assert dlg.bounds_flags() == (False, False)
