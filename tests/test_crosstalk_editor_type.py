"""The crosstalk editor's Type control: provenance must survive a round-trip.

A profile's `type` decides which group it lands in and how far the next person trusts it, so a
save that silently relabelled one would be worse than having no control at all.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import TYPE_MEASURED, TYPE_SPECSHEET, TYPE_TUNED, CrosstalkProfiles

_IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """User dir on tmp, bundled gallery pointed at nothing, or the real 16 show up and the
    assertions below stop being about the fixture."""
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(tmp_path / "_none"))


def _dialog(_app, select: str):
    from negpy.desktop.view.widgets.crosstalk_editor_dialog import CrosstalkEditorDialog

    return CrosstalkEditorDialog(select, 0.5)


def test_type_survives_a_save(_app, tmp_path):
    CrosstalkProfiles.save("Mine", _IDENTITY, TYPE_TUNED)
    dlg = _dialog(_app, "Mine")

    dlg._set_type(TYPE_MEASURED)
    dlg._on_save()

    assert CrosstalkProfiles.get_type("Mine") == TYPE_MEASURED
    assert dict(CrosstalkProfiles.grouped_profiles())["Measured"] == ["Mine"]


def test_selecting_a_profile_shows_its_type(_app):
    CrosstalkProfiles.save("Sheet", _IDENTITY, TYPE_SPECSHEET)
    dlg = _dialog(_app, "Sheet")
    assert dlg.selected_type() == TYPE_SPECSHEET


def test_unknown_type_falls_back_to_tuned_not_the_first_entry(_app, tmp_path):
    """A hand-written type must not be relabelled 'spec sheet' just because that entry sits
    first in the combo."""
    with open(tmp_path / "odd.toml", "w", encoding="utf-8") as f:
        f.write('name = "Odd"\ntype = "handed-down-by-owls"\nmatrix = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]\n')

    dlg = _dialog(_app, "Odd")
    assert dlg.selected_type() == TYPE_TUNED


def test_renaming_keeps_the_type(_app):
    CrosstalkProfiles.save("Before", _IDENTITY, TYPE_MEASURED)
    dlg = _dialog(_app, "Before")

    dlg.name_edit.setText("After")
    dlg._on_save()

    assert CrosstalkProfiles.get_type("After") == TYPE_MEASURED
    assert "Before" not in CrosstalkProfiles.list_profiles()
