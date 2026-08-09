"""The crosstalk editor's Process control: a matrix must reach the film it was made for.

The render gates the unmix on the process a profile declares, and the sidebar dropdown filters
on the same key. So a profile saved without one — or with the wrong one — is not merely
mislabelled: it silently never applies, which is the failure this control exists to prevent.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest
from PyQt6.QtWidgets import QApplication

from negpy.features.process.models import ProcessMode
from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.crosstalk import CrosstalkProfiles

_IDENTITY = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(APP_CONFIG, "crosstalk_dir", str(tmp_path))
    monkeypatch.setattr("negpy.services.assets.crosstalk.get_resource_path", lambda _: str(tmp_path / "_none"))


def _dialog(_app, select: str, process_mode=None):
    from negpy.desktop.view.widgets.crosstalk_editor_dialog import CrosstalkEditorDialog

    return CrosstalkEditorDialog(select, 0.5, process_mode)


def test_new_matrix_is_for_the_process_in_use(_app):
    """The whole point: a slide user must be able to build a matrix they can then select."""
    dlg = _dialog(_app, CrosstalkProfiles.DEFAULT_NAME, ProcessMode.E6)
    dlg._on_new()

    name = dlg.selected_name()
    assert CrosstalkProfiles.get_process(name) == str(ProcessMode.E6)
    assert dict(CrosstalkProfiles.grouped_profiles(ProcessMode.E6))["Tuned on a rig"] == [name]


def test_process_survives_a_save(_app):
    CrosstalkProfiles.save("Mine", _IDENTITY, process=str(ProcessMode.C41))
    dlg = _dialog(_app, "Mine")

    dlg._set_process(str(ProcessMode.E6))
    dlg._on_save()

    assert CrosstalkProfiles.get_process("Mine") == str(ProcessMode.E6)
    # It leaves the C-41 dropdown entirely; only the built-in is left there.
    c41 = [n for _, names in CrosstalkProfiles.grouped_profiles(ProcessMode.C41) for n in names]
    assert c41 == [CrosstalkProfiles.DEFAULT_NAME]


def test_selecting_a_profile_shows_its_process(_app):
    CrosstalkProfiles.save("Slide", _IDENTITY, process=str(ProcessMode.E6))
    dlg = _dialog(_app, "Slide")
    assert dlg.selected_process() == str(ProcessMode.E6)


def test_a_copy_inherits_the_process(_app):
    """The numbers describe one dye set; carrying them to another film is not a copy."""
    CrosstalkProfiles.save("Slide", _IDENTITY, process=str(ProcessMode.E6))
    dlg = _dialog(_app, "Slide", ProcessMode.C41)
    dlg._on_copy()

    assert CrosstalkProfiles.get_process(dlg.selected_name()) == str(ProcessMode.E6)


def test_unknown_process_falls_back_to_c41(_app, tmp_path):
    with open(tmp_path / "odd.toml", "w", encoding="utf-8") as f:
        f.write('name = "Odd"\nprocess = "K-14"\nmatrix = [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]\n')

    dlg = _dialog(_app, "Odd")
    assert dlg.selected_process() == str(ProcessMode.C41)


def test_preview_carries_the_process(_app):
    """Without it the sidebar's render gate discards the preview and the editor looks dead."""
    CrosstalkProfiles.save("Slide", _IDENTITY, process=str(ProcessMode.E6))
    dlg = _dialog(_app, "Slide")

    seen: list = []
    dlg.matrix_previewed.connect(lambda m, s, p: seen.append(p))
    dlg._emit_preview()

    assert seen == [str(ProcessMode.E6)]
