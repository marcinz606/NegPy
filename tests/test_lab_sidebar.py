"""Offline test for the Lab sidebar's sharpen-method combo.

Guards the QVariant gotcha: combo items store the plain str (SharpenMethod.value)
but the config holds a StrEnum for pre-existing edits (missing key -> enum
default). findData(enum) returns -1, so the combo must look up by str.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

from dataclasses import replace
from types import SimpleNamespace

from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.lab import LabSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.lab.models import SharpenMethod

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _sidebar(sharpen_method=SharpenMethod.USM, meters=None, **lab_fields) -> LabSidebar:
    lab = replace(WorkspaceConfig().lab, sharpen_method=sharpen_method, **lab_fields)
    config = replace(WorkspaceConfig(), lab=lab)
    state = SimpleNamespace(config=config, auto_meters={"h": meters or {}}, current_file_hash="h")
    controller = SimpleNamespace(state=state, image_updated=SimpleNamespace(connect=lambda _slot: None))
    return LabSidebar(controller)


def test_enum_default_selects_unsharp_mask() -> None:
    # Pre-existing edit: no serialized key, field is the StrEnum default.
    sidebar = _sidebar(SharpenMethod.USM)
    assert sidebar.sharpen_method_combo.currentData() == "usm"


def test_enum_rl_selects_deconvolution() -> None:
    sidebar = _sidebar(SharpenMethod.RL)
    assert sidebar.sharpen_method_combo.currentData() == "rl"


def test_plain_str_from_loaded_edit_selects() -> None:
    # Edits saved after the feature carry a plain str through the flat dict.
    sidebar = _sidebar("usm")
    assert sidebar.sharpen_method_combo.currentData() == "usm"


def test_auto_masking_shows_the_measured_value_and_resets_to_it() -> None:
    sidebar = _sidebar(meters={"auto_masking": 0.31}, sharpen_auto_masking=True, sharpen_masking=0.19)
    sidebar._sync_masking()
    assert sidebar.auto_masking_btn.isChecked()
    assert abs(sidebar.sharpen_masking_slider.value() - 0.40) < 1e-6
    assert sidebar.sharpen_masking_slider.is_default(0.31)
    assert not sidebar.auto_masking_hint.isHidden()


def test_auto_masking_drag_stores_the_offset() -> None:
    sidebar = _sidebar(meters={"auto_masking": 0.31}, sharpen_auto_masking=True)
    stored = {}
    sidebar.update_config_section = lambda _section, **kw: stored.update(kw)
    sidebar._set_masking(0.5, persist=True)
    assert abs(stored["sharpen_masking"] - 0.29) < 1e-6
    sidebar._set_masking(0.31, persist=True)
    assert stored["sharpen_masking"] == 0.1


def test_masking_without_auto_hides_the_hint() -> None:
    sidebar = _sidebar(meters={"auto_masking": 0.31}, sharpen_masking=0.25)
    sidebar._sync_masking()
    assert abs(sidebar.sharpen_masking_slider.value() - 0.25) < 1e-6
    assert sidebar.auto_masking_hint.isHidden()


def test_a_slider_resets_to_the_shipped_default_not_the_config_it_was_built_from() -> None:
    # A carried (sticky) value at startup is an edit, as the card count reads it.
    sidebar = _sidebar(chroma_denoise=0.0)
    slider = sidebar.chroma_denoise_slider
    assert slider.value() == 0.0
    assert slider.default_value() == WorkspaceConfig().lab.chroma_denoise
    assert not slider._edited_dot.isHidden()


def test_skin_protection_and_the_sharpen_rail_gray_out_at_their_parents_neutral() -> None:
    sidebar = _sidebar(saturation=1.0, sharpen=0.0)
    sidebar.sync_ui()
    rail = (sidebar.sharpen_method_combo, sidebar.auto_masking_btn, sidebar.sharpen_radius_slider, sidebar.sharpen_masking_slider)
    assert not sidebar.skin_protection_slider.isEnabled()
    assert not any(w.isEnabled() for w in rail)

    sidebar.saturation_slider.setValue(1.3)
    sidebar.sharpen_slider.setValue(0.4)
    sidebar._sync_rails()
    assert sidebar.skin_protection_slider.isEnabled()
    assert all(w.isEnabled() for w in rail)
