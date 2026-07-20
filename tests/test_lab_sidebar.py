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


def _sidebar(sharpen_method) -> LabSidebar:
    lab = replace(WorkspaceConfig().lab, sharpen_method=sharpen_method)
    config = replace(WorkspaceConfig(), lab=lab)
    controller = SimpleNamespace(state=SimpleNamespace(config=config))
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
