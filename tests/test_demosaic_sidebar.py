"""Offline test for the Demosaic panel's combos.

Guards the loaded-edit case: a flat dict hands ProcessConfig a plain str, not the
StrEnum, so the combos must still select it and write back an enum.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

from dataclasses import replace
from types import SimpleNamespace

from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.demosaic import DemosaicSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.process.models import DemosaicMode

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _sidebar(preview, export):
    process = replace(WorkspaceConfig().process, demosaic_preview=preview, demosaic_export=export)
    config = replace(WorkspaceConfig(), process=process)
    applied: list = []
    repo = SimpleNamespace(get_global_setting=lambda *a, **k: None, save_global_setting=lambda *a, **k: None)
    controller = SimpleNamespace(
        state=SimpleNamespace(config=config),
        session=SimpleNamespace(repo=repo, update_config=lambda *a, **k: None),
        apply_config=lambda cfg, **k: applied.append(cfg),
    )
    return DemosaicSidebar(controller), applied


def test_defaults_show_auto() -> None:
    sidebar, _ = _sidebar(DemosaicMode.AUTO, DemosaicMode.AUTO)
    assert sidebar.preview_combo.currentText() == "Auto"
    assert sidebar.export_combo.currentText() == "Auto"


def test_plain_str_from_a_loaded_edit_selects() -> None:
    sidebar, _ = _sidebar("VNG", "AHD")
    sidebar.sync_ui()
    assert sidebar.preview_combo.currentText() == "VNG"
    assert sidebar.export_combo.currentText() == "AHD"


def test_an_unbuilt_algorithm_reads_back_as_auto() -> None:
    # AMAZE is a GPL pack absent from a permissive libraw build, so it has no combo row.
    sidebar, _ = _sidebar("AMAZE", DemosaicMode.AUTO)
    sidebar.sync_ui()
    assert sidebar.preview_combo.currentText() == "Auto"


def test_picking_an_algorithm_writes_the_enum() -> None:
    sidebar, applied = _sidebar(DemosaicMode.AUTO, DemosaicMode.AUTO)
    sidebar.export_combo.setCurrentText("DHT")
    assert applied and applied[-1].process.demosaic_export == DemosaicMode.DHT
    assert applied[-1].process.demosaic_preview == DemosaicMode.AUTO
