"""Offline test for the Alternative Processes sidebar's sensitizer combo.

Same QVariant gotcha as the Lab sharpen-method combo: items store the plain str
(Sensitizer.value) while the config holds a StrEnum, and findData(enum) returns
-1. Without the str lookup the combo snapped back to Classic on every re-sync,
so picking New looked like it did nothing.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

from dataclasses import replace
from types import SimpleNamespace

from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.sidebar.altprocess import AltProcessSidebar
from negpy.domain.models import WorkspaceConfig
from negpy.features.altprocess.models import AltProcess, Sensitizer
from negpy.features.process.models import ProcessMode

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _sidebar(sensitizer, alt_process=AltProcess.CYANOTYPE) -> AltProcessSidebar:
    base = WorkspaceConfig()
    config = replace(
        base,
        process=replace(base.process, process_mode=ProcessMode.BW),
        altproc=replace(base.altproc, alt_process=alt_process, cyano_sensitizer=sensitizer),
    )
    controller = SimpleNamespace(state=SimpleNamespace(config=config))
    return AltProcessSidebar(controller)


def test_enum_default_selects_classic() -> None:
    assert _sidebar(Sensitizer.CLASSIC).sensitizer_combo.currentData() == "classic"


def test_enum_new_selects_ware() -> None:
    assert _sidebar(Sensitizer.NEW).sensitizer_combo.currentData() == "new"


def test_resync_keeps_the_picked_sensitizer() -> None:
    sidebar = _sidebar(Sensitizer.NEW)
    sidebar.sync_ui()
    assert sidebar.sensitizer_combo.currentData() == "new"


def test_only_the_selected_process_shows_its_controls() -> None:
    cyano = _sidebar(Sensitizer.CLASSIC)
    cyano.sync_ui()
    assert cyano.cyano_block.isVisibleTo(cyano) and not cyano.lith_block.isVisibleTo(cyano)

    lith = _sidebar(Sensitizer.CLASSIC, alt_process=AltProcess.LITH)
    lith.sync_ui()
    assert lith.lith_block.isVisibleTo(lith) and not lith.cyano_block.isVisibleTo(lith)
    assert not lith.sabattier_block.isVisibleTo(lith)

    sab = _sidebar(Sensitizer.CLASSIC, alt_process=AltProcess.SABATTIER)
    sab.sync_ui()
    assert sab.sabattier_block.isVisibleTo(sab)
    assert not sab.lith_block.isVisibleTo(sab) and not sab.cyano_block.isVisibleTo(sab)


def test_neither_block_shows_outside_bw() -> None:
    base = WorkspaceConfig()
    config = replace(base, altproc=replace(base.altproc, alt_process=AltProcess.CYANOTYPE))
    sidebar = AltProcessSidebar(SimpleNamespace(state=SimpleNamespace(config=config)))
    sidebar.sync_ui()
    assert not sidebar.cyano_block.isVisibleTo(sidebar)
    assert not sidebar.mode_btn.isEnabled()
    for alt in (AltProcess.LITH, AltProcess.SABATTIER):
        config = replace(base, altproc=replace(base.altproc, alt_process=alt))
        sidebar = AltProcessSidebar(SimpleNamespace(state=SimpleNamespace(config=config)))
        sidebar.sync_ui()
        assert not sidebar.lith_block.isVisibleTo(sidebar) and not sidebar.sabattier_block.isVisibleTo(sidebar)


def test_a_color_frame_grays_the_panel_and_says_why() -> None:
    sidebar = _sidebar(Sensitizer.CLASSIC)
    sidebar.sync_ui()
    assert sidebar.mode_btn.isEnabled() and not sidebar.color_hint.isVisibleTo(sidebar)

    config = sidebar.state.config
    sidebar.controller.state.config = replace(config, process=replace(config.process, process_mode=ProcessMode.C41))
    sidebar.sync_ui()
    assert not sidebar.mode_btn.isEnabled() and sidebar.color_hint.isVisibleTo(sidebar)
