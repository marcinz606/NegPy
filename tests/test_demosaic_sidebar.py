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
from negpy.features.process.models import DemosaicMode, ProcessMode

if not QApplication.instance():
    _app = QApplication(sys.argv)


def _sidebar(preview, export):
    process = replace(WorkspaceConfig().process, demosaic_preview=preview, demosaic_export=export)
    config = replace(WorkspaceConfig(), process=process)
    applied: list = []
    repo = SimpleNamespace(get_global_setting=lambda *a, **k: None, save_global_setting=lambda *a, **k: None)
    # No active roll (state.active_roll_id absent -> getattr below reads None): set_roll_default
    # falls back to a plain per-frame apply_config, same as demosaic always behaved before it.
    controller = SimpleNamespace(
        state=SimpleNamespace(config=config, active_roll_id=None, preview_cam_xyz=None, current_file_hash=None),
        session=SimpleNamespace(repo=repo, update_config=lambda *a, **k: None),
        apply_config=lambda cfg, **k: applied.append(cfg),
    )
    controller.set_roll_default = lambda card_key, persist=True, readback_metrics=True, **changes: controller.apply_config(
        replace(controller.state.config, process=replace(controller.state.config.process, **changes)),
        persist=persist,
        readback_metrics=readback_metrics,
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


def _wb_sidebar(linear_raw=True):
    from unittest.mock import MagicMock

    cfg = WorkspaceConfig()
    cfg = replace(cfg, process=replace(cfg.process, linear_raw=linear_raw))
    ctrl = MagicMock()
    ctrl.state.config = cfg
    ctrl.state.preview_cam_xyz = None
    return DemosaicSidebar(ctrl)


def _to_mode(w, **process):
    cfg = w.state.config
    w.state.config = replace(cfg, process=replace(cfg.process, **process))
    w.sync_ui()


def _to_rgbscan(w, **rgbscan):
    cfg = w.state.config
    w.state.config = replace(cfg, rgbscan=replace(cfg.rgbscan, **rgbscan))
    w.sync_ui()


def test_capture_row_stays_visible_and_greys_out_per_reason():
    """A hidden sticky setting follows a frame into Transparency unseen, so the row stays
    visible and greys out instead."""
    w = _wb_sidebar(linear_raw=True)
    w.sync_ui()
    assert w.linear_raw_btn.currentData() is True
    for widget in (w.capture_header, w.linear_raw_btn):
        assert not widget.isHidden()
        assert widget.isEnabled()
    assert w.capture_hint.isHidden()

    # Transparency transfer: Linear RAW is inert here.
    _to_mode(w, process_mode=ProcessMode.E6)
    for widget in (w.capture_header, w.linear_raw_btn):
        assert not widget.isHidden()
    assert not w.linear_raw_btn.isEnabled()
    assert not w.capture_hint.isHidden()

    # Positive decodes on the source's own profile, so Linear RAW decides the decode once more.
    _to_mode(w, positive_source=True)
    assert w.linear_raw_btn.isEnabled()
    assert not w.capture_hint.isHidden()

    _to_mode(w, process_mode=ProcessMode.C41)
    assert w.linear_raw_btn.isEnabled()
    assert w.capture_hint.isHidden()


def test_linear_raw_unlocks_on_the_transfer_with_positive_source_on():
    """Positive on the transfer makes effective_linear_raw read the stored flag again
    (see effective_linear_raw), so a stale Linear RAW left on from a real scan would
    silently defeat Positive's own decode. Must be live, not greyed, so the user can see
    and clear it — the old \"inert either way\" reasoning stopped being true."""
    w = _wb_sidebar(linear_raw=True)
    _to_mode(w, process_mode=ProcessMode.E6)
    assert not w.linear_raw_btn.isEnabled()

    _to_mode(w, positive_source=True)
    assert w.linear_raw_btn.isEnabled()
    assert w.linear_raw_btn.currentData() is True

    _to_mode(w, positive_source=False)
    assert not w.linear_raw_btn.isEnabled()


def test_linear_raw_locks_for_an_rgb_scan_triplet():
    """Every triplet exposure decodes neutral regardless of this choice (see
    tests/test_rgbscan_white_balance.py), so it locks rather than sitting live with no effect."""
    w = _wb_sidebar(linear_raw=True)
    w.sync_ui()
    assert w.linear_raw_btn.isEnabled()
    assert w.capture_hint.isHidden()

    _to_rgbscan(w, enabled=True, green_path="/x/g.arw", blue_path="/x/b.arw")
    assert w.linear_raw_btn.isEnabled() is False
    assert w.linear_raw_btn.currentData() is True  # the sticky value is still shown, just not editable
    assert not w.capture_hint.isHidden()

    _to_rgbscan(w, enabled=False)
    assert w.linear_raw_btn.isEnabled()
    assert w.capture_hint.isHidden()


def test_the_capture_choice_reaches_the_controller():
    """A Raw Decode-card, roll-eligible field -- set_roll_default, not a bare
    apply_config, so a roll (if any) and its other frames pick the change up too."""
    w = _wb_sidebar(linear_raw=False)
    w.sync_ui()
    assert w.linear_raw_btn.currentData() is False

    w.linear_raw_btn.setCurrentIndex(w.linear_raw_btn.findData(True))
    args, kwargs = w.controller.set_roll_default.call_args
    assert args[0] == "demosaic"
    assert kwargs["linear_raw"] is True

    w.linear_raw_btn.setCurrentIndex(w.linear_raw_btn.findData(False))
    _args, kwargs = w.controller.set_roll_default.call_args
    assert kwargs["linear_raw"] is False
