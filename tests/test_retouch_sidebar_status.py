from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.retouch import RetouchSidebar


def _sidebar(*, has_ir: bool, enabled: bool) -> RetouchSidebar:
    controller = MagicMock()
    controller.state = AppState(has_ir=has_ir, current_file_hash="active-file")
    controller.state.config = replace(
        controller.state.config,
        retouch=replace(
            controller.state.config.retouch,
            ir_dust_remove=enabled,
        ),
    )
    return RetouchSidebar(controller)


def test_ir_status_reports_applied_mask_pixels(qapp) -> None:
    sidebar = _sidebar(has_ir=True, enabled=True)

    sidebar._on_metrics_available(
        {
            "source_hash": "active-file",
            "ir_dust_status": "applied",
            "ir_dust_mask_pixels": 12_311,
        }
    )

    assert sidebar.ir_status_label.text() == "IR repair applied · 12,311 pixels repaired"
    assert sidebar.ir_status_label.property("hint") == "muted"


def test_ir_status_makes_a_skipped_repair_visible(qapp) -> None:
    sidebar = _sidebar(has_ir=True, enabled=True)

    sidebar._on_metrics_available(
        {
            "source_hash": "active-file",
            "ir_dust_status": "skipped",
            "ir_dust_skipped": "IR support could not be identified safely",
        }
    )

    assert sidebar.ir_status_label.text() == (
        "IR repair skipped · IR support could not be identified safely"
    )
    assert sidebar.ir_status_label.property("hint") == "warning"


def test_ir_status_distinguishes_disabled_from_missing_ir(qapp) -> None:
    disabled = _sidebar(has_ir=True, enabled=False)
    disabled.sync_ui()
    assert disabled.ir_status_label.text() == "IR repair is off"

    unavailable = _sidebar(has_ir=False, enabled=True)
    unavailable.sync_ui()
    assert unavailable.ir_status_label.text() == "IR repair unavailable · no IR channel"


def test_ir_controls_restore_their_help_after_switching_from_a_non_ir_scan(qapp) -> None:
    sidebar = _sidebar(has_ir=False, enabled=True)
    assert sidebar.ir_dust_btn.toolTip() == "No IR channel in this scan"

    sidebar.state.has_ir = True
    sidebar.sync_ui()

    assert "scanner IR channel" in sidebar.ir_dust_btn.toolTip()
    assert "IR dust sensitivity" in sidebar.ir_threshold_slider.toolTip()


def test_ir_status_ignores_previous_file_metrics_until_current_render_succeeds(
    qapp,
) -> None:
    sidebar = _sidebar(has_ir=True, enabled=True)
    sidebar.state.current_file_hash = "previous-file"
    sidebar._on_metrics_available(
        {
            "source_hash": "previous-file",
            "ir_dust_status": "applied",
            "ir_dust_mask_pixels": 500,
        }
    )
    assert "500 pixels repaired" in sidebar.ir_status_label.text()

    sidebar.state.current_file_hash = "new-file"
    sidebar.state.last_metrics = {
        "source_hash": "previous-file",
        "ir_dust_status": "applied",
        "ir_dust_mask_pixels": 500,
    }
    sidebar.sync_ui()
    sidebar._on_metrics_available(
        {
            "source_hash": "previous-file",
            "ir_dust_status": "skipped",
            "ir_dust_skipped": "late result from previous frame",
        }
    )

    assert sidebar.ir_status_label.text() == "IR repair is on · waiting for render"
    assert "previous" not in sidebar.ir_status_label.toolTip()
