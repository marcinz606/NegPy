from dataclasses import replace
from unittest.mock import MagicMock

from negpy.desktop.session import AppState
from negpy.desktop.view.sidebar.retouch import RetouchSidebar
from negpy.features.retouch.models import RetouchConfig


def _sidebar():
    controller = MagicMock()
    controller.state = AppState()
    return controller, RetouchSidebar(controller)


def test_retouch_sidebar_builds_all_sections(qapp):
    _, sb = _sidebar()
    for name in ("auto_dust_btn", "pick_dust_btn", "pick_scratch_btn", "ir_dust_btn", "overlay_btn"):
        assert getattr(sb, name) is not None


def test_ir_tooltip_restores_after_ir_loads(qapp):
    """The stale 'No IR channel' tooltip must clear once a scan with IR loads
    (the bug: re-enabling read back the overwritten tooltip)."""
    controller, sb = _sidebar()
    controller.state.has_ir = False
    sb.sync_ui()
    assert sb.ir_dust_btn.toolTip() == "No IR channel in this scan"

    controller.state.has_ir = True
    sb.sync_ui()
    assert "recover the image" in sb.ir_dust_btn.toolTip().lower()


def test_ir_degenerate_shows_hint(qapp):
    controller, sb = _sidebar()
    controller.state.has_ir = True
    controller.state.ir_degenerate = True
    sb.sync_ui()
    assert "image content" in sb.ir_dust_btn.toolTip().lower()


def test_ir_buttons_unchecked_without_ir(qapp):
    """No IR plane → IR toggles show off (and disabled), never checked-but-greyed;
    the real state returns when an IR file loads again."""
    controller, sb = _sidebar()
    controller.state.config = replace(
        controller.state.config,
        retouch=RetouchConfig(ir_dust_remove=True, ir_attenuation=True),
    )
    controller.state.has_ir = False
    sb.sync_ui()
    assert not sb.ir_dust_btn.isChecked()
    assert not sb.ir_dust_btn.isEnabled()

    controller.state.has_ir = True
    sb.sync_ui()
    assert sb.ir_dust_btn.isChecked()


def test_manual_heal_count_label(qapp):
    controller, sb = _sidebar()
    controller.state.config = replace(
        controller.state.config,
        retouch=RetouchConfig(manual_heal_strokes=[([[0.5, 0.5]], 5.0, 0.0, 0.0)]),
    )
    sb.sync_ui()
    assert sb.heals_subheader.text() == "MANUAL HEAL · 1"
