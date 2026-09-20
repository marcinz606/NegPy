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


def test_right_click_toggle_sits_beside_optical_removal(qapp):
    _, sb = _sidebar()
    row = next(
        lay
        for i in range(sb.layout.count())
        if (lay := sb.layout.itemAt(i).layout()) is not None and any(lay.itemAt(j).widget() is sb.auto_dust_btn for j in range(lay.count()))
    )
    assert [row.itemAt(j).widget() for j in range(row.count())] == [sb.auto_dust_btn, sb.right_click_btn]
    assert sb.right_click_btn.isCheckable() and sb.right_click_btn.text() == "", "an icon-only toggle"


def test_right_click_toggle_reads_the_session_setting(qapp):
    controller, sb = _sidebar()
    controller.state.right_click_excludes = True
    sb.sync_ui()
    assert sb.right_click_btn.isChecked()

    controller.state.right_click_excludes = False
    sb.sync_ui()
    assert not sb.right_click_btn.isChecked()


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


def test_brush_size_shows_while_optical_removal_is_on(qapp):
    """The exclusion band is painted with no tool active, so the slider it reads has to be
    on screen there. Its keyboard steps are gated on the same visibility."""
    controller, sb = _sidebar()
    cfg = controller.state.config

    controller.state.config = replace(cfg, retouch=replace(cfg.retouch, dust_remove=False))
    sb.sync_ui()
    assert not sb.manual_size_slider.isVisibleTo(sb), "no tool and no detector, so nothing sizes a brush"

    controller.state.config = replace(cfg, retouch=replace(cfg.retouch, dust_remove=True))
    sb.sync_ui()
    assert sb.manual_size_slider.isVisibleTo(sb)


def test_brush_size_spans_the_shared_range(qapp):
    """One range for the slider, the canvas wheel and the pinch, so none can leave the
    others' bounds."""
    from negpy.features.retouch.models import HEAL_SIZE_MAX, HEAL_SIZE_MIN

    _, sb = _sidebar()
    sb.manual_size_slider.setValue(HEAL_SIZE_MAX + 40.0)
    assert sb.manual_size_slider.value() == HEAL_SIZE_MAX
    sb.manual_size_slider.setValue(HEAL_SIZE_MIN - 40.0)
    assert sb.manual_size_slider.value() == HEAL_SIZE_MIN
