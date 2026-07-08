from PyQt6.QtCore import Qt

from negpy.desktop.view.canvas.hud import CanvasHud


def test_hud_is_mouse_transparent(qapp):
    hud = CanvasHud()
    assert hud.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_update_info_sets_pills(qapp):
    hud = CanvasHud()
    hud.update_info("roll.nef", "4032 x 3024 px", "C-41 | sRGB", "Edits: 7", "Crop", "3 / 24")
    assert hud.lbl_top_left.text() == "roll.nef · 4032 x 3024 px"
    assert hud.lbl_top_right.text() == "C-41 | sRGB"
    assert hud.lbl_bottom_left.text() == "Edits: 7 · Crop"
    assert hud.lbl_bottom_right.text() == "3 / 24"
    assert not hud.lbl_top_left.isHidden()
    assert not hud.lbl_bottom_right.isHidden()


def test_update_info_hides_empty_pills(qapp):
    hud = CanvasHud()
    hud.update_info("roll.nef", "4032 x 3024 px", "C-41 | sRGB", "Edits: 7", "Crop", "3 / 24")
    hud.update_info("", "", "", "", "", "")
    for lbl in (hud.lbl_top_left, hud.lbl_top_right, hud.lbl_bottom_left, hud.lbl_bottom_right):
        assert lbl.isHidden()


def test_update_info_omits_missing_tool_and_pos(qapp):
    hud = CanvasHud()
    hud.update_info("roll.nef", "4032 x 3024 px", "C-41 | sRGB", "Edits: 0", "", "")
    assert hud.lbl_bottom_left.text() == "Edits: 0"
    assert hud.lbl_bottom_right.isHidden()


def test_show_message_lowercases_and_arms_timer(qapp):
    hud = CanvasHud()
    hud.showMessage("Settings Copied")
    assert hud.toast.text() == "settings copied"
    assert not hud.toast.isHidden()
    assert hud._toast_timer.isActive()
    assert hud._toast_timer.interval() == 2500

    hud.showMessage("boom", timeout=1500)
    assert hud._toast_timer.interval() == 1500


def test_show_message_suppresses_image_updated(qapp):
    hud = CanvasHud()
    hud.showMessage("Image Updated")
    assert hud.toast.isHidden()


def test_set_progress_show_hide(qapp):
    hud = CanvasHud()
    hud.set_progress(1, 4)
    assert not hud.progress.isHidden()
    assert hud.progress.maximum() == 4
    assert hud.progress.value() == 1

    hud.set_progress(0, 0)
    assert hud.progress.isHidden()

    hud.set_progress(2, 4)
    hud.hide_progress()
    assert hud.progress.isHidden()
