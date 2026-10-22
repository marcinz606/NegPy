from PyQt6.QtCore import Qt

from negpy.desktop.view.canvas.hud import CanvasHud


def test_hud_is_mouse_transparent(qapp):
    hud = CanvasHud()
    assert hud.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_update_info_sets_pills(qapp):
    hud = CanvasHud()
    hud.update_info("roll.nef · 4032 x 3024 px", "C-41", "Print 30 cm @ 300 dpi")
    assert hud.lbl_top_left.text() == "roll.nef · 4032 x 3024 px"
    assert hud.lbl_top_right.text() == "C-41"
    assert hud.lbl_bottom_left.text() == "Print 30 cm @ 300 dpi"
    assert not hud.lbl_top_left.isHidden()


def test_update_info_hides_empty_pills(qapp):
    hud = CanvasHud()
    hud.update_info("roll.nef", "C-41", "Print 30 cm @ 300 dpi")
    hud.update_info("", "", "")
    for lbl in (hud.lbl_top_left, hud.lbl_top_right, hud.lbl_bottom_left, hud.lbl_bottom_right):
        assert lbl.isHidden()


def test_show_message_keeps_case_and_arms_timer(qapp):
    """A path or a filename in a toast must stay as written."""
    hud = CanvasHud()
    hud.showMessage("Settings Copied")
    assert hud.toast.text() == "Settings Copied"
    assert not hud.toast.isHidden()
    assert hud._toast_timer.isActive()
    assert hud._toast_timer.interval() == 2500

    hud.showMessage("boom", timeout=1500)
    assert hud._toast_timer.interval() == 1500


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


def test_zoom_note_survives_a_metadata_refresh(qapp):
    hud = CanvasHud()
    hud.set_zoom_note("preview res · HQ off")
    hud.update_info("roll.nef", "", "")
    assert hud.lbl_bottom_right.text() == "preview res · HQ off"


def test_zoom_note_alone_shows_the_pill(qapp):
    hud = CanvasHud()
    hud.set_zoom_note("preview res · HQ off")
    assert not hud.lbl_bottom_right.isHidden()
    hud.set_zoom_note("")
    assert hud.lbl_bottom_right.isHidden()


def test_show_message_kind_colours_the_toast(qapp):
    """A failure and a progress note must not look the same."""
    hud = CanvasHud()
    hud.showMessage("settings pasted")
    info_qss = hud.toast.styleSheet()
    hud.showMessage("Could not write /tmp/x.tif", kind="error")
    assert hud.toast.styleSheet() != info_qss
    assert "D32F2F" in hud.toast.styleSheet().upper()


def test_frame_position_follows_the_film_strip_order_and_filter(qapp):
    from negpy.desktop.session import AppState, AssetListModel
    from negpy.desktop.view.canvas.toolbar import frame_position

    state = AppState()
    state.uploaded_files = [
        {"name": "c.tif", "path": "/r/c.tif", "hash": "h3"},
        {"name": "a.tif", "path": "/r/a.tif", "hash": "h1"},
        {"name": "b.tif", "path": "/r/b.tif", "hash": "h2"},
    ]
    model = AssetListModel(state)

    assert frame_position(model, 1) == "1 / 3"
    model.set_sort_descending(True)
    assert frame_position(model, 1) == "3 / 3"
    model.set_filter("b", regex=False)
    assert frame_position(model, 1) == ""
    assert frame_position(model, 2) == ""


def _config(**meta):
    from dataclasses import replace

    from negpy.domain.models import ExportResolutionMode, WorkspaceConfig

    cfg = WorkspaceConfig()
    return replace(
        cfg,
        metadata=replace(cfg.metadata, **meta),
        export=replace(cfg.export, export_resolution_mode=ExportResolutionMode.PRINT, export_print_size=30.0, export_dpi=300),
    )


def test_info_off_shows_nothing():
    from negpy.desktop.view.canvas.hud import INFO_OFF, frame_info

    assert frame_info(INFO_OFF, _config(), "roll.nef", "4032 x 3024 px", "C-41") == ("", "", "")


def test_info_basic_is_name_size_and_mode():
    from negpy.desktop.view.canvas.hud import INFO_BASIC, frame_info

    assert frame_info(INFO_BASIC, _config(), "roll.nef", "4032 x 3024 px", "C-41", "3 / 36", "Crystal") == (
        "roll.nef · 4032 x 3024 px",
        "C-41",
        "",
    )


def test_info_full_adds_gear_proof_and_export():
    from negpy.desktop.view.canvas.hud import INFO_FULL, frame_info

    cfg = _config(
        film="Portra 400", film_iso=400, push_pull=1, camera_make="Nikon", camera_model="Nikon F3", focal_length_mm=50.0, max_aperture=1.4
    )
    top_left, top_right, bottom_left = frame_info(INFO_FULL, cfg, "roll.nef", "4032 x 3024 px", "C-41", "3 / 36", "Crystal")
    assert top_left.split("\n") == ["3 / 36 · roll.nef · 4032 x 3024 px", "Portra 400 @ EI 800", "Nikon F3 · 50mm f/1.4"]
    assert top_right == "C-41\nProof: Crystal"
    assert bottom_left == "Print 30 cm @ 300 dpi"


def test_info_full_skips_empty_metadata_lines():
    from negpy.desktop.view.canvas.hud import INFO_FULL, frame_info

    top_left, top_right, _ = frame_info(INFO_FULL, _config(), "roll.nef", "", "C-41")
    assert top_left == "roll.nef"
    assert top_right == "C-41"
