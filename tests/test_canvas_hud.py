from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget

from negpy.desktop.view.canvas.hud import CanvasHud
from negpy.desktop.view.styles.theme import THEME


def _hud(width=1200):
    """The parent is kept on the strip: a parentless strip that shows itself is a top-level window."""
    parent = QWidget()
    hud = CanvasHud(parent)
    hud.resize(width, hud.minimumHeight())
    hud._keepalive = parent
    return hud


def test_an_empty_strip_hides_so_the_canvas_keeps_its_height(qapp):
    hud = _hud()
    assert hud.isHidden()
    hud.update_info("roll.nef", "C-41")
    assert not hud.isHidden()
    hud.update_info("", "")
    assert hud.isHidden()


def test_update_info_fills_both_sides(qapp):
    hud = _hud()
    hud.update_info("roll.nef · 4032 x 3024 px", "C-41")
    assert hud.lbl_left.text() == "roll.nef · 4032 x 3024 px"
    assert hud.lbl_right.text() == "C-41"


def test_show_message_keeps_case_and_arms_timer(qapp):
    """A path or a filename in a message must stay as written."""
    hud = _hud()
    hud.showMessage("Settings Copied")
    assert hud.message.text() == "Settings Copied"
    assert not hud.message.isHidden()
    assert not hud.isHidden()
    assert hud._message_timer.interval() == 2500

    hud.showMessage("boom", timeout=1500)
    assert hud._message_timer.interval() == 1500


def test_an_expired_message_hides_an_otherwise_empty_strip(qapp):
    hud = _hud()
    hud.showMessage("Settings copied")
    hud._message_timer.timeout.emit()
    assert hud.message.isHidden()
    assert hud.isHidden()


def test_set_progress_show_hide(qapp):
    hud = _hud()
    hud.set_progress(1, 4)
    assert hud.progress_visible()
    assert not hud.isHidden()

    hud.set_progress(0, 0)
    assert not hud.progress_visible()

    hud.set_progress(2, 4)
    hud.hide_progress()
    assert not hud.progress_visible()


def _bar_pixels(hud):
    image = hud.grab().toImage()
    y = image.height() - 2
    return [image.pixelColor(x, y).name() for x in range(image.width())]


def test_a_short_batch_paints_one_segment_per_item(qapp):
    hud = _hud(width=400)
    hud.set_progress(1, 4)
    row = _bar_pixels(hud)
    fill, track = QColor(THEME.progress_fill).name(), QColor(THEME.border_primary).name()
    assert row[10] == fill and row[150] == track and row[390] == track
    assert row[99] not in (fill, track), "a gap separates the segments"


def test_a_long_job_paints_one_continuous_bar(qapp):
    hud = _hud(width=400)
    hud.set_progress(50, 100)
    row = _bar_pixels(hud)
    fill, track = QColor(THEME.progress_fill).name(), QColor(THEME.border_primary).name()
    assert set(row[:198]) == {fill}
    assert set(row[202:]) == {track}


def test_zoom_note_survives_a_metadata_refresh(qapp):
    hud = _hud()
    hud.set_zoom_note("preview res · HQ off")
    hud.update_info("roll.nef", "C-41")
    assert hud.lbl_right.text() == "C-41 · preview res · HQ off"


def test_zoom_note_alone_shows_the_strip(qapp):
    hud = _hud()
    hud.set_zoom_note("preview res · HQ off")
    assert not hud.isHidden()
    hud.set_zoom_note("")
    assert hud.isHidden()


def test_show_message_kind_colours_the_message(qapp):
    """A failure and a progress note must not look the same."""
    hud = _hud()
    hud.showMessage("settings pasted")
    info_qss = hud.message.styleSheet()
    hud.showMessage("Could not write /tmp/x.tif", kind="error")
    assert hud.message.styleSheet() != info_qss
    assert THEME.error_text.upper() in hud.message.styleSheet().upper()


def test_abort_shows_only_for_an_abortable_job(qapp):
    hud = _hud()
    fired = []
    hud.abort_requested.connect(lambda: fired.append(True))
    hud.set_abortable(True)
    assert not hud.abort_button.isHidden()
    hud.abort_button.click()
    assert fired == [True]
    assert not hud.abort_button.isEnabled()
    assert hud.message.text() == "Aborting…"
    hud.set_abortable(False)
    assert hud.abort_button.isHidden()
    assert hud.message.isHidden()


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

    assert frame_info(INFO_OFF, _config(), "roll.nef", "4032 x 3024 px", ["Paper"]) == ("", "")


def test_info_basic_is_name_size_and_the_brief_recipe():
    from negpy.desktop.view.canvas.hud import INFO_BASIC, frame_info

    recipe = ["Crystal Archive", "Print Density 1.10", "Grade ISO-R 2"]
    assert frame_info(INFO_BASIC, _config(), "roll.nef", "4032 x 3024 px", recipe, "3 / 36", "Crystal") == (
        "roll.nef · 4032 x 3024 px",
        "Crystal Archive · Print Density 1.10 · Grade ISO-R 2",
    )


def test_info_full_adds_gear_proof_and_export():
    from negpy.desktop.view.canvas.hud import INFO_FULL, frame_info

    cfg = _config(
        film="Portra 400", film_iso=400, push_pull=1, camera_make="Nikon", camera_model="Nikon F3", focal_length_mm=50.0, max_aperture=1.4
    )
    left, right = frame_info(INFO_FULL, cfg, "roll.nef", "4032 x 3024 px", ["Crystal Archive", "Snap +0.20"], "3 / 36", "Crystal")
    assert left == "3 / 36 · roll.nef · 4032 x 3024 px · Portra 400 @ EI 800 · Nikon F3 · 50mm f/1.4"
    assert right == "Crystal Archive · Snap +0.20 · Print 30 cm @ 300 dpi · Proof: Crystal"


def test_info_full_skips_empty_metadata_lines():
    from negpy.desktop.view.canvas.hud import INFO_FULL, frame_info

    left, right = frame_info(INFO_FULL, _config(), "roll.nef", "", [])
    assert left == "roll.nef"
    assert right == "Print 30 cm @ 300 dpi"


def test_a_negative_timeout_holds_the_message_until_cleared(qapp):
    hud = _hud()
    hud.showMessage("Exporting: a.tif (1/3)", -1)
    assert not hud._message_timer.isActive()
    hud.clear_message("Exporting: a.tif (1/3)")
    assert hud.message.isHidden()


def test_clear_message_leaves_a_newer_message(qapp):
    hud = _hud()
    hud.showMessage("Processing…", -1)
    hud.showMessage("Export complete in 2.00s", 3000)
    hud.clear_message("Processing…")
    assert hud.message.text() == "Export complete in 2.00s"
    assert not hud.message.isHidden()


def test_the_strip_and_the_find_row_share_one_height(qapp):
    from negpy.desktop.view.styles.templates import TOP_ROW_HEIGHT
    from negpy.desktop.view.widgets.command_palette import FindField

    assert _hud().minimumHeight() == FindField(lambda: None).height() == TOP_ROW_HEIGHT


def test_a_bar_track_is_as_tall_as_a_bar_field():
    from negpy.desktop.view.styles.templates import BAR_BUTTON_HEIGHT, BAR_TRACK_BUTTON_HEIGHT, BAR_TRACK_PAD

    assert BAR_TRACK_BUTTON_HEIGHT + 2 * BAR_TRACK_PAD == BAR_BUTTON_HEIGHT
