from dataclasses import replace

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QObject, QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication

from negpy.desktop.view.widgets.contact_sheet_dialog import ContactSheetDialog
from negpy.domain.models import WorkspaceConfig
from negpy.services.export.contact_sheet_layout import DEFAULT_PAPER, ContactSheetSettings, SheetFormat
from negpy.services.export.contact_sheet_roll import PROOF_GRADE, FrameFacts, SheetFrame, SheetLook, StraightProof, straight_proof_config


class FakeTiles(QObject):
    tile_ready = pyqtSignal(int, int, object)

    def __init__(self):
        super().__init__()
        self.requests = []
        self.cancelled = []

    def render(self, frames):
        self.requests.append(frames)
        return len(self.requests)

    def cancel(self, generation):
        self.cancelled.append(generation)


def _frames(n=38, size=(3000, 2000), rejected=()):
    return [
        SheetFrame(
            {"path": f"/r/{i}.tif", "name": f"{i}.tif", "hash": str(i), "excluded": i in rejected},
            WorkspaceConfig(),
            FrameFacts(scan_size=size),
        )
        for i in range(n)
    ]


@pytest.fixture
def dialog():
    made = []

    def make(
        n=38,
        film=SheetFormat.FULL_FRAME,
        frame_size="6×6",
        settings=None,
        tiles=None,
        lane_busy=None,
        rejected=(),
        proof=None,
        scene_proof=None,
        frames=None,
    ):
        d = ContactSheetDialog(
            frames if frames is not None else _frames(n, rejected=rejected),
            film,
            frame_size,
            settings or ContactSheetSettings(),
            lambda fmt: SheetLook(),
            "ROLL 1",
            "/out",
            tiles=tiles,
            lane_busy=lane_busy,
            proof=proof,
            scene_proof=scene_proof,
        )
        d.resize(1000, 700)
        d.canvas.resize(600, 600)
        made.append(d)
        return d

    yield make
    for d in made:
        d.reject()


def _mouse(kind, pos, buttons=Qt.MouseButton.LeftButton):
    button = Qt.MouseButton.LeftButton if kind != QEvent.Type.MouseMove else Qt.MouseButton.NoButton
    return QMouseEvent(kind, pos, button, buttons, Qt.KeyboardModifier.NoModifier)


def _drag(canvas, start, end):
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, start))
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, end))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton))


def test_opens_on_the_saved_paper(dialog):
    d = dialog()
    assert d.settings() == ContactSheetSettings()
    assert d.paper_btn.text() == DEFAULT_PAPER.label
    plan = d.plan()
    assert (plan.frames_per_strip, len(plan.pages)) == (6, 1)


def test_dragging_the_right_edge_widens_the_paper_about_its_center(dialog):
    d = dialog()
    canvas = d.canvas
    rect = canvas.sheet_rect()
    scale = canvas.px_per_mm()
    _drag(canvas, QPointF(rect.right(), rect.center().y()), QPointF(rect.right() + 40 * scale / 2, rect.center().y()))
    width, height = d.settings().paper_width, d.settings().paper_height
    assert width == pytest.approx(DEFAULT_PAPER.width + 40, abs=1.0)
    assert height == DEFAULT_PAPER.height
    assert d.plan().frames_per_strip == 7
    assert d.paper_btn.text() == "Custom"


def test_a_drag_near_an_ilford_size_snaps_to_it(dialog):
    d = dialog(settings=ContactSheetSettings(paper_width=250.0, paper_height=300.0))
    canvas = d.canvas
    rect = canvas.sheet_rect()
    scale = canvas.px_per_mm()
    target = QPointF(
        rect.center().x() + (DEFAULT_PAPER.width / 2 + 0.6) * scale, rect.center().y() + (DEFAULT_PAPER.height / 2 - 0.7) * scale
    )
    _drag(canvas, rect.bottomRight(), target)
    assert (d.settings().paper_width, d.settings().paper_height) == (DEFAULT_PAPER.width, DEFAULT_PAPER.height)


def test_hovering_a_handle_shows_its_resize_cursor(dialog):
    d = dialog()
    canvas = d.canvas
    rect = canvas.sheet_rect()
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, QPointF(rect.left(), rect.center().y()), Qt.MouseButton.NoButton))
    assert canvas.cursor().shape() == Qt.CursorShape.SizeHorCursor
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, rect.topLeft(), Qt.MouseButton.NoButton))
    assert canvas.cursor().shape() == Qt.CursorShape.SizeFDiagCursor


def test_return_in_a_size_box_commits_without_exporting(dialog):
    d = dialog()
    accepted = []
    d.accepted.connect(lambda: accepted.append(True))
    spin = d.width_slider.spin
    spin.setFocus()
    spin.lineEdit().setText("260.0 mm")
    QApplication.sendEvent(spin, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
    assert accepted == []
    assert d.settings().paper_width == 260.0


def test_a_new_format_turns_the_paper_when_that_needs_fewer_sheets(dialog):
    d = dialog(n=9, film=SheetFormat.MEDIUM, frame_size="6×6")
    d.frame_size_btn.setCurrentIndex(3)  # 6×8
    assert d.film() == (SheetFormat.MEDIUM, "6×8")
    assert (d.settings().paper_width, d.settings().paper_height) == (DEFAULT_PAPER.height, DEFAULT_PAPER.width)
    assert len(d.plan().pages) == 1


def test_frame_size_only_shows_for_120(dialog):
    d = dialog()
    d.show()
    assert not d.frame_size_rail.isVisible()
    d.format_btn.setCurrentIndex(2)
    assert d.frame_size_rail.isVisible()


def test_resolution_over_the_pixel_budget_steps_down(dialog):
    d = dialog(settings=ContactSheetSettings(dpi=600))
    assert d.settings().dpi == 600
    d.paper_btn.setCurrentIndex(9)  # 40.6 × 50.8 cm
    assert d.settings().dpi == 300
    assert not d.dpi_btn.choice_menu.actions()[2].isEnabled()


def test_a_paper_that_holds_no_frame_disables_export(dialog):
    d = dialog(n=4, film=SheetFormat.MEDIUM, frame_size="6×17")
    assert not d.paper_btn.choice_menu.actions()[0].isEnabled()
    d._render()
    assert d.export_btn.isEnabled()


def test_export_stays_open_while_another_batch_runs(dialog):
    d = dialog(lane_busy=lambda: "Export is running; export once it finishes.")
    accepted = []
    d.accepted.connect(lambda: accepted.append(True))
    d.export_btn.click()
    assert accepted == []
    assert "running" in d.summary_label.text()


def test_tiles_fill_in_and_closing_cancels_them(dialog):
    tiles = FakeTiles()
    d = dialog(n=3, tiles=tiles)
    assert len(tiles.requests) == 1
    tiles.tile_ready.emit(1, 0, np.full((200, 300, 3), 90, np.uint8))
    tiles.tile_ready.emit(99, 1, np.full((200, 300, 3), 90, np.uint8))  # a stale generation
    assert d._tiles[0] is not None and d._tiles[1] is None
    d.reject()
    assert tiles.cancelled == [1]
    tiles.tile_ready.emit(1, 2, np.full((200, 300, 3), 90, np.uint8))
    assert d._tiles[2] is None


def test_roll_label_toggle_reaches_the_settings(dialog):
    d = dialog()
    d.label_btn.setChecked(False)
    assert d.settings().roll_label is False
    assert d.look().label == ""


def test_rejected_frames_start_left_out(dialog):
    d = dialog(n=8, rejected=(2, 5))
    assert [f.name for f in d.kept_frames()] == ["0.tif", "1.tif", "3.tif", "4.tif", "6.tif", "7.tif"]
    assert d.numbers() == (0, 1, 3, 4, 6, 7)
    d._render()
    assert "2 left out" in d.summary_label.text()


def test_picking_shows_every_frame_and_a_click_toggles_it(dialog):
    d = dialog(n=8, rejected=(2,))
    d.pick_btn.setChecked(True)
    d._render()
    canvas = d.canvas
    windows = dict(canvas._frames)
    assert set(windows) == set(range(8))
    center = canvas._frame_rect(windows[4]).center()
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, center))
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, canvas._frame_rect(windows[2]).center()))
    assert d.numbers() == (0, 1, 2, 3, 5, 6, 7)
    d.pick_btn.setChecked(False)
    d._render()
    assert [index for index, _rect in d.canvas._frames] == [0, 1, 2, 3, 5, 6, 7]


def test_clicks_only_pick_while_picking(dialog):
    d = dialog(n=4)
    d._render()
    rect = d.canvas._frame_rect(dict(d.canvas._frames)[1])
    d.canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, rect.center()))
    assert d.numbers() == (0, 1, 2, 3)


def test_leaving_every_frame_out_disables_export(dialog):
    d = dialog(n=2, rejected=(0, 1))
    d._render()
    assert not d.export_btn.isEnabled()


def test_white_paper_toggle_reaches_the_look_and_the_settings(dialog):
    d = dialog()
    assert not d.look().white_paper
    d.white_btn.setChecked(True)
    assert d.look().white_paper
    assert d.settings().white_paper is True


def test_edge_print_toggle_reaches_the_look_and_the_settings(dialog):
    d = dialog()
    assert d.look().edge.printed
    d.edge_btn.setChecked(False)
    assert not d.look().edge.printed
    assert d.settings().edge_print is False


def test_straight_proof_is_offered_only_when_it_can_be_made(dialog):
    d = dialog(proof=StraightProof(reason="Run Roll Analysis on this roll first."))
    assert not d.print_btn.choice_menu.actions()[1].isEnabled()
    assert "Roll Analysis" in d.proof_label.text()


def test_straight_proof_prints_its_own_configs_and_tiles(dialog):
    tiles = FakeTiles()
    frames = _frames(3)
    proofed = tuple(replace(f, config=straight_proof_config(f.config, (-2.0,) * 3, (-0.5,) * 3)) for f in frames)
    d = dialog(n=3, tiles=tiles, proof=StraightProof(proofed, "", "Assumes every frame was scanned at one exposure."))
    tiles.tile_ready.emit(1, 0, np.full((200, 300, 3), 90, np.uint8))
    d.print_btn.setCurrentIndex(1)
    assert len(tiles.requests) == 2 and tiles.requests[1] == proofed
    assert d.straight_proof()
    assert d._tiles[0] is None
    tiles.tile_ready.emit(2, 0, np.full((200, 300, 3), 40, np.uint8))
    assert int(d._tiles[0][0, 0, 0]) == 40
    assert all(f.config.exposure.grade == PROOF_GRADE for f in d.kept_frames())
    assert "one exposure" in d.proof_label.text()
    d.print_btn.setCurrentIndex(0)
    assert int(d._tiles[0][0, 0, 0]) == 90
    assert len(tiles.requests) == 2
    d.reject()
    assert tiles.cancelled == [1, 2]


def _scened(frames, scenes):
    return [
        replace(f, asset={**f.asset, "scene": (ordinal, f"s{ordinal}", f"Scene {ordinal}")}) if ordinal else f
        for f, ordinal in zip(frames, scenes)
    ]


def test_scene_order_needs_scenes(dialog):
    d = dialog(n=4)
    assert not d.order_btn.choice_menu.actions()[1].isEnabled()


def test_scene_order_groups_frames_and_starts_each_scene_on_a_new_strip(dialog):
    frames = _scened(_frames(8), [2, 1, 1, 0, 2, 1, 0, 2])
    d = dialog(frames=frames)
    d.order_btn.setCurrentIndex(1)
    assert d.numbers() == (1, 2, 5, 0, 4, 7, 3, 6)
    assert d.breaks() == (3, 6)
    assert [(s.first, s.count) for s in d.plan().pages[0].strips] == [(0, 3), (3, 3), (6, 2)]
    assert [f.name for f in d.kept_frames()] == ["1.tif", "2.tif", "5.tif", "0.tif", "4.tif", "7.tif", "3.tif", "6.tif"]
    assert d.settings().by_scene is True
    d.order_btn.setCurrentIndex(0)
    assert d.numbers() == tuple(range(8)) and d.breaks() == ()


def test_scene_order_proofs_each_scene_at_its_own_metering(dialog):
    tiles = FakeTiles()
    frames = _scened(_frames(3), [1, 1, 2])
    roll_proof = tuple(replace(f, config=straight_proof_config(f.config, (-2.0,) * 3, (-0.5,) * 3)) for f in frames)
    scene_proofed = tuple(replace(f, config=straight_proof_config(f.config, (-3.0,) * 3, (-1.0,) * 3)) for f in frames)
    d = dialog(
        frames=frames,
        tiles=tiles,
        proof=StraightProof(roll_proof, "", "roll"),
        scene_proof=StraightProof(scene_proofed, "", "Each scene prints at its own metering."),
    )
    d.print_btn.setCurrentIndex(1)
    assert d.kept_frames()[0].config.process.locked_floors == (-2.0,) * 3
    d.order_btn.setCurrentIndex(1)
    assert d.kept_frames()[0].config.process.locked_floors == (-3.0,) * 3
    assert tiles.requests[-1] == scene_proofed
    assert "own metering" in d.proof_label.text()


def test_a_proof_unavailable_in_scene_order_falls_back_to_as_edited(dialog):
    frames = _scened(_frames(3), [1, 1, 2])
    roll_proof = tuple(replace(f, config=straight_proof_config(f.config, (-2.0,) * 3, (-0.5,) * 3)) for f in frames)
    d = dialog(frames=frames, proof=StraightProof(roll_proof, "", "roll"), scene_proof=StraightProof(reason="Run Scene Analysis first."))
    d.print_btn.setCurrentIndex(1)
    d.order_btn.setCurrentIndex(1)
    assert d.print_btn.currentIndex() == 0 and not d.straight_proof()
    assert "Scene Analysis" in d.proof_label.text()


def test_the_preview_keeps_its_size_when_the_roll_spills_onto_a_second_sheet(dialog):
    d = dialog(n=38)
    d.show()
    d._render()
    QApplication.processEvents()
    before = d.canvas.size()
    d._set_paper(DEFAULT_PAPER.width, 260.0, snap=False)
    d._render()
    QApplication.processEvents()
    assert len(d.plan().pages) == 2
    assert d.canvas.size() == before


def test_a_drag_measures_from_where_the_sheet_was_when_it_began(dialog):
    d = dialog()
    d.show()
    QApplication.processEvents()
    canvas = d.canvas
    rect = canvas.sheet_rect()
    scale = canvas.px_per_mm()
    canvas.mousePressEvent(_mouse(QEvent.Type.MouseButtonPress, QPointF(rect.center().x(), rect.bottom())))
    canvas.resize(canvas.width(), canvas.height() - 40)
    canvas.mouseMoveEvent(_mouse(QEvent.Type.MouseMove, QPointF(rect.center().x(), rect.bottom() + 5 * scale)))
    canvas.mouseReleaseEvent(_mouse(QEvent.Type.MouseButtonRelease, QPointF(rect.center().x(), rect.bottom()), Qt.MouseButton.NoButton))
    assert d.settings().paper_height == pytest.approx(DEFAULT_PAPER.height + 10, abs=1.0)


def test_film_base_follows_white_paper_and_reaches_the_look_and_the_settings(dialog):
    d = dialog()
    assert not d.film_base_btn.isEnabled()  # nothing to print white until the paper is
    d.white_btn.setChecked(True)
    assert d.film_base_btn.isEnabled() and d.look().film_base
    d.film_base_btn.setChecked(False)
    assert not d.look().film_base
    assert d.settings().film_base is False
