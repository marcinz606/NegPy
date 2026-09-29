import re
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import tifffile
from PyQt6.QtCore import QPoint, QRectF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

from negpy.desktop.view import shortcut_registry
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.section_help_dialog import has_guide
from negpy.desktop.view.widgets.tutorial_demo import write_demo
from negpy.desktop.view.widgets.tutorial_overlay import Offer, TutorialOverlay, TutorialStep
from negpy.desktop.view.widgets.tutorial_steps import build


def _host() -> QMainWindow:
    win = QMainWindow()
    win.setGeometry(100, 100, 900, 700)
    win.show()
    return win


def _started(win: QMainWindow, steps: list[TutorialStep]) -> tuple[TutorialOverlay, list[bool]]:
    overlay = TutorialOverlay(win)
    finished: list[bool] = []
    overlay.finished.connect(finished.append)
    overlay.start(steps)
    QApplication.processEvents()
    return overlay, finished


def _click(btn) -> None:
    QTest.mouseClick(btn, Qt.MouseButton.LeftButton)
    QApplication.processEvents()


def test_built_steps_are_consistent() -> None:
    steps = build(MagicMock())
    chapters = list(dict.fromkeys(s.chapter for s in steps))
    # Each chapter is one run, so the chapter menu and Skip Chapter land on its start.
    assert [s.chapter for s in steps] == [c for c in chapters for s in steps if s.chapter == c]
    for s in steps:
        words = len(re.sub(r"<[^>]+>", " ", s.body).split())
        assert words <= 60, f"{s.title}: {words} words"
        assert not s.guide[0] or has_guide(s.guide[0]), f"{s.title}: no guide for {s.guide[0]}"
        assert bool(s.task) == (s.watch is not None), s.title


def test_tour_keys_come_from_the_registry() -> None:
    source = open("negpy/desktop/view/widgets/tutorial_steps.py", encoding="utf-8").read()
    ids = set(re.findall(r"_k\('([a-z_]+)'\)", source))
    assert ids and ids <= set(shortcut_registry.REGISTRY)
    assert "show_tour" in shortcut_registry.REGISTRY


def test_next_button_walks_to_the_end() -> None:
    overlay, finished = _started(_host(), [TutorialStep("A", "One", "x", lambda _: None), TutorialStep("A", "Two", "y", lambda _: None)])
    _click(overlay._next_btn)
    assert overlay.index == 1
    assert overlay._next_btn.text() == "Done"
    _click(overlay._next_btn)
    assert finished == [True]
    assert not overlay.isVisible()


def test_chapters_skip_and_jump() -> None:
    steps = [TutorialStep(c, f"{c}{i}", "x", lambda _: None) for c in ("A", "B", "C") for i in range(2)]
    overlay, _ = _started(_host(), steps)
    assert overlay._counter.text() == "1 of 2"
    _click(overlay._skip_btn)
    assert overlay.index == 2
    assert overlay._chapter_btn is not None
    overlay._chapter_btn.setCurrentIndex(2)
    assert overlay.index == 4
    assert not overlay._skip_btn.isVisible()


def test_task_marks_done_and_waits_for_next() -> None:
    state = {"v": 0}
    steps = [
        TutorialStep("A", "Try", "x", lambda _: None, task="Change v.", watch=lambda _: state["v"]),
        TutorialStep("A", "Next", "y", lambda _: None),
    ]
    overlay, _ = _started(_host(), steps)
    assert overlay._next_btn.text() == "Skip Step"
    state["v"] = 1
    overlay._poll()
    assert overlay._task_lbl.property("hint") == "success"
    assert overlay._next_btn.text() == "Next"
    QTest.qWait(1000)
    assert overlay.index == 0
    _click(overlay._next_btn)
    assert overlay.index == 1


def test_skip_step_advances_without_the_task() -> None:
    steps = [
        TutorialStep("A", "Try", "x", lambda _: None, task="Never.", watch=lambda _: 0),
        TutorialStep("A", "Next", "y", lambda _: None),
    ]
    overlay, _ = _started(_host(), steps)
    _click(overlay._next_btn)
    assert overlay.index == 1


def test_target_takes_real_clicks_through_the_mask() -> None:
    win = _host()
    target = QPushButton("Target", win)
    target.setGeometry(40, 40, 120, 30)
    target.show()
    overlay, _ = _started(win, [TutorialStep("A", "Look", "x", lambda _: target)])
    if overlay.isWindow():
        return
    assert not overlay.mask().contains(target.geometry().center())
    assert overlay.mask().contains(QPoint(600, 600))
    assert overlay.mask().contains(overlay._popup.geometry().center())


def test_picture_stays_clear_and_the_card_avoids_it() -> None:
    win = QMainWindow()
    win.setGeometry(50, 50, 1200, 800)
    win.show()
    picture = QWidget(win)
    picture.setGeometry(0, 0, 900, 800)
    target = QPushButton("Slider", win)
    target.setGeometry(1000, 100, 100, 30)
    picture.show()
    target.show()
    overlay, _ = _started(win, [TutorialStep("A", "Drag", "x", lambda _: target, also=lambda _: picture)])
    if overlay.isWindow():
        return
    assert not overlay.mask().contains(QPoint(450, 700))
    card = overlay._popup.geometry()
    assert overlay._covered(QRectF(card)) < card.width() * card.height() / 2


def test_offer_shows_while_visible_and_runs() -> None:
    ran: list[bool] = []
    flag = {"on": True}
    offer = Offer("Load", lambda _w: ran.append(True), lambda _w: flag["on"])
    overlay, _ = _started(_host(), [TutorialStep("A", "Try", "x", lambda _: None, offer=offer)])
    assert overlay._offer_btn.isVisible()
    _click(overlay._offer_btn)
    assert ran == [True]
    flag["on"] = False
    overlay._poll()
    assert not overlay._offer_btn.isVisible()


def test_collapsed_ancestor_opens_for_the_step_and_closes_after() -> None:
    win = _host()
    content = QWidget()
    target = QPushButton("Inside")
    QVBoxLayout(content).addWidget(target)
    section = CollapsibleSection("Card", expanded=False)
    section.set_content(content)
    win.setCentralWidget(section)
    QApplication.processEvents()
    overlay, _ = _started(win, [TutorialStep("A", "Inside", "x", lambda _: target)])
    assert section.toggle_button.isChecked()
    overlay.dismiss()
    assert not section.toggle_button.isChecked()


def test_start_at_resumes_the_saved_step() -> None:
    steps = [TutorialStep("A", str(i), "x", lambda _: None) for i in range(4)]
    overlay, finished = _started(_host(), steps)
    overlay.goto(2)
    overlay.dismiss()
    assert finished == [False]
    assert overlay.index == 2
    overlay.start(steps, at=overlay.index)
    assert overlay._title_lbl.text() == "2"


def test_post_hook_runs_when_a_step_is_left() -> None:
    left: list[str] = []
    steps = [TutorialStep("A", n, "x", lambda _: None, post_hook=lambda _w, n=n: left.append(n)) for n in ("one", "two")]
    overlay, _ = _started(_host(), steps)
    _click(overlay._next_btn)
    assert left == ["one"]
    overlay.dismiss()
    assert left == ["one", "two"]


def test_put_down_escapes_until_the_canvas_is_plain() -> None:
    from negpy.desktop.session import ToolMode
    from negpy.desktop.view.widgets.tutorial_steps import _put_down

    state = SimpleNamespace(
        active_tool=ToolMode.CROP_MANUAL,
        test_strip=False,
        test_strip_pending=False,
        negative_peek=False,
        embedded_peek=False,
        flat_peek=False,
        compare_mode=True,
        grain_focuser=False,
        zone_arm_target=None,
    )

    def esc() -> None:
        if state.compare_mode:
            state.compare_mode = False
        else:
            state.active_tool = ToolMode.NONE

    w = SimpleNamespace(state=state, shortcut_manager=SimpleNamespace(action_for=lambda _id: esc), light_table_active=lambda: False)
    _put_down(w)  # type: ignore[arg-type]
    assert not state.compare_mode and state.active_tool == ToolMode.NONE


def test_mask_outlines_follow_the_dodge_and_burn_tab() -> None:
    from negpy.desktop.view.sidebar.right_panel import RightPanel

    state = SimpleNamespace(local_masks_shown=True)
    stub = SimpleNamespace(
        _group_keys=["roll", "frame"],
        _active_group=1,
        _active_index=0,
        _section_tab_index={"local_section": 1},
        controller=SimpleNamespace(session=SimpleNamespace(state=state), config_updated=MagicMock()),
    )
    RightPanel._sync_local_masks(stub)  # type: ignore[arg-type]
    assert state.local_masks_shown is False
    stub._active_index = 1
    RightPanel._sync_local_masks(stub)  # type: ignore[arg-type]
    assert state.local_masks_shown is True
    assert stub.controller.config_updated.emit.call_count == 2


def test_demo_negative_is_written_once(tmp_path) -> None:
    path = write_demo(tmp_path)
    img = tifffile.imread(path)
    assert img.dtype.name == "uint16" and img.ndim == 3 and img.shape[2] == 3
    stamp = path.stat().st_mtime_ns
    assert write_demo(tmp_path) == path
    assert path.stat().st_mtime_ns == stamp


def test_tutorial_overlay_uses_top_level_window_on_windows() -> None:
    win = QMainWindow()
    overlay = TutorialOverlay(win)

    if sys.platform == "win32":
        assert overlay.isWindow()
        assert overlay.windowType() == Qt.WindowType.Tool
        assert overlay.windowFlags() & Qt.WindowType.FramelessWindowHint
    else:
        assert not overlay.isWindow()


def test_tall_card_lights_only_its_visible_part_and_scrolls_to_focus() -> None:
    from PyQt6.QtWidgets import QScrollArea

    win = _host()
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    card = QWidget()
    card.setMinimumHeight(3000)
    focus = QPushButton("Deep", card)
    focus.setGeometry(10, 2500, 100, 30)
    scroll.setWidget(card)
    win.setCentralWidget(scroll)
    QApplication.processEvents()

    def reveal(widget: QWidget, centered: bool = False) -> None:
        scroll.ensureWidgetVisible(widget)

    win.right_panel = SimpleNamespace(reveal_widget=reveal, scroll_to=reveal)  # type: ignore[attr-defined]
    overlay, _ = _started(win, [TutorialStep("A", "Card", "x", lambda _: card, focus=lambda _: focus)])
    hole = overlay._hole
    assert hole is not None
    viewport = scroll.viewport()
    top_left = overlay.mapFromGlobal(viewport.mapToGlobal(QPoint(0, 0)))
    assert hole == QRectF(top_left.x(), top_left.y(), viewport.width(), viewport.height())
    assert scroll.verticalScrollBar().value() > 0
    overlay.dismiss()


def test_focus_control_gets_a_ring_and_stays_clickable() -> None:
    win = _host()
    card = QWidget(win)
    card.setGeometry(500, 100, 300, 400)
    focus = QPushButton("Strip", card)
    focus.setGeometry(20, 200, 80, 30)
    card.show()
    overlay, _ = _started(win, [TutorialStep("A", "Card", "x", lambda _: card, focus=lambda _: focus)])
    rect = overlay._focus
    assert rect is not None
    mask = overlay.mask()
    assert not mask.contains(rect.center().toPoint())
    assert mask.contains(QPoint(int(rect.left()) - 4, int(rect.center().y())))
    assert not mask.contains(QPoint(int(rect.left()) - 20, int(rect.center().y())))
    overlay.dismiss()
