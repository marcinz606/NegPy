from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Optional

import qtawesome as qta
from PyQt6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QRegion, QTextOption
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QTextBrowser, QVBoxLayout, QWidget

from negpy.desktop.view.styles.templates import hint_label, labeled_action, set_hint_kind
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.choice_button import ChoiceButton
from negpy.desktop.view.widgets.collapsible import CollapsibleSection

if TYPE_CHECKING:
    from negpy.desktop.view.main_window import MainWindow

Window = Callable[["MainWindow"], object]


class Offer:
    """A button on the step's card, shown while visible(window) holds."""

    __slots__ = ("label", "run", "visible")

    def __init__(self, label: str, run: Callable[["MainWindow"], None], visible: Callable[["MainWindow"], bool]) -> None:
        self.label = label
        self.run = run
        self.visible = visible


class TutorialStep:
    """task + watch make a step interactive: it is done once watch(window) differs from its
    value when the step opened. also is a second clickable area without the ring, the canvas a
    tool acts on. guide is a (USER_GUIDE panel key, panel title) pair."""

    __slots__ = ("chapter", "title", "body", "target", "task", "watch", "also", "guide", "offer", "pre_hook")

    def __init__(
        self,
        chapter: str,
        title: str,
        body: str,
        target: Callable[["MainWindow"], Optional[QWidget]],
        *,
        task: str = "",
        watch: Optional[Window] = None,
        also: Optional[Callable[["MainWindow"], Optional[QWidget]]] = None,
        guide: tuple[str, str] = ("", ""),
        offer: Optional[Offer] = None,
        pre_hook: Optional[Callable[["MainWindow"], None]] = None,
    ) -> None:
        self.chapter = chapter
        self.title = title
        self.body = body
        self.target = target
        self.task = task
        self.watch = watch
        self.also = also
        self.guide = guide
        self.offer = offer
        self.pre_hook = pre_hook


_NAV_KEYS = {Qt.Key.Key_Right, Qt.Key.Key_Left, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}


class TutorialOverlay(QWidget):
    """Full-window tour: a scrim with a cutout over the target, which stays clickable, and a card."""

    finished = pyqtSignal(bool)  # True = completed all steps, False = skipped/dismissed

    _PAD = 10
    _POPUP_W = 360
    _GAP = 16
    _POLL_MS = 150
    _ADVANCE_MS = 800

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self._win = window
        # A native canvas paints over sibling widgets on Windows, so the overlay is its own window there.
        self._use_top_level_window = sys.platform == "win32"

        if self._use_top_level_window:
            self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._sync_geometry()
        window.installEventFilter(self)

        self._steps: list[TutorialStep] = []
        self._chapters: list[str] = []
        self._idx = 0
        self._baseline: object = None
        self._task_done = False
        self._hole: Optional[QRectF] = None
        self._also: Optional[QRectF] = None
        self._collapsed: list[CollapsibleSection] = []

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self._POLL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._advance_timer = QTimer(self)
        self._advance_timer.setSingleShot(True)
        self._advance_timer.timeout.connect(self._next)

        self._build_popup()
        self.hide()

    @property
    def index(self) -> int:
        return self._idx

    def _build_popup(self) -> None:
        self._popup = QFrame(self)
        self._popup.setFixedWidth(self._POPUP_W)
        self._popup.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME.bg_header};
                border: 1px solid {THEME.border_primary};
                border-radius: 6px;
            }}
            QLabel {{ border: none; background: transparent; }}
        """)

        layout = QVBoxLayout(self._popup)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self._header = QHBoxLayout()
        self._chapter_btn: Optional[ChoiceButton] = None
        self._header.addStretch()
        self._counter = hint_label()
        self._counter.setWordWrap(False)
        self._header.addWidget(self._counter)
        layout.addLayout(self._header)

        self._title_lbl = QLabel()
        self._title_lbl.setStyleSheet(f"color: {THEME.text_primary}; font-size: {THEME.font_size_title}px; font-weight: bold;")
        self._title_lbl.setWordWrap(True)
        layout.addWidget(self._title_lbl)

        self._body_lbl = QTextBrowser()
        self._body_lbl.setReadOnly(True)
        self._body_lbl.setOpenExternalLinks(False)
        self._body_lbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._body_lbl.setFrameShape(QFrame.Shape.NoFrame)
        self._body_lbl.setStyleSheet(
            f"QTextBrowser {{ background: transparent; border: none; color: {THEME.text_secondary}; font-size: {THEME.font_size_base}px; }}"
        )
        layout.addWidget(self._body_lbl)

        self._task_row = QWidget()
        task_layout = QHBoxLayout(self._task_row)
        task_layout.setContentsMargins(0, 0, 0, 0)
        task_layout.setSpacing(THEME.space_md)
        self._task_icon = QLabel()
        self._task_lbl = hint_label()
        task_layout.addWidget(self._task_icon, 0, Qt.AlignmentFlag.AlignTop)
        task_layout.addWidget(self._task_lbl, 1)
        layout.addWidget(self._task_row)

        extras = QHBoxLayout()
        extras.setSpacing(6)
        self._offer_btn = labeled_action("fa5s.play-circle", "", "")
        self._offer_btn.clicked.connect(self._run_offer)
        self._more_btn = labeled_action("fa5s.book-open", " Read More…", "Open the full guide for this panel")
        self._more_btn.clicked.connect(self._read_more)
        extras.addWidget(self._offer_btn)
        extras.addWidget(self._more_btn)
        extras.addStretch()
        layout.addLayout(extras)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._prev_btn = labeled_action("fa5s.arrow-left", "", "Back (←)")
        self._prev_btn.clicked.connect(self._prev)
        self._skip_btn = labeled_action("", "Skip Chapter", "Go to the next chapter of the tour")
        self._skip_btn.clicked.connect(self._skip_chapter)
        self._next_btn = labeled_action("", "Next", "Next (→ or Enter). Esc ends the tour", primary=True)
        self._next_btn.clicked.connect(self._next)
        btn_row.addWidget(self._prev_btn)
        btn_row.addWidget(self._skip_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._next_btn)
        layout.addLayout(btn_row)

        # The overlay keeps the focus, so its navigation keys still reach it after a click.
        for btn in (self._offer_btn, self._more_btn, self._prev_btn, self._skip_btn, self._next_btn):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _build_chapter_button(self) -> None:
        if self._chapter_btn is not None:
            self._header.removeWidget(self._chapter_btn)
            self._chapter_btn.deleteLater()
        self._chapter_btn = ChoiceButton(tuple(("", c) for c in self._chapters), "Jump to a chapter of the tour")
        self._chapter_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._chapter_btn.currentChanged.connect(self._jump_chapter)
        self._header.insertWidget(0, self._chapter_btn)

    # Public API

    def start(self, steps: list[TutorialStep], at: int = 0) -> None:
        self._steps = steps
        self._chapters = list(dict.fromkeys(s.chapter for s in steps))
        self._collapsed = []
        self._build_chapter_button()
        self._sync_geometry()
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self._poll_timer.start()
        self.goto(max(0, min(at, len(steps) - 1)))

    def dismiss(self) -> None:
        self._close(False)

    def _close(self, completed: bool) -> None:
        self._poll_timer.stop()
        self._advance_timer.stop()
        for section in self._collapsed:
            section.set_expanded(False)
        self._collapsed = []
        self.clearMask()
        self.hide()
        self.finished.emit(completed)

    # Navigation

    def _next(self) -> None:
        if self._idx >= len(self._steps) - 1:
            self._close(True)
        else:
            self.goto(self._idx + 1)

    def _prev(self) -> None:
        if self._idx > 0:
            self.goto(self._idx - 1)

    def _chapter_start(self, chapter: str) -> int:
        return next(i for i, s in enumerate(self._steps) if s.chapter == chapter)

    def _skip_chapter(self) -> None:
        pos = self._chapters.index(self._steps[self._idx].chapter)
        if pos + 1 < len(self._chapters):
            self.goto(self._chapter_start(self._chapters[pos + 1]))

    def _jump_chapter(self, pos: int) -> None:
        self.goto(self._chapter_start(self._chapters[pos]))

    def goto(self, idx: int) -> None:
        self._advance_timer.stop()
        self._idx = idx
        step = self._steps[idx]

        if step.pre_hook:
            step.pre_hook(self._win)
        target = step.target(self._win)
        if target is not None:
            self._reveal(target)
        self._baseline = step.watch(self._win) if step.watch else None
        self._task_done = False

        pos = self._chapters.index(step.chapter)
        if self._chapter_btn is not None:
            self._chapter_btn.blockSignals(True)
            self._chapter_btn.setCurrentIndex(pos)
            self._chapter_btn.blockSignals(False)
        in_chapter = [i for i, s in enumerate(self._steps) if s.chapter == step.chapter]
        self._counter.setText(f"{in_chapter.index(idx) + 1} of {len(in_chapter)}")
        self._title_lbl.setText(step.title)
        self._set_body(step.body)

        from negpy.desktop.view.widgets.section_help_dialog import has_guide

        self._task_row.setVisible(bool(step.task))
        self._task_lbl.setText(step.task)
        self._more_btn.setVisible(bool(step.guide[0]) and has_guide(step.guide[0]))
        self._prev_btn.setVisible(idx > 0)
        self._skip_btn.setVisible(pos < len(self._chapters) - 1)
        self._sync_task()
        self._sync_offer()

        # Keys a task asks for are the main window's shortcuts, which a separate overlay window blocks.
        if step.watch is not None and self._use_top_level_window:
            self._win.activateWindow()
        self._hole = self._target_rect(target)
        self._also = self._target_rect(step.also(self._win)) if step.also else None
        self._layout()

    def _set_body(self, html: str) -> None:
        self._body_lbl.setHtml(html)
        doc = self._body_lbl.document()
        if doc is None:
            return
        doc.setDefaultFont(self._body_lbl.font())
        opt = QTextOption()
        opt.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        doc.setDefaultTextOption(opt)
        margin = int(doc.documentMargin())
        # -2: the popup QFrame's 1px stylesheet border eats one pixel on each side.
        doc.setTextWidth(self._POPUP_W - 32 - 2 - 2 * margin)
        content_h = int(doc.size().height()) + 2 * margin
        self._body_lbl.setFixedHeight(min(content_h, max(80, self._win.height() - 280)))

    def _sync_task(self) -> None:
        step = self._steps[self._idx]
        done = self._task_done
        self._task_icon.setPixmap(
            qta.icon("fa5s.check-circle" if done else "fa5.circle", color=THEME.status_success if done else THEME.text_hint).pixmap(
                THEME.font_size_base, THEME.font_size_base
            )
        )
        set_hint_kind(self._task_lbl, "success" if done else "muted")
        last = self._idx == len(self._steps) - 1
        self._next_btn.setText("Done" if last else ("Skip Step" if step.watch and not done else "Next"))

    def _sync_offer(self) -> None:
        offer = self._steps[self._idx].offer
        show = offer is not None and offer.visible(self._win)
        if offer is not None:
            self._offer_btn.setText(" " + offer.label)
        if show != self._offer_btn.isVisible():
            self._offer_btn.setVisible(show)
            self._layout()

    def _run_offer(self) -> None:
        offer = self._steps[self._idx].offer
        if offer is not None:
            offer.run(self._win)

    def _read_more(self) -> None:
        from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog

        key, title = self._steps[self._idx].guide
        SectionHelpDialog(key, title, parent=self._win).exec()

    def _poll(self) -> None:
        if not self._steps:
            return
        step = self._steps[self._idx]
        if step.watch is not None and not self._task_done and step.watch(self._win) != self._baseline:
            self._task_done = True
            self._sync_task()
            self._advance_timer.start(self._ADVANCE_MS)
        self._sync_offer()
        hole = self._target_rect(step.target(self._win))
        also = self._target_rect(step.also(self._win)) if step.also else None
        if (hole, also) != (self._hole, self._also):
            self._hole, self._also = hole, also
            self._layout()

    # Layout helpers

    def _reveal(self, target: QWidget) -> None:
        for dock in (getattr(self._win, "session_dock", None), getattr(self._win, "drawer", None)):
            if dock is not None and dock.isAncestorOf(target) and not dock.isVisible():
                dock.show()
        parent = target.parentWidget()
        while parent is not None:
            if (
                isinstance(parent, CollapsibleSection)
                and parent.collapsible
                and not parent.toggle_button.isChecked()
                and parent not in self._collapsed
            ):
                self._collapsed.append(parent)
            parent = parent.parentWidget()
        right_panel = getattr(self._win, "right_panel", None)
        if right_panel is not None:
            right_panel.reveal_widget(target)
        else:
            for section in self._collapsed:
                section.expand()

    def _target_rect(self, target: Optional[QWidget]) -> Optional[QRectF]:
        if target is None or not target.isVisible():
            return None
        lp = self.mapFromGlobal(target.mapToGlobal(target.rect().topLeft()))
        rect = QRectF(lp.x(), lp.y(), target.width(), target.height()).intersected(QRectF(self.rect()))
        return rect if not rect.isEmpty() else None

    def _layout(self) -> None:
        self._position_popup()
        holes = [h for h in (self._hole, self._also) if h is not None]
        if not holes:
            self.clearMask()
        else:
            # The holes pass real input through to the target; the card stays clickable where it overlaps.
            region = QRegion(self.rect())
            for h in holes:
                region = region.subtracted(QRegion(h.toAlignedRect()))
            self.setMask(region.united(QRegion(self._popup.geometry())))
        self.update()

    def _position_popup(self) -> None:
        lyt = self._popup.layout()
        if lyt is not None:
            lyt.activate()
        self._popup.adjustSize()
        pw, ph = self._POPUP_W, self._popup.height()
        ow, oh, m = self.width(), self.height(), 8

        if self._hole is None:
            self._popup.setGeometry((ow - pw) // 2, (oh - ph) // 2, pw, ph)
            return

        hi = self._hole.adjusted(-self._PAD, -self._PAD, self._PAD, self._PAD)
        top = max(m, min(int(hi.top()), oh - ph - m))
        center_x = max(m, min(int(hi.center().x()) - pw // 2, ow - pw - m))
        # Beside the target first (the controls dock is on the right), then below or above it.
        for x, y in (
            (int(hi.left()) - self._GAP - pw, top),
            (int(hi.right()) + self._GAP, top),
            (center_x, int(hi.bottom()) + self._GAP),
            (center_x, int(hi.top()) - self._GAP - ph),
        ):
            if m <= x <= ow - pw - m and m <= y <= oh - ph - m:
                self._popup.setGeometry(x, y, pw, ph)
                return
        # A target that fills the window, the canvas: the card sits in its bottom-right corner.
        x = max(m, min(int(hi.right()) - self._GAP - pw, ow - pw - m))
        y = max(m, min(int(hi.bottom()) - self._GAP - ph, oh - ph - m))
        self._popup.setGeometry(x, y, pw, ph)

    def _sync_geometry(self) -> None:
        if self._use_top_level_window:
            self.setGeometry(self._win.geometry())
        else:
            self.setGeometry(self._win.rect())

    # Qt overrides

    def paintEvent(self, a0) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scrim = QColor(0, 0, 0, 170)

        full = QPainterPath()
        full.addRect(QRectF(self.rect()))
        if self._also is not None:
            also = QPainterPath()
            also.addRect(self._also)
            full = full.subtracted(also)
        if self._hole is None:
            painter.fillPath(full, scrim)
            return
        hi = self._hole.adjusted(-self._PAD, -self._PAD, self._PAD, self._PAD)
        hole = QPainterPath()
        hole.addRoundedRect(hi, 6, 6)
        painter.fillPath(full.subtracted(hole), scrim)
        painter.setPen(QPen(QColor(THEME.accent_primary), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(hi, 6, 6)

        card = QRectF(self._popup.geometry())
        if not card.intersects(hi):
            # A leader from the card's nearest edge to the target's.
            c = hi.center()
            start = QPointF(min(max(c.x(), card.left()), card.right()), min(max(c.y(), card.top()), card.bottom()))
            end = QPointF(min(max(start.x(), hi.left()), hi.right()), min(max(start.y(), hi.top()), hi.bottom()))
            painter.drawLine(start, end)

    def mousePressEvent(self, a0) -> None:  # type: ignore[override]
        if a0 is not None:
            a0.accept()

    def _on_task(self) -> bool:
        return bool(self._steps) and self._steps[self._idx].watch is not None

    def event(self, a0: Optional[QEvent]) -> bool:  # type: ignore[override]
        # Claims the navigation keys ahead of the window's shortcuts, except on a task, which may need them.
        if a0 is not None and a0.type() == QEvent.Type.ShortcutOverride:
            key = a0.key()  # type: ignore[attr-defined]
            if key == Qt.Key.Key_Escape or (key in _NAV_KEYS and not self._on_task()):
                a0.accept()
                return True
        return super().event(a0)

    def keyPressEvent(self, a0) -> None:  # type: ignore[override]
        if a0 is None:
            return
        k = a0.key()
        if k == Qt.Key.Key_Escape:
            self.dismiss()
        elif self._on_task():
            super().keyPressEvent(a0)
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._next()
        elif k == Qt.Key.Key_Left:
            self._prev()
        else:
            super().keyPressEvent(a0)

    def eventFilter(self, a0: Optional[QObject], a1: Optional[QEvent]) -> bool:  # type: ignore[override]
        if (
            a0 is self._win
            and a1 is not None
            and self.isVisible()
            and a1.type()
            in {
                QEvent.Type.Move,
                QEvent.Type.Resize,
                QEvent.Type.Show,
                QEvent.Type.WindowStateChange,
            }
        ):
            self._sync_geometry()
            if self._steps:
                step = self._steps[self._idx]
                self._hole = self._target_rect(step.target(self._win))
                self._also = self._target_rect(step.also(self._win)) if step.also else None
            self._layout()
        return False
