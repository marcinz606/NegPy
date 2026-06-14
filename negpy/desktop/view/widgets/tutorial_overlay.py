from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QEvent, QObject, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from negpy.desktop.view.styles.theme import THEME

if TYPE_CHECKING:
    from negpy.desktop.view.main_window import MainWindow


class TutorialStep:
    __slots__ = ("title", "body", "target", "section_attr", "pre_hook")

    def __init__(
        self,
        title: str,
        body: str,
        target: Callable[["MainWindow"], Optional[QWidget]],
        section_attr: str = "",
        pre_hook: Optional[Callable[["MainWindow"], None]] = None,
    ) -> None:
        self.title = title
        self.body = body
        self.target = target
        self.section_attr = section_attr
        self.pre_hook = pre_hook


class TutorialOverlay(QWidget):
    """Full-window tutorial overlay: dark scrim with cutout + popup card."""

    finished = pyqtSignal(bool)  # True = completed all steps, False = skipped/dismissed

    _PAD = 10
    _POPUP_W = 340
    _GAP = 16

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self._win = window
        self._use_top_level_window = sys.platform == "win32"

        if self._use_top_level_window:
            self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
            self.setWindowModality(Qt.WindowModality.WindowModal)
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._sync_geometry()
        window.installEventFilter(self)

        self._steps: list[TutorialStep] = []
        self._idx: int = 0
        self._expanded: dict[str, bool] = {}

        self._build_popup()
        self.hide()

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

        self._counter = QLabel()
        self._counter.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;")
        layout.addWidget(self._counter)

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

        self._hint_lbl = QLabel("Enter / → to advance  ·  ← to go back  ·  Esc to dismiss")
        self._hint_lbl.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;")
        layout.addWidget(self._hint_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._prev_btn = QPushButton("← Back")
        self._prev_btn.clicked.connect(self._prev)
        self._prev_btn.setStyleSheet(self._btn_qss(accent=False, muted=False))

        self._skip_btn = QPushButton("Skip tour")
        self._skip_btn.clicked.connect(self.dismiss)
        self._skip_btn.setStyleSheet(self._btn_qss(accent=False, muted=True))

        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._next)
        self._next_btn.setStyleSheet(self._btn_qss(accent=True, muted=False))

        btn_row.addWidget(self._prev_btn)
        btn_row.addWidget(self._skip_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._next_btn)
        layout.addLayout(btn_row)

    def _btn_qss(self, accent: bool, muted: bool) -> str:
        if accent:
            bg, fg, border, hover = THEME.accent_primary, "#FFFFFF", "none", THEME.accent_secondary
        elif muted:
            bg, fg, border, hover = "transparent", THEME.text_muted, "none", THEME.bg_panel
        else:
            bg, fg, border, hover = "transparent", THEME.text_primary, f"1px solid {THEME.border_primary}", THEME.bg_panel
        return (
            f"QPushButton {{ background: {bg}; color: {fg}; border: {border}; "
            f"border-radius: 3px; padding: 5px 14px; font-size: {THEME.font_size_base}px; }}"
            f"QPushButton:hover {{ background: {hover}; }}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, steps: list[TutorialStep]) -> None:
        self._steps = steps
        self._idx = 0
        self._expanded = {}
        self._sync_geometry()
        self.show()
        self.raise_()
        self.activateWindow()
        self._popup.raise_()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self._goto(0)

    def dismiss(self) -> None:
        self._restore_sections()
        self.hide()
        self.finished.emit(False)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _next(self) -> None:
        if self._idx >= len(self._steps) - 1:
            self._restore_sections()
            self.hide()
            self.finished.emit(True)
        else:
            self._goto(self._idx + 1)

    def _prev(self) -> None:
        if self._idx > 0:
            self._goto(self._idx - 1)

    def _goto(self, idx: int) -> None:
        self._idx = idx
        step = self._steps[idx]

        if step.pre_hook:
            step.pre_hook(self._win)

        if step.section_attr:
            self._win.right_panel.reveal_section(step.section_attr)
            section = getattr(self._win.controls_panel, step.section_attr, None)
            if section is not None and not section.toggle_button.isChecked():
                if step.section_attr not in self._expanded:
                    self._expanded[step.section_attr] = False
                section.toggle_button.setChecked(True)

        target = step.target(self._win)
        if target is not None:
            self._win.right_panel.scroll_to(target)

        total = len(self._steps)
        self._counter.setText(f"Step {idx + 1} of {total}")
        self._title_lbl.setText(step.title)
        self._body_lbl.setHtml(step.body)
        doc = self._body_lbl.document()
        if doc is not None:
            doc.setDefaultFont(self._body_lbl.font())
            doc.setTextWidth(self._POPUP_W - 32)
            content_h = int(doc.size().height()) + 8
            max_h = max(80, self._win.height() - 220)
            self._body_lbl.setFixedHeight(min(content_h, max_h))
        self._prev_btn.setVisible(idx > 0)
        self._skip_btn.setVisible(idx < total - 1)
        self._next_btn.setText("Done" if idx == total - 1 else "Next →")

        self._popup.adjustSize()
        self._position_popup(target)
        self.update()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _target_rect(self, target: Optional[QWidget]) -> Optional[QRectF]:
        if target is None or not target.isVisible():
            return None
        gp = target.mapToGlobal(target.rect().topLeft())
        lp = self.mapFromGlobal(gp)
        return QRectF(lp.x(), lp.y(), target.width(), target.height())

    def _position_popup(self, target: Optional[QWidget]) -> None:
        lyt = self._popup.layout()
        if lyt is not None:
            lyt.activate()
        self._popup.adjustSize()
        ph = self._popup.height()
        pw = self._POPUP_W
        ow, oh = self.width(), self.height()

        tr = self._target_rect(target)
        if tr is None:
            self._popup.setGeometry((ow - pw) // 2, (oh - ph) // 2, pw, ph)
            return

        hi = tr.adjusted(-self._PAD, -self._PAD, self._PAD, self._PAD)

        # Prefer left of target (sidebar is right dock); fall back to right
        x = int(hi.left()) - self._GAP - pw
        if x < 8:
            x = int(hi.right()) + self._GAP
        x = max(8, min(x, ow - pw - 8))
        y = max(8, min(int(hi.top()), oh - ph - 8))

        self._popup.setGeometry(x, y, pw, ph)

    # ------------------------------------------------------------------
    # Section state
    # ------------------------------------------------------------------

    def _restore_sections(self) -> None:
        for attr, was_expanded in self._expanded.items():
            section = getattr(self._win.controls_panel, attr, None)
            if section is not None and not was_expanded:
                section.toggle_button.setChecked(False)
        self._expanded = {}

    def _sync_geometry(self) -> None:
        if self._use_top_level_window:
            self.setGeometry(self._win.geometry())
        else:
            self.setGeometry(self._win.rect())

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def paintEvent(self, a0) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        target: Optional[QWidget] = None
        if self._steps:
            target = self._steps[self._idx].target(self._win)

        tr = self._target_rect(target)
        scrim = QColor(0, 0, 0, 170)

        if tr is not None:
            hi = tr.adjusted(-self._PAD, -self._PAD, self._PAD, self._PAD)
            full = QPainterPath()
            full.addRect(QRectF(self.rect()))
            hole = QPainterPath()
            hole.addRoundedRect(hi, 6, 6)
            painter.fillPath(full.subtracted(hole), scrim)
            painter.setPen(QPen(QColor(THEME.accent_primary), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(hi, 6, 6)
        else:
            painter.fillRect(self.rect(), scrim)

    def mousePressEvent(self, a0) -> None:  # type: ignore[override]
        if a0 is not None:
            a0.accept()

    def keyPressEvent(self, a0) -> None:  # type: ignore[override]
        if a0 is None:
            return
        k = a0.key()
        if k == Qt.Key.Key_Escape:
            self.dismiss()
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Space):
            self._next()
        elif k == Qt.Key.Key_Left:
            self._prev()
        else:
            super().keyPressEvent(a0)

    def eventFilter(self, a0: Optional[QObject], a1: Optional[QEvent]) -> bool:  # type: ignore[override]
        if (
            a0 is self._win
            and a1 is not None
            and a1.type()
            in {
                QEvent.Type.Move,
                QEvent.Type.Resize,
                QEvent.Type.Show,
                QEvent.Type.WindowStateChange,
            }
        ):
            self._sync_geometry()
            target: Optional[QWidget] = None
            if self._steps:
                target = self._steps[self._idx].target(self._win)
            self._position_popup(target)
            self.update()
        return False
