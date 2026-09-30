"""Contact Sheet dialog: a low-res proof of the sheet whose paper edges drag to resize it."""

from dataclasses import replace
from typing import Callable, Optional, Protocol, Sequence

import numpy as np
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from negpy.desktop.view.canvas.overlay import draw_resize_handles, draw_view_badge, hit_resize_handle, resize_cursor
from negpy.desktop.view.styles.templates import (
    FIELD_LABEL_WIDTH,
    field_label,
    hint_label,
    icon_button,
    labeled_toggle,
    pin_dialog_default,
    set_hint_kind,
    tool_toggle,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.choice_button import ChoiceButton
from negpy.desktop.view.widgets.marks import draw_mark_badge
from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog, has_guide
from negpy.desktop.view.widgets.sliders import CompactSlider, SliderGroup, align_slider_columns
from negpy.services.export.contact_sheet import ContactSheetService
from negpy.services.export.contact_sheet_layout import (
    DEFAULT_PAPER,
    DPI_CHOICES,
    FRAME_SIZES_120,
    ILFORD_PAPERS,
    PAPER_MAX,
    PAPER_MIN,
    ContactSheetSettings,
    SheetFormat,
    SheetPlan,
    best_dpi,
    better_orientation,
    clamp_paper,
    dpi_allowed,
    film_geometry,
    paper_preset,
    plan_capacity,
    plan_sheets,
    snap_paper,
)
from negpy.services.export.contact_sheet_roll import (
    SheetFrame,
    SheetLook,
    StraightProof,
    scene_breaks,
    scene_of,
    scene_order,
    turns_for,
)

_GUIDE_KEY = "contact_sheet"
_FORMATS: tuple[SheetFormat, ...] = (SheetFormat.FULL_FRAME, SheetFormat.HALF_FRAME, SheetFormat.MEDIUM)
_FRAME_SIZES: tuple[str, ...] = tuple(FRAME_SIZES_120)
_CUSTOM_PAPER = len(ILFORD_PAPERS)
_PRINTS = ("As Edited", "Straight Proof")
_AS_EDITED, _STRAIGHT_PROOF = 0, 1
_ORDERS = ("Date", "Scene")
_BY_DATE, _BY_SCENE = 0, 1
# Tile sets: as edited, straight proof at the roll's metering, straight proof at each scene's.
_EDITED_TILES, _PROOF_TILES, _SCENE_PROOF_TILES = 0, 1, 2
_CANVAS_PAD = 28.0
# Tiles arrive one by one; a repaint per batch keeps a fast roll from re-rendering per frame.
_TILE_REPAINT_MS = 120


class PreviewTiles(Protocol):
    """Where the dialog's preview tiles come from (the controller's worker, or a test double)."""

    tile_ready: object

    def render(self, frames: tuple[SheetFrame, ...]) -> int: ...

    def cancel(self, generation: int) -> None: ...


def _mm(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


class SheetCanvas(QWidget):
    """The sheet on the canvas background; corner and edge handles resize the paper about its
    center. The scale is held while a handle is dragged, so the handle stays under the pointer.
    While picking, a click on a frame toggles it and left-out frames carry the Film Strip's
    reject mark."""

    paper_dragged = pyqtSignal(float, float)
    paper_released = pyqtSignal()
    frame_clicked = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        # Taking focus on press commits a half-typed size box before the drag moves the paper.
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._paper = (1.0, 1.0)
        self._image: Optional[QImage] = None
        self._drag: Optional[str] = None
        self._held_scale: Optional[float] = None
        self._readout = ""
        self._frames: list[tuple[int, QRectF]] = []
        self._left_out: set[int] = set()
        self._picking = False
        self._hover: Optional[int] = None

    def set_paper(self, width: float, height: float, readout: str = "") -> None:
        self._paper = (width, height)
        self._readout = readout
        self.update()

    def set_image(self, image: Optional[QImage]) -> None:
        self._image = image
        self.update()

    def set_frames(self, frames: list[tuple[int, QRectF]], left_out: set[int], picking: bool) -> None:
        """Frame windows on the shown sheet, in paper millimetres, by frame index."""
        self._frames = frames
        self._left_out = set(left_out)
        self._picking = picking
        if not picking:
            self._hover = None
        self.update()

    def dragging(self) -> bool:
        return self._drag is not None

    def _frame_rect(self, rect_mm: QRectF) -> QRectF:
        s = self.px_per_mm()
        origin = self.sheet_rect().topLeft()
        return QRectF(origin.x() + rect_mm.x() * s, origin.y() + rect_mm.y() * s, rect_mm.width() * s, rect_mm.height() * s)

    def _frame_at(self, pos: QPointF) -> Optional[int]:
        if not self._picking:
            return None
        for index, rect_mm in self._frames:
            if self._frame_rect(rect_mm).contains(pos):
                return index
        return None

    def px_per_mm(self) -> float:
        """Logical pixels per millimetre the sheet is shown at."""
        if self._held_scale is not None:
            return self._held_scale
        w, h = self._paper
        avail_w = max(1.0, self.width() - 2 * _CANVAS_PAD)
        avail_h = max(1.0, self.height() - 2 * _CANVAS_PAD)
        return max(0.05, min(avail_w / w, avail_h / h))

    def sheet_rect(self) -> QRectF:
        s = self.px_per_mm()
        w, h = self._paper[0] * s, self._paper[1] * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _handles(self) -> tuple[dict[str, QPointF], dict[str, QPointF]]:
        r = self.sheet_rect()
        corners = {"tl": r.topLeft(), "tr": r.topRight(), "br": r.bottomRight(), "bl": r.bottomLeft()}
        edges = {
            "top": QPointF(r.center().x(), r.top()),
            "bottom": QPointF(r.center().x(), r.bottom()),
            "left": QPointF(r.left(), r.center().y()),
            "right": QPointF(r.right(), r.center().y()),
        }
        return corners, edges

    def _handle_at(self, pos: QPointF) -> Optional[str]:
        corners, edges = self._handles()
        return hit_resize_handle(pos, corners) or hit_resize_handle(pos, edges)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(THEME.canvas_bg_dark_grey))
        rect = self.sheet_rect()
        if self._image is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(rect, self._image)
        if self._picking:
            for index, rect_mm in self._frames:
                rect = self._frame_rect(rect_mm)
                if index in self._left_out:
                    painter.fillRect(rect, QColor(0, 0, 0, 170))
                    draw_mark_badge(painter, rect, check=False)
                if index == self._hover:
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.setPen(QPen(QColor(THEME.accent_primary), 2))
                    painter.drawRect(rect)
        corners, edges = self._handles()
        draw_resize_handles(painter, corners, edges)
        if self._drag is not None and self._readout:
            width = painter.fontMetrics().horizontalAdvance(self._readout) + 2 * THEME.space_lg
            draw_view_badge(painter, self._readout, (self.width() - width) / 2, THEME.space_lg, width)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        handle = self._handle_at(event.position())
        if event.button() == Qt.MouseButton.LeftButton and handle is not None:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._held_scale = self.px_per_mm()
            self._drag = handle
            event.accept()
            return
        frame = self._frame_at(event.position())
        if event.button() == Qt.MouseButton.LeftButton and frame is not None:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.frame_clicked.emit(frame)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self._drag is None:
            handle = self._handle_at(pos)
            frame = None if handle is not None else self._frame_at(pos)
            if frame != self._hover:
                self._hover = frame
                self.update()
            if handle is not None:
                self.setCursor(resize_cursor(handle))
            elif frame is not None:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.unsetCursor()
            return
        s = self.px_per_mm()
        width, height = self._paper
        cx, cy = self.width() / 2, self.height() / 2
        if self._drag in ("left", "right", "tl", "tr", "bl", "br"):
            width = 2 * abs(pos.x() - cx) / s
        if self._drag in ("top", "bottom", "tl", "tr", "bl", "br"):
            height = 2 * abs(pos.y() - cy) / s
        self.paper_dragged.emit(width, height)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover is not None:
            self._hover = None
            self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag is None:
            super().mouseReleaseEvent(event)
            return
        self._drag = None
        self._held_scale = None
        self.paper_released.emit()
        self.update()


class ContactSheetDialog(QDialog):
    """Lay the roll out on paper, then export. The frames arrive sorted and resolved; the
    dialog chooses the format, the paper and the resolution."""

    def __init__(
        self,
        frames: Sequence[SheetFrame],
        film_format: SheetFormat,
        frame_size: str,
        settings: ContactSheetSettings,
        look_for: Callable[[SheetFormat], SheetLook],
        label_text: str,
        out_dir: str,
        tiles: Optional[PreviewTiles] = None,
        lane_busy: Optional[Callable[[], str]] = None,
        proof: Optional[StraightProof] = None,
        scene_proof: Optional[StraightProof] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Contact Sheet")
        self.setModal(True)
        self.resize(1080, 760)

        self._frames = tuple(frames)
        self._format = film_format
        self._frame_size = frame_size if frame_size in FRAME_SIZES_120 else _FRAME_SIZES[1]
        self._look_for = look_for
        self._label_text = label_text
        self._out_dir = out_dir
        self._source = tiles
        self._lane_busy = lane_busy
        self._paper = (settings.paper_width, settings.paper_height)
        self._dpi = settings.dpi
        # Print variants: the frames as edited, and as a straight proof at the roll's or at each
        # scene's metering, when one can be made. Every variant keeps the frames' order.
        self._proof = proof if proof is not None else StraightProof(reason="")
        self._scene_proof = scene_proof if scene_proof is not None else StraightProof(reason="")
        self._variant_frames: list[Optional[tuple[SheetFrame, ...]]] = [
            self._frames,
            self._proof.frames if self._proof.available else None,
            self._scene_proof.frames if self._scene_proof.available else None,
        ]
        self._variant = _EDITED_TILES
        self._tile_sets: list[list[Optional[np.ndarray]]] = [[None] * len(self._frames) for _ in self._variant_frames]
        self._shape_sets: list[list[Optional[tuple[int, int]]]] = [[None] * len(self._frames) for _ in self._variant_frames]
        self._tiles_done = [0] * len(self._variant_frames)
        self._generations: dict[int, int] = {}
        self._tiles = self._tile_sets[_EDITED_TILES]
        self._tile_shapes = self._shape_sets[_EDITED_TILES]
        self._turns = [0] * len(self._frames)
        self._has_scenes = any(scene_of(frame) for frame in self._frames)
        self._by_scene = settings.by_scene and self._has_scenes
        self._display = scene_order(self._frames) if self._by_scene else list(range(len(self._frames)))
        self._page = 0
        self._plan: Optional[SheetPlan] = None
        self._look = look_for(film_format)
        # Frames rejected in triage start left out; a click in Pick Frames puts them back.
        self._left_out: set[int] = {i for i, frame in enumerate(self._frames) if frame.asset.get("excluded")}

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render)

        root = QHBoxLayout(self)
        root.setContentsMargins(THEME.space_2xl, THEME.space_2xl, THEME.space_2xl, THEME.space_2xl)
        root.setSpacing(THEME.space_2xl)

        left = QVBoxLayout()
        left.setSpacing(THEME.space_md)
        self.canvas = SheetCanvas()
        self.canvas.paper_dragged.connect(self._on_paper_dragged)
        self.canvas.paper_released.connect(self._on_paper_released)
        self.canvas.frame_clicked.connect(self._on_frame_clicked)
        left.addWidget(self.canvas, 1)
        nav = QHBoxLayout()
        nav.addStretch()
        self.prev_btn = icon_button("fa5s.chevron-left", "Previous sheet")
        self.prev_btn.clicked.connect(lambda: self._set_page(self._page - 1))
        self.page_label = hint_label()
        self.next_btn = icon_button("fa5s.chevron-right", "Next sheet")
        self.next_btn.clicked.connect(lambda: self._set_page(self._page + 1))
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.page_label)
        nav.addWidget(self.next_btn)
        nav.addStretch()
        left.addLayout(nav)
        root.addLayout(left, 1)

        root.addWidget(self._build_controls(settings))

        self._set_paper(*self._paper, snap=False)
        pin_dialog_default(self.export_btn, scope=self)
        align_slider_columns(self)

        if self._source is not None and self._frames:
            self._source.tile_ready.connect(self._on_tile_ready)
        self._apply_variant()

    # ── controls ────────────────────────────────────────────────────────

    def _build_controls(self, settings: ContactSheetSettings) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(340)
        col = QVBoxLayout(panel)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(THEME.space_lg)

        self.format_btn = ChoiceButton(
            tuple(("", fmt.value) for fmt in _FORMATS),
            "The film the frames were shot on; read from the frames' Format metadata and the roll's Half Frame mode",
        )
        self.format_btn.setCurrentIndex(_FORMATS.index(self._format))
        self.format_btn.currentChanged.connect(self._on_format_changed)
        col.addLayout(self._row("Format", self.format_btn))

        self.frame_size_btn = ChoiceButton(tuple(("", size) for size in _FRAME_SIZES), "The 120 camera's frame size")
        self.frame_size_btn.setCurrentIndex(_FRAME_SIZES.index(self._frame_size))
        self.frame_size_btn.currentChanged.connect(self._on_frame_size_changed)
        self.frame_size_rail = SliderGroup(self._row("Frame size", self.frame_size_btn))
        col.addWidget(self.frame_size_rail)

        paper_row = QHBoxLayout()
        label = field_label("Paper")
        label.setFixedWidth(FIELD_LABEL_WIDTH)
        paper_row.addWidget(label)
        self.paper_btn = ChoiceButton(
            tuple(("", paper.label) for paper in ILFORD_PAPERS) + (("", "Custom"),),
            "Ilford sheet size, or Custom for a size dragged or typed",
        )
        self.paper_btn.currentChanged.connect(self._on_paper_choice)
        paper_row.addWidget(self.paper_btn, 1)
        self.turn_btn = icon_button("fa5s.sync-alt", "Turn the paper (swap width and height)")
        self.turn_btn.clicked.connect(lambda: self._set_paper(self._paper[1], self._paper[0], snap=False))
        paper_row.addWidget(self.turn_btn)
        col.addLayout(paper_row)

        self.width_slider = CompactSlider("Width", PAPER_MIN, PAPER_MAX, DEFAULT_PAPER.width, step=0.1, precision=10, unit=" mm")
        self.width_slider.setToolTip("Paper width; drag the sheet's left or right edge to change it")
        self.height_slider = CompactSlider("Height", PAPER_MIN, PAPER_MAX, DEFAULT_PAPER.height, step=0.1, precision=10, unit=" mm")
        self.height_slider.setToolTip("Paper height; drag the sheet's top or bottom edge to change it")
        for slider in (self.width_slider, self.height_slider):
            slider.valueChanged.connect(self._on_size_slider)
            # Return in a size box commits the size; without this the dialog's Export takes it.
            slider.spin.installEventFilter(self)
        col.addWidget(SliderGroup(self.width_slider, self.height_slider))

        self.dpi_btn = ChoiceButton(
            tuple(("", str(dpi)) for dpi in DPI_CHOICES), "Resolution of the exported sheet; the file prints at its true size at this DPI"
        )
        self.dpi_btn.setCurrentIndex(DPI_CHOICES.index(self._dpi) if self._dpi in DPI_CHOICES else 1)
        self.dpi_btn.currentChanged.connect(self._on_dpi_changed)
        col.addLayout(self._row("DPI", self.dpi_btn))

        self.order_btn = ChoiceButton(
            tuple(("", order) for order in _ORDERS),
            "Date lays the frames out in the order they were created. Scene groups them by scene, "
            "each scene starting a new strip, and a straight proof then meters each scene on its own",
        )
        scene_action = self.order_btn.choice_menu.actions()[_BY_SCENE]
        scene_action.setEnabled(self._has_scenes)
        if not self._has_scenes:
            scene_action.setToolTip("No frame of this roll is in a scene")
        self.order_btn.setCurrentIndex(_BY_SCENE if self._by_scene else _BY_DATE)
        self.order_btn.currentChanged.connect(self._on_order_changed)
        col.addLayout(self._row("Order", self.order_btn))

        self.print_btn = ChoiceButton(
            tuple(("", name) for name in _PRINTS),
            "As Edited prints each frame with its own edit. Straight Proof prints the whole roll at one "
            "exposure and grade, as a darkroom proof, so thin and dense negatives show as they are",
        )
        self.print_btn.currentChanged.connect(lambda _index: self._apply_variant())
        col.addLayout(self._row("Print", self.print_btn))
        self.proof_label = hint_label()
        self.proof_label.setWordWrap(True)
        col.addWidget(self.proof_label)

        toggles = QHBoxLayout()
        toggles.setSpacing(THEME.space_md)
        self.edge_btn = labeled_toggle(
            "fa5s.barcode",
            " Edge Print",
            settings.edge_print,
            "Print the maker's edge markings: stock name, frame numbers and the DX barcode. Off leaves plain film",
        )
        self.edge_btn.toggled.connect(lambda _checked: self._restyle())
        toggles.addWidget(self.edge_btn, 1)
        self.label_btn = labeled_toggle(
            "fa5s.tag", " Roll Label", settings.roll_label, "Print the roll name, film, developer, camera and date above the strips"
        )
        self.label_btn.toggled.connect(self._on_label_toggled)
        toggles.addWidget(self.label_btn, 1)
        col.addLayout(toggles)

        self.pick_btn = tool_toggle(
            "fa5s.times-circle",
            "Pick Frames",
            "Show every frame; click one to leave it out of the sheet or put it back. Frames rejected in the Film Strip start left out",
        )
        self.pick_btn.toggled.connect(lambda _checked: self._schedule_render())
        col.addWidget(self.pick_btn)

        self.summary_label = hint_label()
        self.summary_label.setWordWrap(True)
        col.addWidget(self.summary_label)
        self.folder_label = hint_label(f"Saves to {self._out_dir}" if self._out_dir else "")
        self.folder_label.setWordWrap(True)
        col.addWidget(self.folder_label)
        col.addStretch()

        footer = QHBoxLayout()
        self.info_btn = icon_button("fa5s.info-circle", "How the contact sheet is laid out")
        self.info_btn.setVisible(has_guide(_GUIDE_KEY))
        self.info_btn.clicked.connect(lambda: SectionHelpDialog(_GUIDE_KEY, "Contact Sheet", self).exec())
        footer.addWidget(self.info_btn)
        footer.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._on_export)
        footer.addWidget(self.cancel_btn)
        footer.addWidget(self.export_btn)
        col.addLayout(footer)

        self.frame_size_rail.setVisible(self._format == SheetFormat.MEDIUM)
        return panel

    @staticmethod
    def _row(text: str, control: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        label = field_label(text)
        label.setFixedWidth(FIELD_LABEL_WIDTH)
        row.addWidget(label)
        row.addWidget(control, 1)
        return row

    # ── state ───────────────────────────────────────────────────────────

    def _geometry(self):
        return film_geometry(self._format, self._frame_size)

    def _label_on(self) -> bool:
        return self.label_btn.isChecked()

    def _set_paper(self, width: float, height: float, snap: bool) -> None:
        if snap:
            snapped = snap_paper(width, height)
            if paper_preset(*snapped) is None:
                # Off an Ilford size only the dragged side moves; the other keeps its exact value.
                snapped = (
                    snapped[0] if width != self._paper[0] else width,
                    snapped[1] if height != self._paper[1] else height,
                )
            width, height = snapped
        width, height = clamp_paper(width, height, self._geometry(), self._label_on())
        self._paper = (round(width, 1), round(height, 1))
        for slider, value in ((self.width_slider, self._paper[0]), (self.height_slider, self._paper[1])):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
        preset = paper_preset(*self._paper)
        self.paper_btn.blockSignals(True)
        self.paper_btn.setCurrentIndex(ILFORD_PAPERS.index(preset) if preset else _CUSTOM_PAPER)
        self.paper_btn.blockSignals(False)
        self._refresh_choices()
        readout = f"{_mm(self._paper[0])} × {_mm(self._paper[1])} mm" + (f" · {preset.label}" if preset else "")
        self.canvas.set_paper(*self._paper, readout)
        self._schedule_render()

    def _refresh_choices(self) -> None:
        """Paper sizes that cannot hold one frame, and resolutions over the pixel budget, are
        not offered; a current choice that became one steps down."""
        geometry = self._geometry()
        landscape = self._paper[0] > self._paper[1]
        for i, paper in enumerate(ILFORD_PAPERS):
            w, h = (paper.height, paper.width) if landscape else (paper.width, paper.height)
            per_strip, strips = plan_capacity(w, h, geometry, self._label_on())
            self.paper_btn.choice_menu.actions()[i].setEnabled(per_strip > 0 and strips > 0)
        for i, dpi in enumerate(DPI_CHOICES):
            self.dpi_btn.choice_menu.actions()[i].setEnabled(dpi_allowed(*self._paper, dpi))
        allowed = best_dpi(*self._paper, self._dpi)
        if allowed != self._dpi:
            self._dpi = allowed
        self.dpi_btn.blockSignals(True)
        self.dpi_btn.setCurrentIndex(DPI_CHOICES.index(self._dpi))
        self.dpi_btn.blockSignals(False)

    def _on_paper_dragged(self, width: float, height: float) -> None:
        self._set_paper(width, height, snap=True)

    def _on_paper_released(self) -> None:
        self._schedule_render()

    def _on_size_slider(self, _value: float) -> None:
        self._set_paper(self.width_slider.value(), self.height_slider.value(), snap=False)

    def _on_paper_choice(self, index: int) -> None:
        if index >= len(ILFORD_PAPERS):
            return
        paper = ILFORD_PAPERS[index]
        if self._paper[0] > self._paper[1]:
            self._set_paper(paper.height, paper.width, snap=False)
        else:
            self._set_paper(paper.width, paper.height, snap=False)

    def _on_format_changed(self, index: int) -> None:
        self._format = _FORMATS[index]
        self.frame_size_rail.setVisible(self._format == SheetFormat.MEDIUM)
        self._film_changed()

    def _on_frame_size_changed(self, index: int) -> None:
        self._frame_size = _FRAME_SIZES[index]
        self._film_changed()

    def _restyle(self) -> None:
        self._look = self._look_for(self._format)
        self._schedule_render()

    def _on_frame_clicked(self, index: int) -> None:
        self._left_out ^= {index}
        self._schedule_render()

    def _shown(self) -> list[int]:
        """Frames the preview lays out, in sheet order: all of them while picking, else the ones printed."""
        if self.pick_btn.isChecked():
            return list(self._display)
        return [i for i in self._display if i not in self._left_out]

    def _breaks(self, indices: Sequence[int]) -> list[int]:
        return scene_breaks([self._frames[i] for i in indices]) if self._by_scene else []

    def _film_changed(self) -> None:
        """A new film turns the paper when that holds the roll on fewer sheets."""
        geometry = self._geometry()
        self._look = self._look_for(self._format)
        self._turns = [turns_for(frame, geometry, shape) if shape else 0 for frame, shape in zip(self._frames, self._tile_shapes)]
        width, height = better_orientation(*self._paper, geometry, len(self.kept_frames()), self._label_on())
        self._set_paper(width, height, snap=False)

    def _on_dpi_changed(self, index: int) -> None:
        self._dpi = DPI_CHOICES[index]

    def _on_label_toggled(self, _checked: bool) -> None:
        self._set_paper(*self._paper, snap=False)

    def _set_page(self, page: int) -> None:
        self._page = page
        self._schedule_render()

    def _request_tiles(self, variant: int) -> None:
        frames = self._variant_frames[variant]
        if self._source is None or frames is None or variant in self._generations.values():
            return
        self._generations[self._source.render(frames)] = variant

    def _on_order_changed(self, index: int) -> None:
        self._by_scene = index == _BY_SCENE
        self._display = scene_order(self._frames) if self._by_scene else list(range(len(self._frames)))
        self._apply_variant()

    def _apply_variant(self) -> None:
        """Point the preview at the tiles of the chosen print and order. A straight proof that
        cannot be made in this order falls back to the frames as edited, and says why."""
        proof = self._scene_proof if self._by_scene else self._proof
        proof_action = self.print_btn.choice_menu.actions()[_STRAIGHT_PROOF]
        proof_action.setEnabled(proof.available)
        proof_action.setToolTip(proof.reason)
        proofing = self.print_btn.currentIndex() == _STRAIGHT_PROOF
        if proofing and not proof.available:
            self.print_btn.blockSignals(True)
            self.print_btn.setCurrentIndex(_AS_EDITED)
            self.print_btn.blockSignals(False)
            proofing = False
        if not proofing:
            variant = _EDITED_TILES
        else:
            variant = _SCENE_PROOF_TILES if self._by_scene else _PROOF_TILES
        self._variant = variant
        self._tiles = self._tile_sets[variant]
        self._tile_shapes = self._shape_sets[variant]
        geometry = self._geometry()
        self._turns = [turns_for(frame, geometry, shape) if shape else 0 for frame, shape in zip(self._frames, self._tile_shapes)]
        self.proof_label.setText(proof.note if proofing else proof.reason)
        self.proof_label.setVisible(bool(self.proof_label.text()))
        self._request_tiles(variant)
        self._schedule_render()

    def _on_tile_ready(self, generation: int, index: int, tile) -> None:
        variant = self._generations.get(generation)
        if variant is None or not 0 <= index < len(self._frames):
            return
        self._tiles_done[variant] += 1
        if tile is not None:
            self._tile_sets[variant][index] = tile
            self._shape_sets[variant][index] = tile.shape[:2]
        if variant != self._variant:
            return
        if tile is not None:
            self._turns[index] = turns_for(self._frames[index], self._geometry(), tile.shape[:2])
        if not self._render_timer.isActive():
            self._render_timer.start(_TILE_REPAINT_MS)

    def _schedule_render(self) -> None:
        self._render_timer.start(0)

    # ── render ──────────────────────────────────────────────────────────

    def plan(self) -> SheetPlan:
        """The sheet as it exports: the frames not left out."""
        kept = self.numbers()
        return plan_sheets(*self._paper, self._geometry(), len(kept), self._label_on(), self._breaks(kept))

    def _render(self) -> None:
        shown = self._shown()
        plan = plan_sheets(*self._paper, self._geometry(), len(shown), self._label_on(), self._breaks(shown))
        self._plan = plan
        pages = len(plan.pages)
        self._page = max(0, min(self._page, pages - 1)) if pages else 0
        self._refresh_summary(plan)
        dpr = self.canvas.devicePixelRatioF() or 1.0
        px_per_mm = self.canvas.px_per_mm() * dpr
        tiles = [self._tiles[i] for i in shown]
        turns = [self._turns[i] for i in shown]
        sheet = ContactSheetService.render_sheet(plan, self._page, tiles, turns, px_per_mm, self.look(), draft=True, numbers=shown)
        sheet = np.ascontiguousarray(sheet)
        image = QImage(sheet.data, sheet.shape[1], sheet.shape[0], sheet.strides[0], QImage.Format.Format_RGB888).copy()
        image.setDevicePixelRatio(dpr)
        self.canvas.set_image(image)
        self.canvas.set_frames(self._frame_windows(plan, shown), self._left_out, self.pick_btn.isChecked())

    def _frame_windows(self, plan: SheetPlan, shown: list[int]) -> list[tuple[int, QRectF]]:
        if not plan.pages:
            return []
        geometry = plan.geometry
        windows = []
        for strip in plan.pages[self._page].strips:
            for slot in range(strip.first, strip.first + strip.count):
                left = strip.x + geometry.frame_center(slot) - strip.roll_start - geometry.frame_along / 2
                rect = QRectF(left, strip.y + geometry.image_top, geometry.frame_along, geometry.frame_across)
                windows.append((shown[slot], rect))
        return windows

    def _refresh_summary(self, plan: SheetPlan) -> None:
        pages = len(plan.pages)
        self.prev_btn.setVisible(pages > 1)
        self.next_btn.setVisible(pages > 1)
        self.page_label.setVisible(pages > 1)
        self.prev_btn.setEnabled(self._page > 0)
        self.next_btn.setEnabled(self._page < pages - 1)
        self.page_label.setText(f"Sheet {self._page + 1} of {pages}")
        final = plan if not self.pick_btn.isChecked() else self.plan()
        self.export_btn.setEnabled(bool(final.pages))
        if not final.pages:
            self.summary_label.setText(final.reason)
            set_hint_kind(self.summary_label, "warning")
            return
        set_hint_kind(self.summary_label, "muted")
        count = len(self.kept_frames())
        strips = sum(len(page.strips) for page in final.pages)
        frames = "frame" if count == 1 else "frames"
        strip_word = "strip" if strips == 1 else "strips"
        sheets = "sheet" if len(final.pages) == 1 else "sheets"
        text = f"{count} {frames} · {strips} {strip_word} of {min(final.frames_per_strip, count)} · {len(final.pages)} {sheets}"
        if self._left_out:
            text += f" · {len(self._left_out)} left out"
        done = self._tiles_done[self._variant]
        if self._source is not None and done < len(self._frames):
            text += f" · preview {done}/{len(self._frames)}"
        self.summary_label.setText(text)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self.canvas.dragging():
            self._schedule_render()

    # ── result ──────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if obj in (self.width_slider.spin, self.height_slider.spin):
                obj.interpretText()
                obj.clearFocus()
                self._set_paper(self.width_slider.value(), self.height_slider.value(), snap=False)
                return True
        return super().eventFilter(obj, event)

    def _on_export(self) -> None:
        busy = self._lane_busy() if self._lane_busy else ""
        if busy:
            self.summary_label.setText(busy)
            set_hint_kind(self.summary_label, "warning")
            return
        self.accept()

    def done(self, result: int) -> None:
        self._render_timer.stop()
        if self._source is not None:
            for generation in self._generations:
                self._source.cancel(generation)
            try:
                self._source.tile_ready.disconnect(self._on_tile_ready)
            except (TypeError, RuntimeError):
                pass
        super().done(result)

    def settings(self) -> ContactSheetSettings:
        return ContactSheetSettings(
            self._paper[0],
            self._paper[1],
            self._dpi,
            self._label_on(),
            self.edge_btn.isChecked(),
            self.order_btn.currentIndex() == _BY_SCENE,
        )

    def kept_frames(self) -> tuple[SheetFrame, ...]:
        """The frames to print in sheet order, each with the config of the chosen print."""
        frames = self._variant_frames[self._variant] or self._frames
        return tuple(frames[i] for i in self.numbers())

    def straight_proof(self) -> bool:
        return self._variant != _EDITED_TILES

    def numbers(self) -> tuple[int, ...]:
        """Each kept frame's place on the roll, in sheet order, so its edge numbers stay its own."""
        return tuple(i for i in self._display if i not in self._left_out)

    def breaks(self) -> tuple[int, ...]:
        """Where a new scene starts a new strip among the kept frames."""
        return tuple(self._breaks(self.numbers()))

    def film(self) -> tuple[SheetFormat, str]:
        return self._format, self._frame_size

    def look(self) -> SheetLook:
        edge = replace(self._look.edge, printed=self.edge_btn.isChecked())
        return replace(self._look, edge=edge, label=self._label_text if self._label_on() else "")
