"""Coolscan roll-scanning sidebar.

Device selection, a whole-roll preview rendered as a thumbnail contact
sheet, per-slot spacing-offset nudge and approval, and one-button batch
fine-scan with progress and a safe-stop control. Mirrors the structure of
`ScanlightSidebar` (gating via `_apply_gating()`, settings persisted the
same way, one status line + progress bar) as closely as the workflow
allows -- preview/approve/batch-scan has no camera-route analogue, so the
contact sheet and per-slot controls below are new, built the simplest way
that fits the surrounding code.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import qtawesome as qta
from PyQt6.QtCore import QSize, Qt, pyqtSlot
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.roll import coolscanpy_roll
from negpy.infrastructure.roll.settings import RollScanSettings

_SLOT_ROLE = Qt.ItemDataRole.UserRole
_WARN_COLOR = "#C8922E"  # matches ScanlightSidebar's advisory tone
_THUMBNAIL_SIZE = QSize(96, 96)


def _thumbnail_pixmap(image) -> QPixmap:
    """A `coolscanpy.Thumbnail.image` array as a displayable QPixmap.

    Deliberately dumb: this is a raw scanner preview frame for framing and
    review, not a NegPy pipeline buffer, so it does not go through
    `ImageConverter.to_qimage`'s color management -- that helper assumes a
    working-space buffer this array isn't. Normalizes whatever numeric
    range/dtype/channel-count arrives to 8-bit RGB.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        peak = float(arr.max()) if arr.size else 1.0
        scale = 255.0 if peak <= 1.0 else (255.0 / peak if peak > 255.0 else 1.0)
        arr = np.clip(arr.astype(np.float64) * scale, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr[..., :3])
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()  # copy: detach from arr's buffer
    return QPixmap.fromImage(qimg)


class CoolscanRollSidebar(QWidget):
    """Whole-roll preview, spacing/approval and batch fine-scan panel."""

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._settings: RollScanSettings = self._load_settings()
        self._devices: list = []
        self._devices_loaded = False
        self._thumbnails: dict[int, object] = {}
        self._scanning = False
        self._stopping = False  # safe-stop acknowledged, waiting for the in-flight frame

        self._init_ui()
        self._connect_signals()

    # ── settings persistence ──────────────────────────────────────────

    def _load_settings(self) -> RollScanSettings:
        data = self.controller.session.repo.get_global_setting("roll_scan_settings", default={})
        if isinstance(data, dict) and data:
            try:
                known = {f.name for f in dataclasses.fields(RollScanSettings)}
                return RollScanSettings(**{k: v for k, v in data.items() if k in known})
            except Exception:
                pass
        return RollScanSettings.defaults()

    def _save_settings(self) -> None:
        self.controller.session.repo.save_global_setting("roll_scan_settings", dataclasses.asdict(self._settings))

    def _update_settings_from_ui(self) -> None:
        updated = dataclasses.replace(
            self._settings,
            last_device_id=self._current_device_id() or self._settings.last_device_id,
            output_folder=self.folder_edit.text().strip(),
            filename_pattern=self.pattern_edit.text().strip() or RollScanSettings.defaults().filename_pattern,
        )
        if updated == self._settings:
            return
        self._settings = updated
        self._save_settings()
        self._apply_gating()

    # ── UI construction ───────────────────────────────────────────────

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)
        layout.setSpacing(10)

        self.preview_btn = QPushButton(qta.icon("fa5s.images", color=THEME.text_primary), " Preview Roll")
        self.preview_btn.setObjectName("scan_btn")
        self.preview_btn.setFixedHeight(40)
        layout.addWidget(self.preview_btn)

        self.gate_hint = QLabel("")
        self.gate_hint.setStyleSheet(f"color: {_WARN_COLOR}; font-size: {THEME.font_size_small}px;")
        self.gate_hint.setWordWrap(True)
        layout.addWidget(self.gate_hint)

        # ── DEVICE ──────────────────────────────────────────
        layout.addWidget(section_subheader("DEVICE"))
        self._setup_hint = QLabel(
            "Roll scanning needs coolscanpy, an optional dependency: `pip install coolscanpy`. See docs/COOLSCANPY_ROLL_SCANNING.md."
        )
        self._setup_hint.setWordWrap(True)
        self._setup_hint.setStyleSheet(f"color: {_WARN_COLOR}; font-size: {THEME.font_size_small}px;")
        self._setup_hint.setVisible(not coolscanpy_roll.available())
        layout.addWidget(self._setup_hint)

        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setToolTip("Select a Coolscan device")
        self.device_combo.addItem("Detecting devices…", None)
        self.refresh_btn = QPushButton()
        self.refresh_btn.setIcon(qta.icon("fa5s.redo", color=THEME.text_secondary))
        self.refresh_btn.setToolTip("Refresh device list")
        self.refresh_btn.setFixedWidth(32)
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_btn)
        layout.addLayout(device_row)

        # ── OUTPUT ──────────────────────────────────────────
        layout.addWidget(section_subheader("OUTPUT"))
        out_form = QFormLayout()
        out_form.setSpacing(6)
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit(self._settings.output_folder)
        self.folder_edit.setPlaceholderText("Output folder…")
        self.folder_browse = QPushButton("…")
        self.folder_browse.setFixedWidth(32)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(self.folder_browse)
        out_form.addRow("Folder", folder_row)
        self.pattern_edit = QLineEdit(self._settings.filename_pattern)
        self.pattern_edit.setToolTip('Jinja2 template. Variables: {{ date }}, {{ seq }} (the slot number).\nExample: {{ date }}_{{ "%03d" % seq }}')
        out_form.addRow("Filename", self.pattern_edit)
        layout.addLayout(out_form)

        # ── PROGRESS / STATUS ────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        # ── CONTACT SHEET ─────────────────────────────────────
        layout.addWidget(section_subheader("CONTACT SHEET"))
        self.contact_sheet = QListWidget()
        self.contact_sheet.setViewMode(QListWidget.ViewMode.IconMode)
        self.contact_sheet.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.contact_sheet.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.contact_sheet.setIconSize(_THUMBNAIL_SIZE)
        self.contact_sheet.setMinimumHeight(160)
        self.contact_sheet.setToolTip("Click a slot to nudge its spacing or approve it; select several, then Scan Selected.")
        layout.addWidget(self.contact_sheet)

        slot_row = QFormLayout()
        slot_row.setSpacing(6)
        self.slot_label = QLabel("—")
        slot_row.addRow("Slot", self.slot_label)
        offset_row = QHBoxLayout()
        self.offset_spin = QSpinBox()
        self.offset_spin.setRange(-500, 500)
        self.offset_spin.setToolTip("Nudge this slot's transport boundary, in native rows at preview resolution")
        self.offset_apply_btn = QPushButton("Apply")
        offset_row.addWidget(self.offset_spin, 1)
        offset_row.addWidget(self.offset_apply_btn)
        slot_row.addRow("Spacing offset", offset_row)
        layout.addLayout(slot_row)

        self.approve_btn = QPushButton(qta.icon("fa5s.check", color=THEME.text_primary), " Approve Slot")
        self.approve_btn.setToolTip("This slot's transport origin was not confidently automatic; approve it before it can be scanned.")
        layout.addWidget(self.approve_btn)

        # ── SCAN ────────────────────────────────────────────
        self.scan_btn = QPushButton(qta.icon("fa5s.camera-retro", color=THEME.text_primary), " Scan Selected")
        self.scan_btn.setObjectName("scan_btn")
        self.scan_btn.setFixedHeight(40)
        layout.addWidget(self.scan_btn)

        self.safe_stop_btn = QPushButton(qta.icon("fa5s.stop", color=THEME.text_secondary), " Safe Stop")
        self.safe_stop_btn.setToolTip("Finish the frame in flight, then stop before the next one -- the transport can't be aborted mid-pull.")
        self.safe_stop_btn.setEnabled(False)
        layout.addWidget(self.safe_stop_btn)

        self._show_slot_detail(None)
        self._apply_gating()
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.folder_browse.clicked.connect(self._on_browse_folder)
        for w in (self.folder_edit, self.pattern_edit):
            w.editingFinished.connect(self._update_settings_from_ui)
        self.preview_btn.clicked.connect(self._on_preview_clicked)
        self.contact_sheet.itemSelectionChanged.connect(self._on_selection_changed)
        self.offset_apply_btn.clicked.connect(self._on_apply_offset)
        self.approve_btn.clicked.connect(self._on_approve)
        self.scan_btn.clicked.connect(self._on_scan_clicked)
        self.safe_stop_btn.clicked.connect(self._on_safe_stop_clicked)

        self.controller.roll_devices_ready.connect(self._on_devices_ready)
        self.controller.roll_opened.connect(self._on_opened)
        self.controller.roll_preview_ready.connect(self._on_preview_ready)
        self.controller.roll_spacing_offset_set.connect(self._on_spacing_offset_set)
        self.controller.roll_approved.connect(self._on_approved)
        self.controller.roll_progress.connect(self._on_progress)
        self.controller.roll_frame_written.connect(self._on_frame_written)
        self.controller.roll_finished.connect(self._on_finished)
        self.controller.roll_cancelled.connect(self._on_cancelled)
        self.controller.roll_error.connect(self._on_error)
        self.controller.roll_status.connect(self._on_status)

    # ── activation hook ───────────────────────────────────────────────

    def on_activated(self) -> None:
        """Called when the Scan tab is switched to."""
        self._setup_hint.setVisible(not coolscanpy_roll.available())
        self._apply_gating()
        if not self._devices_loaded:
            self._request_devices()

    # ── devices ───────────────────────────────────────────────────────

    def _request_devices(self) -> None:
        if not coolscanpy_roll.available():
            return
        self.device_combo.clear()
        self.device_combo.addItem("Detecting devices…", None)
        self.device_combo.setEnabled(False)
        self.controller.request_roll_devices()

    def _on_refresh(self) -> None:
        self._devices_loaded = False
        self._request_devices()

    @pyqtSlot(list)
    def _on_devices_ready(self, devices: list) -> None:
        self._devices = devices
        self._devices_loaded = True
        self.device_combo.clear()
        self.device_combo.setEnabled(True)

        if not devices:
            self.device_combo.addItem("No Coolscan devices detected", None)
            self.device_combo.setEnabled(False)
            self._apply_gating()
            return

        for d in devices:
            label = f"{d.vendor} {d.model}" if getattr(d, "vendor", "") else d.model
            self.device_combo.addItem(label, d.id)

        if self._settings.last_device_id:
            idx = self.device_combo.findData(self._settings.last_device_id)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)

        self._apply_gating()

    def _on_device_changed(self, _index: int) -> None:
        self._clear_contact_sheet()
        self._update_settings_from_ui()
        self._apply_gating()

    def _current_device_id(self) -> str | None:
        return self.device_combo.currentData()

    @pyqtSlot(str)
    def _on_opened(self, device_id: str) -> None:
        self._set_status(f"Roll open on {device_id}.")

    # ── preview / contact sheet ─────────────────────────────────────

    def _clear_contact_sheet(self) -> None:
        self.contact_sheet.clear()
        self._thumbnails = {}
        self._show_slot_detail(None)

    def _on_preview_clicked(self) -> None:
        device_id = self._current_device_id()
        if not device_id:
            return
        from negpy.desktop.workers.roll_worker import RollPreviewRequest

        self._clear_contact_sheet()
        self._set_status("Reading roll transport…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.controller.start_roll_preview(RollPreviewRequest(device_id=device_id))

    @pyqtSlot(list)
    def _on_preview_ready(self, thumbnails: list) -> None:
        self.progress_bar.setVisible(False)
        self._thumbnails = {t.slot: t for t in thumbnails}
        self.contact_sheet.clear()
        for t in sorted(self._thumbnails):
            self._add_slot_item(self._thumbnails[t])
        self._show_slot_detail(None)
        self._apply_gating()

    def _add_slot_item(self, thumb) -> None:
        pixmap = _thumbnail_pixmap(thumb.image).scaled(
            _THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        item = QListWidgetItem(QIcon(pixmap), f"Slot {thumb.slot}" + (" ⚠" if thumb.needs_approval else ""))
        item.setData(_SLOT_ROLE, thumb.slot)
        if thumb.needs_approval:
            item.setForeground(QColor(_WARN_COLOR))
        self.contact_sheet.addItem(item)

    def _on_selection_changed(self) -> None:
        current = self.contact_sheet.currentItem()
        slot = current.data(_SLOT_ROLE) if current is not None else None
        self._show_slot_detail(self._thumbnails.get(slot) if slot is not None else None)
        self._apply_gating()

    def _show_slot_detail(self, thumb) -> None:
        if thumb is None:
            self.slot_label.setText("—")
            self.offset_spin.setEnabled(False)
            self.offset_apply_btn.setEnabled(False)
            self.approve_btn.setVisible(False)
            return
        self.slot_label.setText(str(thumb.slot))
        self.offset_spin.setEnabled(True)
        self.offset_apply_btn.setEnabled(True)
        self.offset_spin.blockSignals(True)
        self.offset_spin.setValue(int(thumb.spacing_offset))
        self.offset_spin.blockSignals(False)
        self.approve_btn.setVisible(bool(thumb.needs_approval))

    def _selected_slots(self) -> list[int]:
        return sorted({item.data(_SLOT_ROLE) for item in self.contact_sheet.selectedItems()})

    # ── spacing offset / approval ────────────────────────────────────

    def _on_apply_offset(self) -> None:
        current = self.contact_sheet.currentItem()
        if current is None:
            return
        slot = current.data(_SLOT_ROLE)
        self.controller.set_roll_spacing_offset(slot, self.offset_spin.value())

    @pyqtSlot(int, int)
    def _on_spacing_offset_set(self, slot: int, offset_rows: int) -> None:
        thumb = self._thumbnails.get(slot)
        if thumb is not None:
            self._thumbnails[slot] = dataclasses.replace(thumb, spacing_offset=offset_rows)
        self._set_status(f"Slot {slot}: spacing offset set to {offset_rows}.")

    def _on_approve(self) -> None:
        current = self.contact_sheet.currentItem()
        if current is None:
            return
        self.controller.approve_roll_slot(current.data(_SLOT_ROLE))

    @pyqtSlot(int)
    def _on_approved(self, slot: int) -> None:
        thumb = self._thumbnails.get(slot)
        if thumb is not None:
            self._thumbnails[slot] = dataclasses.replace(thumb, needs_approval=False)
        for i in range(self.contact_sheet.count()):
            item = self.contact_sheet.item(i)
            if item.data(_SLOT_ROLE) == slot:
                item.setText(f"Slot {slot}")
                item.setData(Qt.ItemDataRole.ForegroundRole, None)  # back to the theme's default text color
                break
        if self.contact_sheet.currentItem() is not None and self.contact_sheet.currentItem().data(_SLOT_ROLE) == slot:
            self.approve_btn.setVisible(False)
        self._set_status(f"Slot {slot} approved.")
        self._apply_gating()

    # ── scan ──────────────────────────────────────────────────────────

    def _on_scan_clicked(self) -> None:
        device_id = self._current_device_id()
        slots = self._selected_slots()
        output_folder = self.folder_edit.text().strip()
        if not device_id or not slots or not output_folder:
            return
        from negpy.desktop.workers.roll_worker import RollBatchScanRequest

        self._update_settings_from_ui()
        self._save_settings()
        req = RollBatchScanRequest(
            device_id=device_id,
            slots=tuple(slots),
            output_folder=output_folder,
            filename_pattern=self.pattern_edit.text().strip() or RollScanSettings.defaults().filename_pattern,
        )
        self.set_scanning(True)
        self.controller.start_roll_scan(req)

    def _on_safe_stop_clicked(self) -> None:
        self._stopping = True
        self.safe_stop_btn.setEnabled(False)
        self._set_status("Stopping after the current frame…")
        self.controller.roll_safe_stop()

    @pyqtSlot(float, str)
    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(fraction * 100))
        if not self._stopping:
            self._set_status(message)

    @pyqtSlot(object)
    def _on_frame_written(self, output) -> None:
        if not self._stopping:
            self._set_status(f"Wrote slot {output.slot}.")

    @pyqtSlot(list)
    def _on_finished(self, outputs: list) -> None:
        self.set_scanning(False)
        self._set_status(f"Scanned {len(outputs)} frame(s).")

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        self.set_scanning(False)
        self._set_status("Stopped.")

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self.set_scanning(False)
        self.progress_bar.setVisible(False)
        self._set_status(f"Error: {msg}")

    @pyqtSlot(str)
    def _on_status(self, msg: str) -> None:
        if not self._stopping:
            self._set_status(msg)

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.status_label.setVisible(bool(text))

    # ── browse ────────────────────────────────────────────────────────

    def _on_browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.folder_edit.setText(folder)
            self._update_settings_from_ui()

    # ── gating ────────────────────────────────────────────────────────

    def _missing_for_preview(self) -> list[str]:
        m = []
        if not coolscanpy_roll.available():
            m.append("install coolscanpy")
        if not self._current_device_id():
            m.append("select a device")
        return m

    def _missing_for_scan(self) -> list[str]:
        m = list(self._missing_for_preview())
        if not self._selected_slots():
            m.append("select at least one slot")
        if not self.folder_edit.text().strip():
            m.append("choose an output folder")
        return m

    def _apply_gating(self) -> None:
        missing_preview = self._missing_for_preview()
        missing_scan = self._missing_for_scan()
        self.preview_btn.setEnabled(not missing_preview and not self._scanning)
        self.scan_btn.setEnabled(not missing_scan and not self._scanning)
        self.safe_stop_btn.setEnabled(self._scanning)
        if missing_scan:
            self.gate_hint.setText("To scan: " + ", ".join(missing_scan) + ".")
            self.gate_hint.setVisible(True)
        else:
            self.gate_hint.setText("")
            self.gate_hint.setVisible(False)

    # ── state helpers ─────────────────────────────────────────────────

    def set_scanning(self, active: bool) -> None:
        self._scanning = active
        if active:
            self._stopping = False
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
        self._apply_gating()
