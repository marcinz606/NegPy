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
from typing import TYPE_CHECKING

import numpy as np
import qtawesome as qta
from PyQt6.QtCore import QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.infrastructure.roll import coolscanpy_roll
from negpy.infrastructure.roll.repair import RepairMode
from negpy.infrastructure.roll.settings import RollScanSettings
from negpy.services.roll.exact_color import PositiveColorMode
from negpy.services.roll.service import RollFrameOutput

if TYPE_CHECKING:
    import coolscanpy

    from negpy.desktop.workers.roll_worker import RollBatchScanRequest

_SLOT_ROLE = Qt.ItemDataRole.UserRole
_WARN_COLOR = "#C8922E"  # matches ScanlightSidebar's advisory tone
_THUMBNAIL_SIZE = QSize(220, 150)


def _thumbnail_rgb8(
    image: "np.ndarray",
    *,
    positive: bool = False,
) -> np.ndarray:
    """Tone a scanner-linear roll thumbnail for display without mutating it.

    The 97-dpi index is scanner-linear negative transmission, not a display
    image.  A peak-normalize followed by ``255 - value`` makes ordinary C-41
    frames look pale cyan and badly overexposed, especially when one hot
    pixel or rail sets the peak.  Raw display therefore ignores only the
    extreme highlight tail.  Positive display converts transmission to
    optical density, applies robust per-channel endpoints, then a display
    gamma.  This is intentionally an auto-toned review preview; the saved
    Nikon-exact positive still comes from the acquisition-bound builder/CMS.
    """
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("roll thumbnail must be a grayscale or RGB image")
    rgb = np.asarray(arr[..., :3], dtype=np.float64)
    if not rgb.size:
        return np.empty((*rgb.shape[:2], 3), dtype=np.uint8)
    finite = np.where(np.isfinite(rgb), rgb, 0.0)
    finite = np.maximum(finite, 0.0)

    if not positive:
        white = max(float(np.percentile(finite, 99.5)), 1.0)
        display = np.clip(finite / white, 0.0, 1.0)
    else:
        channel_white = np.maximum(
            np.percentile(finite, 99.5, axis=(0, 1)),
            1.0,
        )
        transmission = np.clip(
            finite / channel_white,
            1.0 / 65535.0,
            1.0,
        )
        density = -np.log10(transmission)
        black = np.percentile(density, 1.0, axis=(0, 1))
        white = np.percentile(density, 99.0, axis=(0, 1))
        span = np.maximum(white - black, np.finfo(np.float64).eps)
        display = np.clip((density - black) / span, 0.0, 1.0)
        display = np.sqrt(display)

    return np.ascontiguousarray(np.rint(display * 255.0).astype(np.uint8))


def _thumbnail_pixmap(image: "np.ndarray", *, positive: bool = False) -> QPixmap:
    """A `coolscanpy.Thumbnail.image` array as a displayable QPixmap."""
    arr = _thumbnail_rgb8(image, positive=positive)
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()  # copy: detach from arr's buffer
    return QPixmap.fromImage(qimg)


class RollPreviewWorkspace(QWidget):
    """Center-stage contact sheet for the whole-roll workflow."""

    back_requested = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        title = QLabel("Roll preview")
        title.setStyleSheet(f"color: {THEME.text_primary}; font-size: 20px; font-weight: 600;")
        subtitle = QLabel("Review, approve, and select frames here. Display mode does not change captured or saved files.")
        subtitle.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        subtitle.setWordWrap(True)
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, 1)

        mode_label = QLabel("Show as")
        mode_label.setStyleSheet(f"color: {THEME.text_secondary};")
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.setObjectName("roll_preview_display_mode")
        self.display_mode_combo.addItem("Positive preview (auto tone)", True)
        self.display_mode_combo.addItem("Negative (raw)", False)
        self.display_mode_combo.setMinimumHeight(40)
        self.display_mode_combo.setMinimumWidth(180)
        self.display_mode_combo.setToolTip(
            "Display only: Positive preview applies optical-density auto tone for exposure and framing review. "
            "Negative shows the scanner-linear thumbnail. Neither changes capture bytes or the saved Nikon-exact color."
        )
        self.back_btn = QPushButton("Back to image editor")
        self.back_btn.setMinimumHeight(40)
        self.back_btn.clicked.connect(lambda _checked=False: self.back_requested.emit())
        header.addWidget(mode_label)
        header.addWidget(self.display_mode_combo)
        header.addWidget(self.back_btn)
        layout.addLayout(header)

        self.contact_sheet = QListWidget()
        self.contact_sheet.setObjectName("roll_preview_contact_sheet")
        self.contact_sheet.setViewMode(QListWidget.ViewMode.IconMode)
        self.contact_sheet.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.contact_sheet.setMovement(QListWidget.Movement.Static)
        self.contact_sheet.setWrapping(True)
        self.contact_sheet.setSpacing(12)
        self.contact_sheet.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.contact_sheet.setIconSize(_THUMBNAIL_SIZE)
        self.contact_sheet.setMinimumHeight(360)
        self.contact_sheet.setToolTip(
            "Click a slot to nudge its spacing or approve it; select several, then use Scan Selected in the controls panel."
        )
        layout.addWidget(self.contact_sheet, 1)


class CoolscanRollSidebar(QWidget):
    """Whole-roll preview, spacing/approval and batch fine-scan panel."""

    workspace_requested = pyqtSignal()

    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self._settings: RollScanSettings = self._load_settings()
        self._devices: list = []
        self._devices_loaded = False
        self._thumbnails: dict[int, "coolscanpy.Thumbnail"] = {}
        self._scanning = False
        self._stopping = False  # safe-stop acknowledged, waiting for the in-flight frame
        self._preview_pending = False
        self._eject_pending = False
        self._eject_latched = False
        self._eject_failed = False
        self._active_scan_request: RollBatchScanRequest | None = None

        self.preview_workspace = RollPreviewWorkspace()
        self.contact_sheet = self.preview_workspace.contact_sheet
        self.preview_display_combo = self.preview_workspace.display_mode_combo
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
            write_unrepaired=self.write_unrepaired_check.isChecked(),
            write_repaired=self.write_repaired_check.isChecked(),
            write_positive=self.write_positive_check.isChecked(),
            repair_mode=self.repair_mode_combo.currentData() or RollScanSettings.defaults().repair_mode,
            positive_mode=self.positive_mode_combo.currentData() or PositiveColorMode.NIKON_EXACT.value,
            hybrid_synthesis_limit_percent=self.hybrid_synthesis_limit_spin.value(),
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
        self.eject_btn = QPushButton(qta.icon("fa5s.eject", color=THEME.text_secondary), " Eject Roll…")
        self.eject_btn.setToolTip(
            "Release the scanner reservation and eject the loaded roll. This invalidates the current preview and frame registration."
        )
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.refresh_btn)
        device_row.addWidget(self.eject_btn)
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
        self.pattern_edit.setToolTip(
            'Jinja2 template. Variables: {{ date }}, {{ seq }} (the slot number).\nExample: {{ date }}_{{ "%03d" % seq }}'
        )
        out_form.addRow("Filename", self.pattern_edit)

        # Three independent tiers, not a single three-way choice -- any combination is
        # valid. Unrepaired defaults on (see RollScanSettings): it is the archival
        # master and the only tier the scanner itself can reproduce.
        tiers_row = QHBoxLayout()
        self.write_unrepaired_check = QCheckBox("Unrepaired")
        self.write_unrepaired_check.setChecked(self._settings.write_unrepaired)
        self.write_unrepaired_check.setToolTip(
            "Tier 1, the archival master: the frame exactly as captured, with no repair applied. "
            "This is the only tier the scanner can reproduce -- turning it off loses data that "
            "cannot be recreated from the other two tiers."
        )
        self.write_repaired_check = QCheckBox("Repaired")
        self.write_repaired_check.setChecked(self._settings.write_repaired)
        self.write_repaired_check.setToolTip(
            "Tier 2: the unrepaired capture with infrared-guided dust/scratch repair applied, "
            "still scanner-linear. It needs the frame-bound scanner prepass and validity evidence; "
            "the Tier 1 TIFFs alone cannot recreate parity repair later."
        )
        self.write_positive_check = QCheckBox("Positive")
        self.write_positive_check.setChecked(self._settings.write_positive)
        self.write_positive_check.setToolTip(
            "Tier 3: the repaired capture converted through the Positive color path selected "
            "below. Nikon exact is the parity default; NegPy approximate is an explicitly "
            "labeled preview choice. Tier 2 must be producible first."
        )
        tiers_row.addWidget(self.write_unrepaired_check)
        tiers_row.addWidget(self.write_repaired_check)
        tiers_row.addWidget(self.write_positive_check)
        out_form.addRow("Write", tiers_row)

        self.positive_mode_combo = QComboBox()
        self.positive_mode_combo.addItem("Nikon C-41 exact (parity)", PositiveColorMode.NIKON_EXACT.value)
        self.positive_mode_combo.addItem("NegPy approximate (preview)", PositiveColorMode.NEGPY_APPROXIMATE.value)
        self.positive_mode_combo.setToolTip(
            "Color path for Tier 3. Nikon exact is the fail-closed parity path and requires "
            "frame-bound builder evidence. NegPy approximate is a selectable preview path "
            "and is never labeled Nikon-exact."
        )
        idx = self.positive_mode_combo.findData(self._settings.positive_mode)
        if idx >= 0:
            self.positive_mode_combo.setCurrentIndex(idx)
        out_form.addRow("Positive color", self.positive_mode_combo)

        self.repair_mode_combo = QComboBox()
        self.repair_mode_combo.addItem("Exact (Nikon parity)", RepairMode.EXACT.value)
        self.repair_mode_combo.addItem("Hybrid (generative severe-defect fill)", RepairMode.HYBRID.value)
        self.repair_mode_combo.setToolTip(
            "Governs Tier 2, and Tier 3 through it. Hybrid uses a generative inpainting model for "
            "severe zero-signal regions and records every synthesized pixel in a disclosure mask. "
            "Those fills are not recovered film data and are not bit-deterministic Nikon output."
        )
        idx = self.repair_mode_combo.findData(self._settings.repair_mode)
        if idx >= 0:
            self.repair_mode_combo.setCurrentIndex(idx)
        out_form.addRow("Repair mode", self.repair_mode_combo)

        self.hybrid_synthesis_limit_spin = QDoubleSpinBox()
        self.hybrid_synthesis_limit_spin.setRange(0.0, 100.0)
        self.hybrid_synthesis_limit_spin.setDecimals(1)
        self.hybrid_synthesis_limit_spin.setSuffix("%")
        self.hybrid_synthesis_limit_spin.setValue(self._settings.hybrid_synthesis_limit_percent)
        self.hybrid_synthesis_limit_spin.setToolTip(
            "Maximum portion of a frame Hybrid may synthesize. This is a ceiling, not a target; "
            "the pinned runtime's own ceiling remains in force if it is lower. Set 0% to forbid synthesis."
        )
        out_form.addRow("Hybrid ceiling", self.hybrid_synthesis_limit_spin)

        layout.addLayout(out_form)

        self.tier_hint = QLabel("")
        self.tier_hint.setStyleSheet(f"color: {_WARN_COLOR}; font-size: {THEME.font_size_small}px;")
        self.tier_hint.setWordWrap(True)
        layout.addWidget(self.tier_hint)

        self.hybrid_guidance = QLabel("")
        self.hybrid_guidance.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_small}px;")
        self.hybrid_guidance.setWordWrap(True)
        layout.addWidget(self.hybrid_guidance)

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

        # The images live in the main center workspace; the sidebar contains
        # only the actions and details needed to operate on that selection.
        layout.addWidget(section_subheader("FRAME REVIEW"))
        self.open_preview_workspace_btn = QPushButton(qta.icon("fa5s.th", color=THEME.text_secondary), " Open Roll Preview")
        self.open_preview_workspace_btn.setMinimumHeight(40)
        self.open_preview_workspace_btn.setToolTip("Show the roll thumbnails in the main workspace")
        layout.addWidget(self.open_preview_workspace_btn)
        selection_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All Frames")
        self.select_all_btn.setToolTip("Select every frame found by the current roll preview")
        self.clear_selection_btn = QPushButton("Clear Selection")
        selection_row.addWidget(self.select_all_btn)
        selection_row.addWidget(self.clear_selection_btn)
        layout.addLayout(selection_row)

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
        self.safe_stop_btn.setToolTip(
            "Finish the frame in flight, then stop before the next one -- the transport can't be aborted mid-pull."
        )
        self.safe_stop_btn.setEnabled(False)
        layout.addWidget(self.safe_stop_btn)

        self._show_slot_detail(None)
        self._apply_gating()
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.eject_btn.clicked.connect(self._on_eject_clicked)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        self.folder_browse.clicked.connect(self._on_browse_folder)
        for w in (self.folder_edit, self.pattern_edit):
            w.editingFinished.connect(self._update_settings_from_ui)
        for cb in (self.write_unrepaired_check, self.write_repaired_check, self.write_positive_check):
            cb.toggled.connect(self._update_settings_from_ui)
        self.positive_mode_combo.currentIndexChanged.connect(self._update_settings_from_ui)
        self.repair_mode_combo.currentIndexChanged.connect(self._update_settings_from_ui)
        self.hybrid_synthesis_limit_spin.valueChanged.connect(self._update_settings_from_ui)
        self.preview_btn.clicked.connect(self._on_preview_clicked)
        self.open_preview_workspace_btn.clicked.connect(lambda _checked=False: self.workspace_requested.emit())
        self.preview_display_combo.currentIndexChanged.connect(self._on_preview_display_changed)
        self.contact_sheet.itemSelectionChanged.connect(self._on_selection_changed)
        self.select_all_btn.clicked.connect(self.contact_sheet.selectAll)
        self.clear_selection_btn.clicked.connect(self.contact_sheet.clearSelection)
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
        self.controller.roll_ejected.connect(self._on_ejected)
        self.controller.roll_eject_error.connect(self._on_eject_error)
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

    def _on_eject_clicked(self) -> None:
        device_id = self._current_device_id()
        if not device_id or self._scanning or self._preview_pending or self._eject_pending or self._eject_latched or self._eject_failed:
            return
        answer = QMessageBox.question(
            self,
            "Eject roll?",
            "Eject the loaded roll?\n\n"
            "The current preview and frame registration will be discarded. "
            "If eject reports an uncertain outcome, do not press it again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._eject_pending = True
        self._eject_latched = True
        self._clear_contact_sheet()
        self._set_status("Ejecting roll…")
        self._apply_gating()
        self.controller.eject_roll(device_id)

    @pyqtSlot(list)
    def _on_devices_ready(self, devices: "list[coolscanpy.DeviceInfo]") -> None:
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
        if not device_id or self._scanning or self._preview_pending or self._eject_pending or self._eject_failed:
            return
        from negpy.desktop.workers.roll_worker import RollPreviewRequest

        self._preview_pending = True
        self.workspace_requested.emit()
        self._clear_contact_sheet()
        self._set_status("Reading roll transport…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._apply_gating()
        self.controller.start_coolscan_roll_preview(RollPreviewRequest(device_id=device_id))

    @pyqtSlot(list)
    def _on_preview_ready(self, thumbnails: "list[coolscanpy.Thumbnail]") -> None:
        self.progress_bar.setVisible(False)
        if self._eject_pending or self._eject_failed or (self._eject_latched and not self._preview_pending):
            self._preview_pending = False
            self._clear_contact_sheet()
            self._apply_gating()
            return
        self._preview_pending = False
        self._eject_latched = False
        self._thumbnails = {t.slot: t for t in thumbnails}
        self.contact_sheet.clear()
        for t in sorted(self._thumbnails):
            self._add_slot_item(self._thumbnails[t])
        self._show_slot_detail(None)
        self.workspace_requested.emit()
        self._apply_gating()

    def _add_slot_item(self, thumb: "coolscanpy.Thumbnail") -> None:
        pixmap = _thumbnail_pixmap(thumb.image, positive=self._preview_is_positive()).scaled(
            _THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        item = QListWidgetItem(QIcon(pixmap), f"Slot {thumb.slot}" + (" ⚠" if thumb.needs_approval else ""))
        item.setData(_SLOT_ROLE, thumb.slot)
        if thumb.needs_approval:
            item.setForeground(QColor(_WARN_COLOR))
        self.contact_sheet.addItem(item)

    def _preview_is_positive(self) -> bool:
        return bool(self.preview_display_combo.currentData())

    def _on_preview_display_changed(self) -> None:
        for i in range(self.contact_sheet.count()):
            item = self.contact_sheet.item(i)
            if item is None:
                continue
            thumb = self._thumbnails.get(item.data(_SLOT_ROLE))
            if thumb is None:
                continue
            pixmap = _thumbnail_pixmap(thumb.image, positive=self._preview_is_positive()).scaled(
                _THUMBNAIL_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item.setIcon(QIcon(pixmap))

    def _on_selection_changed(self) -> None:
        current = self.contact_sheet.currentItem()
        slot = current.data(_SLOT_ROLE) if current is not None else None
        self._show_slot_detail(self._thumbnails.get(slot) if slot is not None else None)
        self._apply_gating()

    def _show_slot_detail(self, thumb: "coolscanpy.Thumbnail | None") -> None:
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
            if item is not None and item.data(_SLOT_ROLE) == slot:
                item.setText(f"Slot {slot}")
                item.setData(Qt.ItemDataRole.ForegroundRole, None)  # back to the theme's default text color
                break
        current = self.contact_sheet.currentItem()
        if current is not None and current.data(_SLOT_ROLE) == slot:
            self.approve_btn.setVisible(False)
        self._set_status(f"Slot {slot} approved.")
        self._apply_gating()

    # ── scan ──────────────────────────────────────────────────────────

    def _on_scan_clicked(self) -> None:
        device_id = self._current_device_id()
        slots = self._selected_slots()
        output_folder = self.folder_edit.text().strip()
        if (
            not device_id
            or not slots
            or not output_folder
            or not self._any_tier_selected()
            or self._scanning
            or self._preview_pending
            or self._eject_pending
            or self._eject_latched
            or self._eject_failed
        ):
            return
        from negpy.desktop.workers.roll_worker import RollBatchScanRequest

        self._update_settings_from_ui()
        self._save_settings()
        req = RollBatchScanRequest(
            device_id=device_id,
            slots=tuple(slots),
            output_folder=output_folder,
            filename_pattern=self.pattern_edit.text().strip() or RollScanSettings.defaults().filename_pattern,
            write_unrepaired=self.write_unrepaired_check.isChecked(),
            write_repaired=self.write_repaired_check.isChecked(),
            write_positive=self.write_positive_check.isChecked(),
            repair_mode=self.repair_mode_combo.currentData() or RollScanSettings.defaults().repair_mode,
            positive_mode=self.positive_mode_combo.currentData() or PositiveColorMode.NIKON_EXACT.value,
            hybrid_synthesis_limit_percent=self.hybrid_synthesis_limit_spin.value(),
        )
        self._active_scan_request = req
        self.set_scanning(True)
        self.controller.start_roll_scan(req)

    def _on_safe_stop_clicked(self) -> None:
        self._stopping = True
        self.safe_stop_btn.setEnabled(False)
        self._set_status("Stopping the current frame safely…")
        self.controller.roll_safe_stop()

    @pyqtSlot(float, str)
    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(int(fraction * 100))
        if not self._stopping:
            self._set_status(message)

    @pyqtSlot(object)
    def _on_frame_written(self, output: RollFrameOutput) -> None:
        if not self._stopping:
            self._set_status(f"Wrote slot {output.slot}.")

    @pyqtSlot(list)
    def _on_finished(self, outputs: list[RollFrameOutput]) -> None:
        issues = self._completion_issues(outputs)
        self._active_scan_request = None
        self.set_scanning(False)
        if issues:
            self._set_status("Completed with issues — " + "; ".join(issues))
        else:
            self._set_status(f"Scanned {len(outputs)} frame(s).")

    @pyqtSlot()
    def _on_cancelled(self) -> None:
        self._active_scan_request = None
        self.set_scanning(False)
        self._set_status("Stopped.")

    @pyqtSlot(bool)
    def _on_ejected(self, triggered: bool) -> None:
        self._eject_pending = False
        self._eject_latched = True
        self._clear_contact_sheet()
        if triggered:
            self._set_status("Eject started. Wait until the roll is fully out; preview registration cleared.")
        else:
            self._eject_failed = True
            self._set_status("This device reported no eject action. Check the film physically; do not retry in this session.")
        self._apply_gating()

    @pyqtSlot(str)
    def _on_eject_error(self, message: str) -> None:
        self._eject_pending = False
        self._eject_latched = True
        self._eject_failed = True
        self._clear_contact_sheet()
        self._set_status(f"Eject outcome uncertain: {message}. Check the film physically; do not retry in this session.")
        self._apply_gating()

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self._active_scan_request = None
        self._preview_pending = False
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

    def _completion_issues(
        self,
        outputs: list[RollFrameOutput],
    ) -> list[str]:
        """Describe requested outputs that the fail-closed service withheld."""

        request = self._active_scan_request
        if request is None:
            return []
        issues: list[str] = []
        completed_slots = {output.slot for output in outputs}
        missing_slots = [slot for slot in request.slots if slot not in completed_slots]
        if missing_slots:
            issues.append("slot(s) " + ", ".join(str(slot) for slot in missing_slots) + " did not complete")
        for output in outputs:
            prefix = f"slot {output.slot}: "
            if request.write_unrepaired and not output.rgb_path:
                issues.append(prefix + "requested unrepaired output unavailable")
            if request.write_repaired and not output.repaired_rgb_path:
                issues.append(prefix + "requested repaired output unavailable")
            repair_was_needed = request.write_repaired or request.write_positive
            if (
                repair_was_needed
                and request.repair_mode == RepairMode.HYBRID.value
                and not (output.native_synthesis_mask_path and output.hybrid_receipt_path)
            ):
                issues.append(prefix + "Hybrid repair degraded or unavailable")
            if request.write_positive and not output.positive_path:
                if request.positive_mode == PositiveColorMode.NIKON_EXACT.value:
                    issues.append(prefix + "Nikon exact positive unavailable")
                else:
                    issues.append(prefix + "requested positive output unavailable")
        return issues

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
        if self._eject_failed:
            m.append("restart after physically checking the uncertain eject")
        return m

    def _any_tier_selected(self) -> bool:
        return self.write_unrepaired_check.isChecked() or self.write_repaired_check.isChecked() or self.write_positive_check.isChecked()

    def _missing_for_scan(self) -> list[str]:
        m = list(self._missing_for_preview())
        if not self._selected_slots():
            m.append("select at least one slot")
        if not self.folder_edit.text().strip():
            m.append("choose an output folder")
        if not self._any_tier_selected():
            m.append("select at least one output tier")
        return m

    def _update_tier_hint(self) -> None:
        """Makes the archival tradeoff legible instead of a silent default: Tier 1 is the
        only tier the scanner itself can reproduce, so turning it off is a real choice
        with a real cost, not just another checkbox."""
        if not self.write_unrepaired_check.isChecked():
            self.tier_hint.setText(
                "Unrepaired (Tier 1) is off. If this frame ever needs to be re-scanned, only "
                "the scanner can reproduce it -- Repaired and Positive are both derived from it."
            )
            self.tier_hint.setVisible(True)
        else:
            self.tier_hint.setText("")
            self.tier_hint.setVisible(False)

    def _update_hybrid_guidance(self) -> None:
        """Explain the choice without silently opting a frame into AI fill."""

        if self.repair_mode_combo.currentData() == RepairMode.HYBRID.value:
            ceiling = self.hybrid_synthesis_limit_spin.value()
            self.hybrid_guidance.setText(
                f"Hybrid is recommended only when the scanner finds severe zero-signal infrared areas. "
                f"It keeps Exact repair elsewhere, discloses every generated pixel, and is capped at {ceiling:.1f}%."
            )
        else:
            self.hybrid_guidance.setText(
                "Exact is recommended for normal dust and scratches with usable infrared, and for Nikon-parity-only output. "
                "Use Hybrid only when infrared has a severe zero-signal area that Exact cannot repair."
            )

    def _apply_gating(self) -> None:
        missing_preview = self._missing_for_preview()
        missing_scan = self._missing_for_scan()
        registration_locked = self._preview_pending or self._eject_pending or self._eject_latched or self._eject_failed
        self.preview_btn.setEnabled(not missing_preview and not self._scanning and not self._preview_pending and not self._eject_pending)
        self.scan_btn.setEnabled(not missing_scan and not self._scanning and not registration_locked)
        self.safe_stop_btn.setEnabled(self._scanning)
        self.device_combo.setEnabled(not self._scanning and not self._preview_pending and not self._eject_pending)
        self.refresh_btn.setEnabled(not self._scanning and not self._preview_pending and not self._eject_pending)
        self.contact_sheet.setEnabled(not self._scanning and not registration_locked)
        self.open_preview_workspace_btn.setEnabled(not self._eject_failed)
        self.select_all_btn.setEnabled(bool(self._thumbnails) and not self._scanning and not registration_locked)
        self.clear_selection_btn.setEnabled(bool(self._selected_slots()) and not self._scanning and not registration_locked)
        self.eject_btn.setEnabled(
            bool(self._current_device_id())
            and not self._scanning
            and not self._preview_pending
            and not self._eject_pending
            and not self._eject_latched
            and not self._eject_failed
        )
        self._update_tier_hint()
        self._update_hybrid_guidance()
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
