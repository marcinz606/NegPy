import piexif
import qtawesome as qta
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QWidget,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME

FORMAT_OPTIONS = ["35mm", "120", "4×5", "8×10", "Other"]
PUSH_PULL_OPTIONS = ["Push +3", "Push +2", "Push +1", "Normal", "Pull -1", "Pull -2", "Pull -3"]
PUSH_PULL_VALUES = [3, 2, 1, 0, -1, -2, -3]


class MetadataSidebar(BaseSidebar):
    """
    Panel for setting custom analog photography metadata on exported files.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)
        conf = self.state.config.metadata

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self._persist_all_metadata_settings)
        self._dirty = False

        # ── CUSTOM METADATA ──────────────────────────────────────────────
        self.layout.addWidget(section_subheader("CUSTOM METADATA"))

        self.layout.addWidget(QLabel("Film"))
        self.film_edit = QLineEdit()
        self.film_edit.setPlaceholderText("e.g. Portra 400")
        self.film_edit.setText(conf.film)
        self.film_edit.setToolTip("Film stock name")
        self.layout.addWidget(self.film_edit)

        self.layout.addWidget(QLabel("Format"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMAT_OPTIONS)
        if conf.format in FORMAT_OPTIONS:
            self.format_combo.setCurrentText(conf.format)
        self.format_combo.setToolTip("Film format")
        self.layout.addWidget(self.format_combo)

        self.format_other_edit = QLineEdit()
        self.format_other_edit.setPlaceholderText("e.g. 6×7")
        self.format_other_edit.setText(conf.format_other)
        self.format_other_edit.setToolTip("Custom format")
        self.format_other_edit.setVisible(conf.format == "Other")
        self.layout.addWidget(self.format_other_edit)

        self.layout.addWidget(QLabel("Developer"))
        self.developer_edit = QLineEdit()
        self.developer_edit.setPlaceholderText("e.g. D-76 1+1")
        self.developer_edit.setText(conf.developer)
        self.developer_edit.setToolTip("Developer and dilution")
        self.layout.addWidget(self.developer_edit)

        self.layout.addWidget(QLabel("Push / Pull"))
        self.push_pull_combo = QComboBox()
        self.push_pull_combo.addItems(PUSH_PULL_OPTIONS)
        idx = PUSH_PULL_VALUES.index(conf.push_pull) if conf.push_pull in PUSH_PULL_VALUES else 3
        self.push_pull_combo.setCurrentIndex(idx)
        self.push_pull_combo.setToolTip("Push or pull processing")
        self.layout.addWidget(self.push_pull_combo)

        self.layout.addWidget(QLabel("Scanning"))
        self.scanning_edit = QLineEdit()
        self.scanning_edit.setPlaceholderText("e.g. Noritsu HS-1800")
        self.scanning_edit.setText(conf.scanning)
        self.scanning_edit.setToolTip("Scanner or scanning method")
        self.layout.addWidget(self.scanning_edit)

        self.sync_check = QCheckBox("Sync custom metadata to all files in batch export")
        self.sync_check.setChecked(conf.sync_to_batch)
        self.sync_check.setToolTip("When checked, all exported files get these custom metadata values")
        self.layout.addWidget(self.sync_check)

        # ── INHERITED FROM SOURCE ────────────────────────────────────────
        self.layout.addWidget(section_subheader("INHERITED FROM SOURCE"))

        hint = QLabel("Locked by default — click 🔓 to edit")
        hint.setStyleSheet(f"font-size: {THEME.font_size_xs}px; color: {THEME.text_muted};")
        self.layout.addWidget(hint)

        self._exif_locked: dict[str, bool] = {
            "camera": True,
            "lens": True,
            "exposure": True,
        }

        # Camera row
        self.camera_label = QLabel("Camera")
        self.layout.addWidget(self.camera_label)
        self.camera_edit = self._make_exif_field("camera")

        # Lens row
        self.lens_label = QLabel("Lens")
        self.layout.addWidget(self.lens_label)
        self.lens_edit = self._make_exif_field("lens")

        # Exposure row
        self.exposure_label = QLabel("Exposure")
        self.layout.addWidget(self.exposure_label)
        self.exposure_edit = self._make_exif_field("exposure")

        # Hidden until EXIF is available
        self._set_inherited_visible(False)

        self.layout.addStretch()

    def _make_exif_field(self, key: str) -> QLineEdit:
        """Create a QHBoxLayout row: QLineEdit + lock/unlock QToolButton."""
        row = QHBoxLayout()
        row.setSpacing(4)

        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText("—")
        self._apply_lock_style(edit, locked=True)

        lock_btn = QToolButton()
        lock_btn.setCheckable(True)
        lock_btn.setToolTip("Unlock to edit")
        self._update_lock_icon(lock_btn, locked=True)
        lock_btn.toggled.connect(lambda checked, k=key, e=edit, b=lock_btn: self._toggle_exif_lock(k, e, b, checked))

        row.addWidget(edit)
        row.addWidget(lock_btn)
        self.layout.addLayout(row)

        # Store lock button reference for visibility/reset
        setattr(self, f"_{key}_lock_btn", lock_btn)

        return edit

    def _apply_lock_style(self, edit: QLineEdit, locked: bool) -> None:
        """Style a QLineEdit based on lock state."""
        if locked:
            edit.setStyleSheet(f"color: {THEME.text_secondary};")
            edit.setReadOnly(True)
        else:
            edit.setStyleSheet(f"color: {THEME.text_primary};")
            edit.setReadOnly(False)

    def _update_lock_icon(self, btn: QToolButton, locked: bool) -> None:
        """Set the lock/unlock icon on a QToolButton."""
        icon_name = "fa5s.lock" if locked else "fa5s.lock-open"
        color = THEME.text_muted if locked else THEME.text_primary
        btn.setIcon(qta.icon(icon_name, color=color))

    def _toggle_exif_lock(self, key: str, edit: QLineEdit, btn: QToolButton, checked: bool) -> None:
        """Handle lock/unlock toggle for an EXIF field."""
        locked = not checked
        self._exif_locked[key] = locked
        self._apply_lock_style(edit, locked=locked)
        self._update_lock_icon(btn, locked=locked)
        if not locked:
            edit.setFocus()

    def _connect_signals(self) -> None:
        # Custom metadata changes → debounced persist
        self.film_edit.textChanged.connect(self._mark_dirty)
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.format_other_edit.textChanged.connect(self._mark_dirty)
        self.developer_edit.textChanged.connect(self._mark_dirty)
        self.push_pull_combo.currentIndexChanged.connect(self._mark_dirty)
        self.scanning_edit.textChanged.connect(self._mark_dirty)
        self.sync_check.toggled.connect(self._mark_dirty)

        # EXIF override edits → debounced persist
        for edit in self._exif_edits().values():
            edit.textChanged.connect(self._mark_dirty)

        # External sync triggers
        self.controller.session.file_selected.connect(self._on_file_selected)

    def _exif_edits(self) -> dict:
        return {
            "camera": self.camera_edit,
            "lens": self.lens_edit,
            "exposure": self.exposure_edit,
        }

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.update_timer.start()

    def _on_format_changed(self, text: str) -> None:
        self.format_other_edit.setVisible(text == "Other")
        self._mark_dirty()

    def _persist_all_metadata_settings(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        fmt = self.format_combo.currentText()
        pp_idx = self.push_pull_combo.currentIndex()

        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            film=self.film_edit.text().strip(),
            format=fmt,
            format_other=self.format_other_edit.text().strip() if fmt == "Other" else "",
            developer=self.developer_edit.text().strip(),
            push_pull=PUSH_PULL_VALUES[pp_idx] if 0 <= pp_idx < len(PUSH_PULL_VALUES) else 0,
            scanning=self.scanning_edit.text().strip(),
            sync_to_batch=self.sync_check.isChecked(),
            camera_override=self.camera_edit.text().strip(),
            lens_override=self.lens_edit.text().strip(),
            exposure_override=self.exposure_edit.text().strip(),
        )

    def sync_ui(self) -> None:
        """Sync custom metadata widgets and EXIF overrides from config (undo/redo, file switch)."""
        if self._dirty:
            return

        conf = self.state.config.metadata

        self.block_signals(True)
        try:
            self.film_edit.setText(conf.film)
            if conf.format in FORMAT_OPTIONS:
                self.format_combo.setCurrentText(conf.format)
            else:
                self.format_combo.setCurrentText("Other")
                self.format_other_edit.setText(conf.format_other)
            self.format_other_edit.setVisible(self.format_combo.currentText() == "Other")
            self.developer_edit.setText(conf.developer)
            idx = PUSH_PULL_VALUES.index(conf.push_pull) if conf.push_pull in PUSH_PULL_VALUES else 3
            self.push_pull_combo.setCurrentIndex(idx)
            self.scanning_edit.setText(conf.scanning)
            self.sync_check.setChecked(conf.sync_to_batch)

            # Refresh EXIF override fields from config
            self._set_exif_text_quiet("camera", conf.camera_override)
            self._set_exif_text_quiet("lens", conf.lens_override)
            self._set_exif_text_quiet("exposure", conf.exposure_override)
        finally:
            self.block_signals(False)

    def _set_exif_text_quiet(self, key: str, text: str) -> None:
        """Set EXIF edit text without triggering signals or dirty flag."""
        edit = self._exif_edits().get(key)
        if edit is None:
            return
        edit.blockSignals(True)
        try:
            edit.setText(text)
        finally:
            edit.blockSignals(False)

    def _on_file_selected(self, _path: str) -> None:
        """Called when the active file changes — sync widgets + EXIF display."""
        self._dirty = False
        self._reset_exif_locks()
        self.sync_ui()
        self._update_exif_display()

    def _reset_exif_locks(self) -> None:
        """Reset all EXIF fields to locked state."""
        for key, edit in self._exif_edits().items():
            self._exif_locked[key] = True
            self._apply_lock_style(edit, locked=True)
            btn = getattr(self, f"_{key}_lock_btn", None)
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
                self._update_lock_icon(btn, locked=True)

    def _update_exif_display(self) -> None:
        """Update the inherited EXIF section from source EXIF or config overrides."""
        conf = self.state.config.metadata
        source_exif = self.state.source_exif
        current_hash = self.state.current_file_hash

        if current_hash and current_hash in source_exif:
            exif = source_exif[current_hash]
            self._update_inherited_display(exif, conf)
        else:
            # No EXIF available — clear fields
            self._set_inherited_visible(False)

    def _update_inherited_display(self, exif: dict, conf) -> None:
        """Parse piexif-format EXIF dict and set QLineEdit text, preferring overrides."""
        zeroth = exif.get("0th", {}) if isinstance(exif, dict) else {}
        exif_tags = exif.get("Exif", {}) if isinstance(exif, dict) else {}

        make = _safe_str(zeroth.get(piexif.ImageIFD.Make, ""))
        model = _safe_str(zeroth.get(piexif.ImageIFD.Model, ""))
        lens = _safe_str(exif_tags.get(piexif.ExifIFD.LensModel, ""))

        # Camera: override > source EXIF
        if conf.camera_override:
            camera = conf.camera_override
        else:
            camera = f"{make} {model}".strip()

        # Lens: override > source EXIF
        if conf.lens_override:
            lens_text = conf.lens_override
        else:
            lens_text = lens

        # Exposure: override > source EXIF
        if conf.exposure_override:
            exp_text = conf.exposure_override
        else:
            exp_parts = _format_exposure(exif_tags)
            exp_text = exp_parts if exp_parts else ""

        self._set_exif_text_quiet("camera", camera)
        self._set_exif_text_quiet("lens", lens_text)
        self._set_exif_text_quiet("exposure", exp_text)

        has_any = bool(camera or lens_text or exp_text)
        self._set_inherited_visible(has_any)

    def _set_inherited_visible(self, visible: bool) -> None:
        for key in ("camera", "lens", "exposure"):
            label = getattr(self, f"{key}_label", None)
            edit = self._exif_edits().get(key)
            btn = getattr(self, f"_{key}_lock_btn", None)
            if label:
                label.setVisible(visible)
            if edit:
                edit.setVisible(visible)
            if btn:
                btn.setVisible(visible)

    def block_signals(self, blocked: bool) -> None:
        for w in self.findChildren(QWidget):
            w.blockSignals(blocked)


def _safe_str(value) -> str:
    """Convert EXIF value to string, handling bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00")
    if value is None:
        return ""
    return str(value).strip("\x00")


def _format_exposure(exif_tags: dict) -> str:
    """Format exposure time, f-number, and ISO into a human-readable string."""
    exposure_time = exif_tags.get(piexif.ExifIFD.ExposureTime)
    f_number = exif_tags.get(piexif.ExifIFD.FNumber)
    iso = exif_tags.get(piexif.ExifIFD.ISOSpeedRatings)

    parts = []

    if exposure_time is not None:
        if isinstance(exposure_time, tuple) and len(exposure_time) == 2:
            num, den = exposure_time
            if num == 0:
                pass
            elif num == 1:
                parts.append(f"1/{den}s")
            else:
                parts.append(f"{num}/{den}s")
        elif isinstance(exposure_time, (int, float)):
            parts.append(f"{exposure_time}s")

    if f_number is not None:
        if isinstance(f_number, tuple) and len(f_number) == 2:
            num, den = f_number
            val = num / den if den else 0
            parts.append(f"f/{val:.1f}")
        elif isinstance(f_number, (int, float)):
            parts.append(f"f/{f_number:.1f}")

    if iso is not None:
        if isinstance(iso, tuple) and len(iso) == 2:
            iso = iso[0] // iso[1] if iso[1] else 0
        parts.append(f"ISO {iso}")

    return "  ".join(parts)
