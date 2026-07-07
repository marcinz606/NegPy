import qtawesome as qta
from dataclasses import asdict, replace
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import field_label, section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.gear_library_dialog import GearLibraryDialog
from negpy.features.metadata.gear_logic import metadata_from_gear
from negpy.features.metadata.gear_models import GearLibrary
from negpy.features.metadata.payload import build_metadata_payload
from negpy.services.assets.gear import GearProfiles

FORMAT_OPTIONS = ["35mm", "120", "4×5", "8×10", "110", "Other"]
PUSH_PULL_OPTIONS = ["Push +3", "Push +2", "Push +1", "Normal", "Pull -1", "Pull -2", "Pull -3"]
PUSH_PULL_VALUES = [3, 2, 1, 0, -1, -2, -3]

_NONE = "— None —"


class MetadataSidebar(BaseSidebar):
    """Panel for analog gear metadata written to exported files."""

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)
        conf = self.state.config.metadata
        self._gear_library: GearLibrary = GearProfiles.load_library()

        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(500)
        self.update_timer.timeout.connect(self._persist_all_metadata_settings)

        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self._update_preview)

        self._dirty = False
        self._exif_locked = {"exposure": True}

        # ── ORIGINAL ANALOG GEAR ─────────────────────────────────────────
        self.layout.addWidget(section_subheader("ORIGINAL ANALOG GEAR"))

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        self.layout.addWidget(field_label("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Reusable camera + lens + film combination")
        preset_row.addWidget(self.preset_combo, 1)
        self.preset_clear_btn = QPushButton("Clear")
        self.preset_clear_btn.setToolTip("Clear gear preset selection")
        preset_row.addWidget(self.preset_clear_btn)
        self.layout.addLayout(preset_row)

        self.layout.addWidget(field_label("Camera"))
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip("Original film camera body")
        self.layout.addWidget(self.camera_combo)

        self.layout.addWidget(field_label("Lens"))
        self.lens_combo = QComboBox()
        self.lens_combo.setToolTip("Original lens used on the film camera")
        self.layout.addWidget(self.lens_combo)

        self.layout.addWidget(field_label("Film stock"))
        self.film_stock_combo = QComboBox()
        self.film_stock_combo.setToolTip("Film stock used for the original capture")
        self.layout.addWidget(self.film_stock_combo)

        self.manage_btn = QPushButton(" Manage…")
        self.manage_btn.setIcon(qta.icon("fa5s.cog", color=THEME.text_primary))
        self.manage_btn.setToolTip("Edit cameras, lenses, film stocks, and gear presets")
        self.layout.addWidget(self.manage_btn)

        # ── PROCESS ──────────────────────────────────────────────────────
        self.layout.addWidget(section_subheader("PROCESS"))

        self.layout.addWidget(field_label("Format"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(FORMAT_OPTIONS)
        if conf.format in FORMAT_OPTIONS:
            self.format_combo.setCurrentText(conf.format)
        self.layout.addWidget(self.format_combo)

        self.format_other_edit = QLineEdit()
        self.format_other_edit.setPlaceholderText("e.g. 6×7")
        self.format_other_edit.setText(conf.format_other)
        self.format_other_edit.setVisible(conf.format == "Other")
        self.layout.addWidget(self.format_other_edit)

        self.layout.addWidget(field_label("Developer"))
        self.developer_edit = QLineEdit()
        self.developer_edit.setPlaceholderText("e.g. D-76 1+1")
        self.developer_edit.setText(conf.developer)
        self.layout.addWidget(self.developer_edit)

        self.layout.addWidget(field_label("Push / Pull"))
        self.push_pull_combo = QComboBox()
        self.push_pull_combo.addItems(PUSH_PULL_OPTIONS)
        idx = PUSH_PULL_VALUES.index(conf.push_pull) if conf.push_pull in PUSH_PULL_VALUES else 3
        self.push_pull_combo.setCurrentIndex(idx)
        self.layout.addWidget(self.push_pull_combo)

        # ── SCANNING ─────────────────────────────────────────────────────
        self.layout.addWidget(section_subheader("SCANNING"))

        self.layout.addWidget(field_label("Scanning"))
        self.scanning_edit = QLineEdit()
        self.scanning_edit.setPlaceholderText("e.g. DSLR copy-stand scan")
        self.scanning_edit.setText(conf.scanning)
        self.layout.addWidget(self.scanning_edit)

        self.sync_check = QCheckBox("Sync custom metadata to all files in batch export")
        self.sync_check.setChecked(conf.sync_to_batch)
        self.layout.addWidget(self.sync_check)

        # ── EXPOSURE ─────────────────────────────────────────────────────
        self.layout.addWidget(section_subheader("EXPOSURE"))

        hint = QLabel("Optional original capture exposure — click 🔓 to edit")
        hint.setStyleSheet(f"font-size: {THEME.font_size_xs}px; color: {THEME.text_muted};")
        self.layout.addWidget(hint)

        self.exposure_label = field_label("Exposure")
        self.layout.addWidget(self.exposure_label)
        self.exposure_edit = self._make_exif_field("exposure")

        # ── METADATA PREVIEW ─────────────────────────────────────────────
        self.preview_content = QWidget()
        preview_layout = QVBoxLayout(self.preview_content)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)

        preview_hint = QLabel("Written to exported files on export.")
        preview_hint.setWordWrap(True)
        preview_hint.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;")
        preview_layout.addWidget(preview_hint)

        self.preview_rows = QVBoxLayout()
        self.preview_rows.setSpacing(2)
        preview_layout.addLayout(self.preview_rows)

        self.preview_empty = QLabel("Select gear or enter process metadata to see a preview.")
        self.preview_empty.setWordWrap(True)
        self.preview_empty.setStyleSheet(f"color: {THEME.text_muted}; font-size: {THEME.font_size_xs}px;")
        preview_layout.addWidget(self.preview_empty)

        self.preview_section = CollapsibleSection("Metadata preview", expanded=True)
        self.preview_section.set_content(self.preview_content)
        self.layout.addWidget(self.preview_section)

        self._refresh_gear_combos()
        self.layout.addStretch()

    def _make_exif_field(self, key: str) -> QLineEdit:
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
        setattr(self, f"_{key}_lock_btn", lock_btn)
        return edit

    def _apply_lock_style(self, edit: QLineEdit, locked: bool) -> None:
        if locked:
            edit.setStyleSheet(f"color: {THEME.text_secondary};")
            edit.setReadOnly(True)
        else:
            edit.setStyleSheet(f"color: {THEME.text_primary};")
            edit.setReadOnly(False)

    def _update_lock_icon(self, btn: QToolButton, locked: bool) -> None:
        icon_name = "fa5s.lock" if locked else "fa5s.lock-open"
        color = THEME.text_muted if locked else THEME.text_primary
        btn.setIcon(qta.icon(icon_name, color=color))

    def _toggle_exif_lock(self, key: str, edit: QLineEdit, btn: QToolButton, checked: bool) -> None:
        locked = not checked
        self._exif_locked[key] = locked
        self._apply_lock_style(edit, locked=locked)
        self._update_lock_icon(btn, locked=locked)
        if not locked:
            edit.setFocus()
        else:
            self._update_exif_display()
        self._mark_dirty()

    def _connect_signals(self) -> None:
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.preset_clear_btn.clicked.connect(self._on_preset_clear)
        self.camera_combo.currentIndexChanged.connect(self._on_gear_changed)
        self.lens_combo.currentIndexChanged.connect(self._on_gear_changed)
        self.film_stock_combo.currentIndexChanged.connect(self._on_gear_changed)
        self.manage_btn.clicked.connect(self._open_gear_library)

        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        self.format_other_edit.textChanged.connect(self._mark_dirty)
        self.developer_edit.textChanged.connect(self._mark_dirty)
        self.push_pull_combo.currentIndexChanged.connect(self._mark_dirty)
        self.scanning_edit.textChanged.connect(self._mark_dirty)
        self.sync_check.toggled.connect(self._mark_dirty)
        self.exposure_edit.textChanged.connect(self._mark_dirty)

        self.controller.session.file_selected.connect(self._on_file_selected)

    def _refresh_gear_combos(self) -> None:
        conf = self.state.config.metadata
        self._gear_library = GearProfiles.load_library()

        def fill(combo: QComboBox, items, selected_id: str) -> None:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(_NONE, "")
            for item in items:
                label = item.resolved_display_name if hasattr(item, "resolved_display_name") else item.display_name
                combo.addItem(label, item.id)
            idx = combo.findData(selected_id or "")
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)

        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem(_NONE, "")
        for p in self._gear_library.gear_presets:
            self.preset_combo.addItem(p.display_name or "Unnamed preset", p.id)
        pidx = self.preset_combo.findData(conf.gear_preset_id or "")
        self.preset_combo.setCurrentIndex(pidx if pidx >= 0 else 0)
        self.preset_combo.blockSignals(False)

        fill(self.camera_combo, self._gear_library.cameras, conf.camera_id)
        fill(self.lens_combo, self._gear_library.lenses, conf.lens_id)
        fill(self.film_stock_combo, self._gear_library.film_stocks, conf.film_stock_id)

    def _on_preset_changed(self, _idx: int) -> None:
        preset_id = self.preset_combo.currentData() or ""
        if not preset_id:
            return
        new_meta = metadata_from_gear(
            self.state.config.metadata,
            self._gear_library,
            gear_preset_id=preset_id,
        )
        self._apply_metadata_config(new_meta)

    def _on_preset_clear(self) -> None:
        cleared = replace(
            self.state.config.metadata,
            gear_preset_id="",
            camera_id="",
            lens_id="",
            film_stock_id="",
            camera_make="",
            camera_model="",
            lens_make="",
            lens_model="",
            focal_length_mm=None,
            max_aperture=None,
            film_iso=None,
            film_manufacturer="",
            film_color_type="",
            film="",
        )
        self._apply_metadata_config(cleared)

    def _on_gear_changed(self, *_args) -> None:
        new_meta = metadata_from_gear(
            self.state.config.metadata,
            self._gear_library,
            gear_preset_id="",
            camera_id=self.camera_combo.currentData() or "",
            lens_id=self.lens_combo.currentData() or "",
            film_stock_id=self.film_stock_combo.currentData() or "",
        )
        new_meta = replace(new_meta, gear_preset_id="")
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.blockSignals(False)
        self._apply_metadata_config(new_meta, refresh_combos=False)

    def _apply_metadata_config(self, new_meta, *, refresh_combos: bool = True) -> None:
        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            **asdict(new_meta),
        )
        if refresh_combos:
            self._refresh_gear_combos()
        self.sync_ui()
        self._schedule_preview()

    def _open_gear_library(self) -> None:
        dlg = GearLibraryDialog(self._gear_library, parent=self)
        dlg.library_changed.connect(self._on_library_changed)
        if dlg.exec():
            self._on_library_changed()

    def _on_library_changed(self) -> None:
        self._gear_library = GearProfiles.load_library()
        self._refresh_gear_combos()
        self._schedule_preview()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.update_timer.start()
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        self.preview_timer.start()

    def _on_format_changed(self, text: str) -> None:
        self.format_other_edit.setVisible(text == "Other")
        self._mark_dirty()

    def _persist_all_metadata_settings(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        fmt = self.format_combo.currentText()
        pp_idx = self.push_pull_combo.currentIndex()

        exposure_override = ""
        if not self._exif_locked.get("exposure", True):
            exposure_override = self.exposure_edit.text().strip()

        self.update_config_section(
            "metadata",
            persist=True,
            render=False,
            readback_metrics=False,
            gear_preset_id=self.preset_combo.currentData() or "",
            camera_id=self.camera_combo.currentData() or "",
            lens_id=self.lens_combo.currentData() or "",
            film_stock_id=self.film_stock_combo.currentData() or "",
            format=fmt,
            format_other=self.format_other_edit.text().strip() if fmt == "Other" else "",
            developer=self.developer_edit.text().strip(),
            push_pull=PUSH_PULL_VALUES[pp_idx] if 0 <= pp_idx < len(PUSH_PULL_VALUES) else 0,
            scanning=self.scanning_edit.text().strip(),
            sync_to_batch=self.sync_check.isChecked(),
            exposure_override=exposure_override,
        )

    def sync_ui(self) -> None:
        if self._dirty:
            return

        conf = self.state.config.metadata

        self.block_signals(True)
        try:
            self._refresh_gear_combos()

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

            if conf.exposure_override:
                self._set_exif_text_quiet("exposure", conf.exposure_override)
            else:
                self._update_exif_display()
        finally:
            self.block_signals(False)

        self._schedule_preview()

    def _set_exif_text_quiet(self, key: str, text: str) -> None:
        edit = getattr(self, f"{key}_edit", None)
        if edit is None:
            return
        edit.blockSignals(True)
        try:
            edit.setText(text)
        finally:
            edit.blockSignals(False)

    def _on_file_selected(self, _path: str) -> None:
        self._dirty = False
        self._reset_exif_locks()
        self.sync_ui()

    def _reset_exif_locks(self) -> None:
        self._exif_locked["exposure"] = True
        self._apply_lock_style(self.exposure_edit, locked=True)
        btn = getattr(self, "_exposure_lock_btn", None)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            self._update_lock_icon(btn, locked=True)

    def _update_exif_display(self) -> None:
        conf = self.state.config.metadata
        if conf.exposure_override:
            self._set_exif_text_quiet("exposure", conf.exposure_override)
        else:
            self._set_exif_text_quiet("exposure", "")

    def _update_preview(self) -> None:
        while self.preview_rows.count():
            item = self.preview_rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        conf = self.state.config.metadata
        source_exif = None
        current_hash = self.state.current_file_hash
        if current_hash and current_hash in self.state.source_exif:
            source_exif = self.state.source_exif[current_hash]

        payload = build_metadata_payload(conf, self._gear_library, source_exif)
        sections = payload.to_preview_sections()

        self.preview_empty.setVisible(not sections)
        mono = f"font-family: Consolas, monospace; font-size: {THEME.font_size_xs}px;"

        for title, rows in sections:
            header = QLabel(title)
            header.setStyleSheet(f"color: {THEME.text_secondary}; font-size: {THEME.font_size_xs}px; font-weight: 600;")
            self.preview_rows.addWidget(header)
            for label, value in rows:
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                lbl = QLabel(label)
                lbl.setStyleSheet(f"color: {THEME.text_muted}; {mono}")
                lbl.setFixedWidth(110)
                val = QLabel(value)
                val.setWordWrap(True)
                val.setStyleSheet(f"color: {THEME.text_primary}; {mono}")
                row_layout.addWidget(lbl)
                row_layout.addWidget(val, 1)
                self.preview_rows.addWidget(row)

    def block_signals(self, blocked: bool) -> None:
        for w in self.findChildren(QWidget):
            w.blockSignals(blocked)
