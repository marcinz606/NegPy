from negpy.kernel.system.i18n import tr
import uuid

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.styles.templates import (
    dialog_pane_qss,
    hint_label,
    icon_button,
    labeled_toggle,
    pin_dialog_default,
    pane_header_qss,
)
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.export_settings_form import ExportSettingsForm
from negpy.domain.models import ColorSpace, ExportFormat, ExportPreset, ExportResolutionMode, preset_display_name
from negpy.features.exposure.models import RenderIntent


def _list_label(preset: ExportPreset) -> str:
    """The list row: a preset that is off says so, since the row has no check mark."""
    name = preset_display_name(preset)
    return name if preset.enabled else tr("{value} (off)", value=name)


class ExportPresetsDialog(QDialog):
    """Modal dialog for managing export presets."""

    presets_changed = pyqtSignal(list)  # emits updated list[ExportPreset]

    def __init__(self, presets: list, parent=None, *, repo=None):
        super().__init__(parent)
        self._presets: list[ExportPreset] = [self._copy_preset(p) for p in presets]
        self._selected_idx: int = -1
        self._updating_form = False

        self.setWindowTitle(tr("Export Presets"))
        self.resize(860, 620)
        self._init_ui()
        if self._presets:
            self._select_row(0)
        remember_dialog_geometry(self, repo, "export_presets")

    # UI setup

    def _init_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Left: preset list + action buttons
        left = QWidget()
        left.setFixedWidth(220)
        left.setObjectName("dialog_pane")
        left.setStyleSheet(dialog_pane_qss())
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(THEME.space_lg, THEME.space_lg, THEME.space_lg, THEME.space_lg)
        left_layout.setSpacing(THEME.space_md)

        list_label = QLabel(tr("PRESETS"))
        list_label.setStyleSheet(pane_header_qss())
        left_layout.addWidget(list_label)

        self.preset_list = QListWidget()
        self.preset_list.setObjectName("preset_list")
        self.preset_list.currentRowChanged.connect(self._on_list_selection_changed)
        left_layout.addWidget(self.preset_list)

        btn_row = QHBoxLayout()
        self.add_btn = icon_button("fa5s.plus", tr("Add print or flat master preset"))
        self.add_btn.clicked.connect(self._show_add_menu)
        self.dup_btn = icon_button("fa5s.copy", tr("Duplicate preset"))
        self.dup_btn.clicked.connect(self._duplicate_preset)
        self.del_btn = icon_button("fa5s.trash-alt", tr("Delete preset"))
        self.del_btn.clicked.connect(self._delete_preset)
        self.up_btn = icon_button("fa5s.arrow-up", tr("Move up"))
        self.up_btn.clicked.connect(self._move_up)
        self.down_btn = icon_button("fa5s.arrow-down", tr("Move down"))
        self.down_btn.clicked.connect(self._move_down)

        for btn in (self.add_btn, self.dup_btn, self.del_btn, self.up_btn, self.down_btn):
            btn_row.addWidget(btn)
        btn_row.addStretch()
        left_layout.addLayout(btn_row)

        root.addWidget(left)

        # Right: edit form in a scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        form_widget = QWidget()
        form_widget.setStyleSheet(f"background: {THEME.bg_dark};")
        self._form_layout = QVBoxLayout(form_widget)
        self._form_layout.setContentsMargins(20, 20, 20, 20)
        self._form_layout.setSpacing(14)

        self._build_form()
        self._form_layout.addStretch()

        scroll.setWidget(form_widget)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(scroll, 1)
        footer = QHBoxLayout()
        footer.setContentsMargins(THEME.space_2xl, THEME.space_xl, THEME.space_2xl, THEME.space_2xl)
        footer.addStretch()
        close_btn = QPushButton(tr("Close"))
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        right_layout.addLayout(footer)
        root.addWidget(right)

        self._rebuild_list()
        pin_dialog_default(close_btn, scope=self)

    def _build_form(self) -> None:
        fl = self._form_layout

        self._no_preset_label = hint_label(tr("No preset selected. Add one with the + button."))
        fl.addWidget(self._no_preset_label)

        self._form_container = QWidget()
        form = QVBoxLayout(self._form_container)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(14)

        # Name & enabled
        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(tr("Preset Name"))
        self.name_edit.textChanged.connect(self._on_name_changed)
        self.enabled_check = labeled_toggle(
            "fa5s.check", tr("Enabled"), False, tr("Run this preset on every export; off keeps it for later")
        )
        self.enabled_check.toggled.connect(self._on_enabled_changed)
        row.addWidget(self.name_edit)
        row.addWidget(self.enabled_check)
        form.addLayout(row)

        self.intent_label = hint_label()
        form.addWidget(self.intent_label)

        # Shared FORMAT / SIZE / COLOR / DESTINATION rows.
        self.form = ExportSettingsForm()
        self.form.changed.connect(self._on_form_changed)
        form.addWidget(self.form)

        fl.addWidget(self._form_container)

    # List management

    def _rebuild_list(self) -> None:
        self.preset_list.blockSignals(True)
        self.preset_list.clear()
        for p in self._presets:
            self.preset_list.addItem(QListWidgetItem(_list_label(p)))
        self.preset_list.blockSignals(False)

        has = len(self._presets) > 0
        self._no_preset_label.setVisible(not has)
        self._form_container.setVisible(has)

    def _select_row(self, idx: int) -> None:
        if 0 <= idx < len(self._presets):
            self._selected_idx = idx
            self.preset_list.blockSignals(True)
            self.preset_list.setCurrentRow(idx)
            self.preset_list.blockSignals(False)
            self._populate_form(self._presets[idx])

    def _on_list_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._presets):
            return
        self._select_row(row)

    # Form population and change handling

    def _populate_form(self, preset: ExportPreset) -> None:
        self._updating_form = True
        try:
            is_flat = preset.render_intent == RenderIntent.FLAT
            self.intent_label.setText(
                tr("Flat: a flat master for editing in another app (16-bit TIFF or lossless JPEG XL).")
                if is_flat
                else tr("Print: the picture as you see it.")
            )
            self.form.set_flat_mode(is_flat)
            self.name_edit.setText(preset.name)
            self.enabled_check.setChecked(preset.enabled)
            self.form.load(preset.to_dict())
        finally:
            self._updating_form = False

    def _on_name_changed(self, text: str) -> None:
        if self._updating_form or self._selected_idx < 0:
            return
        self._presets[self._selected_idx].name = text
        item = self.preset_list.item(self._selected_idx)
        if item:
            item.setText(_list_label(self._presets[self._selected_idx]))
        self._emit_changed()

    def _on_enabled_changed(self, _on: bool) -> None:
        if self._updating_form or self._selected_idx < 0:
            return
        self._presets[self._selected_idx].enabled = self.enabled_check.isChecked()
        item = self.preset_list.item(self._selected_idx)
        if item:
            item.setText(_list_label(self._presets[self._selected_idx]))
        self._emit_changed()

    def _on_form_changed(self) -> None:
        if self._updating_form or self._selected_idx < 0:
            return
        self._write_form_to_preset(self._presets[self._selected_idx])
        self._emit_changed()

    def _write_form_to_preset(self, preset: ExportPreset) -> None:
        """Every form key that names a preset field, so a row added to the form
        cannot be left behind here. id/name/enabled/render_intent are not form
        keys and stay as they are."""
        for key, value in self.form.values().items():
            if key in ExportPreset.__dataclass_fields__:
                setattr(preset, key, value)
        preset.filename_pattern = preset.filename_pattern or "{{ original_name }}"

    # Preset actions

    def _show_add_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction(tr("Print Preset"), self._add_print_preset)
        menu.addAction(tr("Flat Master Preset"), self._add_flat_preset)
        menu.exec(self.add_btn.mapToGlobal(self.add_btn.rect().bottomLeft()))

    def _append_preset(self, preset: ExportPreset) -> None:
        self._presets.append(preset)
        self._rebuild_list()
        self._select_row(len(self._presets) - 1)
        self._emit_changed()

    def _add_print_preset(self) -> None:
        self._append_preset(ExportPreset(id=str(uuid.uuid4()), name=tr("New Preset")))

    def _add_flat_preset(self) -> None:
        self._append_preset(
            ExportPreset(
                id=str(uuid.uuid4()),
                name=tr("Flat Master"),
                render_intent=RenderIntent.FLAT,
                export_fmt=ExportFormat.TIFF,
                export_resolution_mode=ExportResolutionMode.ORIGINAL.value,
                export_color_space=ColorSpace.PROPHOTO.value,
                filename_pattern="{{ original_name }}_flat",
            )
        )

    def _duplicate_preset(self) -> None:
        if self._selected_idx < 0:
            return
        src = self._presets[self._selected_idx]
        dup = ExportPreset.from_dict({**src.to_dict(), "id": str(uuid.uuid4()), "name": tr("{name} Copy", name=src.name)})
        self._presets.insert(self._selected_idx + 1, dup)
        self._rebuild_list()
        self._select_row(self._selected_idx + 1)
        self._emit_changed()

    def _delete_preset(self) -> None:
        if self._selected_idx < 0 or not self._presets:
            return
        if not confirm_delete_named(self, tr("Preset"), preset_display_name(self._presets[self._selected_idx])):
            return
        self._presets.pop(self._selected_idx)
        new_idx = min(self._selected_idx, len(self._presets) - 1)
        self._rebuild_list()
        if new_idx >= 0:
            self._select_row(new_idx)
        else:
            self._selected_idx = -1
            self._no_preset_label.setVisible(True)
            self._form_container.setVisible(False)
        self._emit_changed()

    def _move_up(self) -> None:
        idx = self._selected_idx
        if idx <= 0:
            return
        self._presets[idx - 1], self._presets[idx] = self._presets[idx], self._presets[idx - 1]
        self._rebuild_list()
        self._select_row(idx - 1)
        self._emit_changed()

    def _move_down(self) -> None:
        idx = self._selected_idx
        if idx < 0 or idx >= len(self._presets) - 1:
            return
        self._presets[idx], self._presets[idx + 1] = self._presets[idx + 1], self._presets[idx]
        self._rebuild_list()
        self._select_row(idx + 1)
        self._emit_changed()

    # Helpers

    @staticmethod
    def _copy_preset(p: ExportPreset) -> ExportPreset:
        return ExportPreset.from_dict(p.to_dict())

    def _emit_changed(self) -> None:
        self.presets_changed.emit(list(self._presets))
