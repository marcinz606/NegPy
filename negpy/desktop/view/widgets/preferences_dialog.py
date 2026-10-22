from negpy.kernel.system.i18n import LANGUAGES, tr
import os
from dataclasses import dataclass, fields

from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.styles.color_vision import PALETTES
from negpy.desktop.view.styles.templates import field_label, hint_label, labeled_action, labeled_toggle, pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.desktop.view.widgets.semantic_download_dialog import ClipDownloadDialog
from negpy.desktop.view.widgets.step_combo import StepCombo
from negpy.domain.types import AppConfig
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.override import (
    PREVIEW_SIZE_DEFAULT,
    PREVIEW_SIZE_MAX,
    PREVIEW_SIZE_MIN,
    load_or_create,
    toml_pinned_keys,
)
from negpy.kernel.system.parallel import parallel_enabled, set_parallel_enabled
from negpy.services.assets import semantic_model

UI_SCALES: tuple[int, ...] = (80, 90, 100, 110, 120)


@dataclass(frozen=True)
class NumberRow:
    key: str
    label: str
    minimum: int
    maximum: int
    step: int
    suffix: str
    hint: str
    tooltip: str
    # Stored in bytes, shown in MB. Nothing else needs a unit conversion.
    scale: int = 1


NUMBER_ROWS: tuple[NumberRow, ...] = (
    NumberRow(
        "preview_render_size",
        tr("Preview size"),
        PREVIEW_SIZE_MIN,
        PREVIEW_SIZE_MAX,
        128,
        " px",
        tr("Long edge of the interactive canvas. Higher is sharper at 100% zoom, and more VRAM and CPU per frame."),
        tr("The long edge of the preview on the canvas."),
    ),
    NumberRow(
        "preview_cache_max_entries",
        tr("Preview cache"),
        1,
        64,
        1,
        tr(" photos"),
        tr("How many recently-viewed photos stay decoded in memory for instant navigation."),
        tr("How many recent photos stay decoded in memory."),
    ),
    NumberRow(
        "preview_cache_max_bytes",
        tr("Preview cache limit"),
        128,
        32768,
        128,
        " MB",
        tr("Memory ceiling for that cache. Lower it on a machine with little RAM."),
        tr("The most memory the preview cache can use."),
        scale=1024 * 1024,
    ),
    NumberRow(
        "preview_cache_max_full_res_entries",
        tr("HQ buffers"),
        1,
        16,
        1,
        tr(" photos"),
        tr("Full-size previews kept in memory. Each uses hundreds of MB; keeping the last one makes going back instant."),
        tr("How many full-size previews stay in memory."),
    ),
    NumberRow(
        "render_memo_max_entries",
        tr("Rendered frames"),
        1,
        64,
        1,
        tr(" frames"),
        tr("Rendered frames held for navigating back with no re-render."),
        tr("How many rendered frames stay in memory for going back."),
    ),
    NumberRow(
        "max_texture_size",
        tr("GPU texture cap"),
        0,
        16384,
        1024,
        " px",
        tr("Largest image size the GPU loads at once. 0 lets the hardware decide."),
        tr("The largest image size the GPU loads at once."),
    ),
)


def default_for(key: str) -> int:
    """The build's own value for a performance key, before any override or preference."""
    if key == "preview_render_size":
        return PREVIEW_SIZE_DEFAULT
    for f in fields(AppConfig):
        if f.name == key:
            return 0 if f.default is None else int(f.default)
    return 0


class PreferencesDialog(QDialog):
    """Every application-wide setting in one place.

    Each row writes as it changes. A row read at startup lights the restart hint instead.
    """

    def __init__(self, controller: AppController, parent=None, pinned_keys: set[str] | None = None):
        super().__init__(parent)
        self.controller = controller
        self.session = controller.session
        self.repo = self.session.repo
        self._window = parent
        self._gpu_available = GPUDevice.get().is_available
        self._pinned = _pinned_keys() if pinned_keys is None else pinned_keys
        self._spins: dict[str, QSpinBox] = {}

        self.setWindowTitle(tr("Preferences"))
        self.resize(700, 720)
        self._init_ui()
        remember_dialog_geometry(self, self.repo, "preferences")

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(THEME.space_xl)

        intro = QLabel(tr("Settings for the whole application. Changes apply as you make them, except where a row says otherwise."))
        intro.setWordWrap(True)
        root.addWidget(intro)

        self._restart_hint = hint_label("", "warning")
        self._restart_hint.hide()
        root.addWidget(self._restart_hint)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {THEME.border_color};")
        root.addWidget(divider)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        container = QWidget()
        sections = QVBoxLayout(container)
        sections.setContentsMargins(0, 0, 0, 0)
        sections.setSpacing(THEME.space_sm)

        for title, builder in (
            (tr("Interface"), self._build_interface),
            (tr("Accessibility"), self._build_accessibility),
            (tr("Performance"), self._build_performance),
            (tr("Session & Storage"), self._build_storage),
        ):
            section = CollapsibleSection(title)
            section.set_content(builder())
            sections.addWidget(section)

        sections.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        footer = QHBoxLayout()
        footer.addStretch()
        close_btn = QPushButton(tr("Close"))
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        root.addLayout(footer)
        pin_dialog_default(close_btn, scope=self)

    def _grid(self) -> tuple[QWidget, QGridLayout]:
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(THEME.space_xl, THEME.space_md, THEME.space_xl, THEME.space_md)
        grid.setHorizontalSpacing(THEME.space_xl)
        grid.setVerticalSpacing(THEME.space_md)
        grid.setColumnStretch(1, 1)
        return host, grid

    @staticmethod
    def _add_toggles(grid: QGridLayout, row: int, *toggles: QPushButton) -> None:
        """Toggles side by side across the row, two to a row."""
        line = QHBoxLayout()
        line.setSpacing(THEME.space_md)
        for toggle in toggles:
            line.addWidget(toggle, 1)
        grid.addLayout(line, row, 0, 1, 2)

    def _build_interface(self) -> QWidget:
        host, grid = self._grid()
        state = self.session.state
        row = 0

        self.language_combo = StepCombo.of(
            [tr(label) for _, label in LANGUAGES],
            tr("The language of the interface, applied when NegPy starts again."),
            [key for key, _ in LANGUAGES],
        )
        saved_language = self.repo.get_global_setting("ui_language", "system")
        self.language_combo.setCurrentIndex(max(0, self.language_combo.findData(saved_language)))
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        grid.addWidget(field_label(tr("Language")), row, 0)
        grid.addWidget(self.language_combo, row, 1)
        row += 1

        grid.addWidget(field_label(tr("UI scale")), row, 0)
        self.scale_combo = StepCombo()
        for pct in UI_SCALES:
            self.scale_combo.addItem(tr("{value}%", value=pct), pct / 100.0)
        current = float(self.repo.get_global_setting("ui_scale", 1.0) or 1.0)
        self.scale_combo.setCurrentIndex(min(range(len(UI_SCALES)), key=lambda i: abs(UI_SCALES[i] / 100.0 - current)))
        self.scale_combo.currentIndexChanged.connect(self._on_ui_scale_changed)
        self.scale_combo.setToolTip(tr("Scale the whole interface"))
        grid.addWidget(self.scale_combo, row, 1)
        row += 1

        grid.addWidget(field_label(tr("Canvas background")), row, 0)
        self.canvas_combo = StepCombo()
        # Darkest to lightest; the item data is the saved index, which is not in that order.
        for idx, (hex_color, _, label) in sorted(enumerate(_canvas_colors()), key=lambda pair: QColor(pair[1][0]).lightness()):
            self.canvas_combo.addItem(_swatch((hex_color,) * 4), label, idx)
        self.canvas_combo.setCurrentIndex(max(self.canvas_combo.findData(state.canvas_bg_index), 0))
        self.canvas_combo.setToolTip(tr("The color around the photo on the canvas"))
        self.canvas_combo.currentIndexChanged.connect(lambda i: self._on_canvas_bg_changed(self.canvas_combo.itemData(i)))
        grid.addWidget(self.canvas_combo, row, 1)
        row += 1

        self.sticky_zoom_box = labeled_toggle(
            "mdi6.magnify-plus-outline",
            tr("Sticky Zoom"),
            state.sticky_zoom,
            tr("Keep the zoom level when switching images, instead of resetting to fit"),
        )
        self.sticky_zoom_box.toggled.connect(self.session.set_sticky_zoom)
        self.invert_zoom_box = labeled_toggle(
            "mdi6.swap-vertical", tr("Reverse Scroll Zoom"), state.invert_zoom_scroll, tr("Scroll up zooms out instead of in")
        )
        self.invert_zoom_box.toggled.connect(self.session.set_invert_zoom_scroll)
        self._add_toggles(grid, row, self.sticky_zoom_box, self.invert_zoom_box)
        row += 1
        self.shortcut_hints_box = labeled_toggle(
            "fa5s.keyboard",
            tr("Shortcut Hints"),
            state.shortcut_hints,
            tr("After you use a control with the mouse, show its keyboard shortcut once per session"),
        )
        self.shortcut_hints_box.toggled.connect(self.session.set_shortcut_hints)
        self._add_toggles(grid, row, self.shortcut_hints_box)
        row += 1

        interface_row, _ = _button_row(
            (
                (tr("Customize Shortcuts…"), "fa5s.keyboard", tr("Change the keys for each action"), self._open_shortcut_editor),
                (tr("Edit Toolbar…"), "fa5s.wrench", tr("Choose the buttons on the canvas bar"), self._open_toolbar_editor),
                (tr("Reset Panel Layout"), "fa5s.thumbtack", tr("Move every panel back to its default place"), self._reset_panel_layout),
            )
        )
        grid.addLayout(interface_row, row, 0, 1, 2)
        return host

    def _build_accessibility(self) -> QWidget:
        host, grid = self._grid()
        grid.addWidget(field_label(tr("Color vision")), 0, 0)
        self.vision_combo = StepCombo()
        for palette in PALETTES:
            self.vision_combo.addItem(_swatch((*palette.pair, *palette.dodge_burn)), palette.label, palette.key)
        self.vision_combo.setCurrentIndex(max(self.vision_combo.findData(self.session.state.color_vision), 0))
        self.vision_combo.setToolTip(tr("Marks told apart by color alone use colors this vision keeps"))
        self.vision_combo.currentIndexChanged.connect(lambda i: self.session.set_color_vision(self.vision_combo.itemData(i)))
        grid.addWidget(self.vision_combo, 0, 1)
        grid.addWidget(
            hint_label(
                tr(
                    "Colors for the dust overlay, the Dodge & Burn masks and Printing Notes. The color-blind choices use the Okabe–Ito palette."
                )
            ),
            1,
            0,
            1,
            2,
        )
        return host

    def _build_performance(self) -> QWidget:
        host, grid = self._grid()
        row = 0

        def hint(widget_or_text, kind: str = "muted") -> None:
            nonlocal row
            grid.addWidget(hint_label(widget_or_text, kind), row, 0, 1, 2)
            row += 1

        self.gpu_box = labeled_toggle(
            "fa5s.microchip",
            tr("GPU Acceleration"),
            self.session.state.gpu_enabled and self._gpu_available,
            tr("Render the pipeline on the GPU"),
        )
        self.parallel_box = labeled_toggle(
            "mdi6.cpu-64-bit",
            tr("Multi-core CPU Rendering"),
            parallel_enabled(),
            tr("Render on all CPU cores, an experimental mode that can close the app without warning."),
        )
        self.parallel_box.toggled.connect(self._on_parallel_changed)
        self._add_toggles(grid, row, self.gpu_box, self.parallel_box)
        row += 1
        if self._gpu_available:
            self.gpu_box.toggled.connect(self._on_gpu_changed)
            hint(tr("Active backend: {value}", value=self._backend_name()))
            if self.session.state.gpu_viewport_failed:
                hint(
                    tr(
                        "GPU viewport failed to start ({gpu_viewport_failed}); the display runs on the CPU.",
                        gpu_viewport_failed=self.session.state.gpu_viewport_failed,
                    ),
                    "warning",
                )
        else:
            self.gpu_box.setEnabled(False)
            hint(tr("No GPU available on this hardware. NegPy uses the CPU."))
        hint(tr("{value} cores available. Applies at once, no restart.", value=os.cpu_count() or "?"))

        self.semantic_box = labeled_toggle(
            "mdi.image-search-outline",
            tr("Search by Meaning"),
            self.session.state.semantic_search_enabled,
            tr("Type a plain-language description in the Film Strip search box to find frames, instead of field:value terms"),
        )
        self.semantic_box.toggled.connect(self._on_semantic_search_changed)
        self.auto_thumbnails_box = labeled_toggle(
            "mdi6.image-sync-outline",
            tr("Update Thumbnails Automatically"),
            self.session.state.auto_thumbnail_refresh,
            tr("Render thumbnails again in the background after a change to many frames"),
        )
        self.auto_thumbnails_box.toggled.connect(self.session.set_auto_thumbnail_refresh)
        self._add_toggles(grid, row, self.semantic_box, self.auto_thumbnails_box)
        row += 1
        if not semantic_model.clip_model_ready():
            hint(
                tr(
                    "Downloads a small model ({MODEL_DOWNLOAD_SIZE}) the first time this is turned on.",
                    MODEL_DOWNLOAD_SIZE=semantic_model.MODEL_DOWNLOAD_SIZE,
                )
            )

        self.low_vram_tiling_box = labeled_toggle(
            "fa5s.memory",
            tr("Reduce Export Memory Use"),
            APP_CONFIG.low_vram_export_tiling,
            tr("Export in smaller pieces, which is slower but stops export crashes on a GPU with little memory."),
        )
        if "low_vram_export_tiling" in self._pinned:
            self.low_vram_tiling_box.setEnabled(False)
            self.low_vram_tiling_box.setToolTip(tr("Set in override.toml, which wins over this dialog"))
        else:
            self.low_vram_tiling_box.toggled.connect(self._on_low_vram_tiling_changed)
        self.vram_warning_box = labeled_toggle(
            "fa5s.exclamation-triangle",
            tr("Show GPU Memory Warning"),
            bool(self.repo.get_global_setting("show_vram_capped_warning", default=True)),
            tr("Show a message when a scan is reduced to fit the GPU texture cap."),
        )
        self.vram_warning_box.toggled.connect(self._on_vram_warning_changed)
        self._add_toggles(grid, row, self.low_vram_tiling_box, self.vram_warning_box)
        row += 1
        hint(
            tr("Reduce Export Memory Use applies to the next export started, no restart needed.")
            + (tr(" Set in override.toml.") if "low_vram_export_tiling" in self._pinned else "")
        )

        for spec in NUMBER_ROWS:
            grid.addWidget(field_label(spec.label), row, 0)
            spin = QSpinBox()
            spin.setRange(spec.minimum, spec.maximum)
            spin.setSingleStep(spec.step)
            spin.setSuffix(spec.suffix)
            spin.setValue(self._stored_number(spec))
            if spec.key in self._pinned:
                spin.setEnabled(False)
                spin.setToolTip(tr("Set in override.toml, which wins over this dialog"))
            else:
                spin.setToolTip(spec.tooltip)
                spin.valueChanged.connect(lambda value, s=spec: self._on_number_changed(s, value))
            self._spins[spec.key] = spin
            grid.addWidget(spin, row, 1)
            row += 1
            note = spec.hint + (tr(" Set in override.toml.") if spec.key in self._pinned else "")
            grid.addWidget(hint_label(note), row, 0, 1, 2)
            row += 1

        return host

    def _build_storage(self) -> QWidget:
        host, grid = self._grid()
        state = self.session.state

        self.sticky_settings_box = labeled_toggle(
            "fa5s.thumbtack",
            tr("Carry Settings Between Frames"),
            state.sticky_settings_enabled,
            tr("Apply Persistent Settings to a file with no saved edit, instead of starting at default settings"),
        )
        self.sticky_settings_box.toggled.connect(self._on_sticky_settings_changed)

        _row, buttons = _button_row(
            (
                (
                    tr("Persistent Settings…"),
                    "fa5s.thumbtack",
                    tr("Choose which edits carry onto the next file you open"),
                    self._open_sticky_dialog,
                ),
                (tr("Manage Database…"), "fa5s.database", tr("See and clear the data that NegPy stores"), self._open_database_dialog),
            )
        )
        self._persistent_settings_button = buttons[0]
        self._persistent_settings_button.setEnabled(state.sticky_settings_enabled)
        # The toggle beside the choice of what it carries; the database tools on their own row.
        self._add_toggles(grid, 0, self.sticky_settings_box, self._persistent_settings_button)
        self._add_toggles(grid, 1, buttons[1])

        grid.addWidget(
            hint_label(tr("Persistent Settings chooses which edits carry onto the next file you open.")),
            2,
            0,
            1,
            2,
        )
        return host

    def _backend_name(self) -> str:
        try:
            return str(self.controller.render_worker.processor.backend_name)
        except Exception:
            return "GPU"

    def _stored_number(self, spec: NumberRow) -> int:
        """What the spin box shows: the live value, which is already the resolved one."""
        live = getattr(APP_CONFIG, spec.key, None)
        value = default_for(spec.key) if live is None else int(live)
        return max(spec.minimum, min(spec.maximum, value // spec.scale))

    def _mark_restart(self) -> None:
        self._restart_hint.setText(tr("Restart NegPy to apply the changed startup settings."))
        self._restart_hint.show()

    def _on_number_changed(self, spec: NumberRow, value: int) -> None:
        self.repo.save_global_setting(spec.key, int(value) * spec.scale)
        self._mark_restart()

    def _on_ui_scale_changed(self, index: int) -> None:
        self.repo.save_global_setting("ui_scale", float(self.scale_combo.itemData(index)))
        self._mark_restart()

    def _on_language_changed(self, index: int) -> None:
        self.repo.save_global_setting("ui_language", self.language_combo.currentData())
        self._mark_restart()

    def _on_sticky_settings_changed(self, checked: bool) -> None:
        self.session.set_sticky_settings_enabled(checked)
        self._persistent_settings_button.setEnabled(checked)

    def _on_canvas_bg_changed(self, index: int) -> None:
        self.session.set_canvas_bg(index)
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            _, (r, g, b), _ = _canvas_colors()[index]
            canvas.set_background_color(r, g, b)

    def _on_gpu_changed(self, checked: bool) -> None:
        if checked != self.session.state.gpu_enabled:
            self.session.set_gpu_enabled(checked)

    def _on_semantic_search_changed(self, checked: bool) -> None:
        if checked and not semantic_model.clip_model_ready():
            if ClipDownloadDialog(self, repo=self.repo).exec() != QDialog.DialogCode.Accepted:
                self.semantic_box.setChecked(False)  # cancelled or failed -- stays off
                return
        if checked != self.session.state.semantic_search_enabled:
            self.session.set_semantic_search_enabled(checked)

    def _on_parallel_changed(self, checked: bool) -> None:
        """Takes effect at once: every kernel is compiled both ways and dispatched per
        call, so there is nothing to recompile and no restart to wait for."""
        set_parallel_enabled(bool(checked))
        self.repo.save_global_settings({"cpu_parallel": bool(checked), "cpu_parallel_active": bool(checked)})
        self.controller.set_status(
            tr("Multi-core CPU rendering is on. Turn it off if the app closes unexpectedly.")
            if checked
            else tr("Multi-core CPU rendering off"),
            5000,
        )

    def _on_low_vram_tiling_changed(self, checked: bool) -> None:
        APP_CONFIG.low_vram_export_tiling = bool(checked)
        self.repo.save_global_setting("low_vram_export_tiling", bool(checked))

    def _on_vram_warning_changed(self, checked: bool) -> None:
        self.repo.save_global_setting("show_vram_capped_warning", bool(checked))

    def _open_shortcut_editor(self) -> None:
        manager = getattr(self._window, "shortcut_manager", None)
        if manager is not None:
            manager.open_editor(self)

    def _open_toolbar_editor(self) -> None:
        toolbar = getattr(self._window, "toolbar", None)
        if toolbar is not None:
            toolbar.open_toolbar_editor()

    def _reset_panel_layout(self) -> None:
        window = self._window
        if window is not None and hasattr(window, "reset_panel_layout"):
            window.reset_panel_layout()

    def _open_sticky_dialog(self) -> None:
        from negpy.desktop.view.widgets.granular_settings_dialog import open_sticky_dialog

        open_sticky_dialog(self, self.controller)

    def _open_database_dialog(self) -> None:
        from negpy.desktop.view.widgets.database_dialog import DatabaseDialog

        DatabaseDialog(self.repo, self.controller, self).exec()


def _swatch(colors: tuple[str, ...]) -> QIcon:
    """Four colors in the quadrants of a framed square, for a combo item."""
    pixmap = QPixmap(16, 16)
    painter = QPainter(pixmap)
    for x, y, color in zip((0, 8, 0, 8), (0, 0, 8, 8), colors):
        painter.fillRect(x, y, 8, 8, QColor(color))
    painter.setPen(QColor(THEME.border_color))
    painter.drawRect(0, 0, 15, 15)
    painter.end()
    return QIcon(pixmap)


def _button_row(items) -> tuple[QHBoxLayout, list[QPushButton]]:
    """Equal thirds across the row: a settings dialog has no reason to rag its actions left."""
    layout = QHBoxLayout()
    layout.setSpacing(THEME.space_md)
    buttons = []
    for label, icon, tooltip, handler in items:
        btn = labeled_action(icon, label, tooltip)
        btn.clicked.connect(handler)
        layout.addWidget(btn, 1)
        buttons.append(btn)
    return layout, buttons


def _pinned_keys() -> set[str]:
    """Which performance keys override.toml has taken over, so their rows stand down."""
    path = APP_CONFIG.override_toml_path
    if not path or not os.path.exists(path):
        return set()
    cfg = load_or_create(path)
    pinned = toml_pinned_keys(cfg)
    if cfg.low_vram_export_tiling is not None:
        pinned = pinned | {"low_vram_export_tiling"}
    return pinned


def _canvas_colors():
    # Deferred: the toolbar reaches this dialog through the shortcut action map.
    from negpy.desktop.view.canvas.toolbar import CANVAS_COLORS

    return CANVAS_COLORS


def open_preferences(parent, controller: AppController) -> None:
    PreferencesDialog(controller, parent).exec()
