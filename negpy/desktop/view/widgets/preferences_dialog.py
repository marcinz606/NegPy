import os
from dataclasses import dataclass, fields

import qtawesome as qta
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
from negpy.desktop.view.styles.templates import default_button_height, field_label, hint_label
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.sliders import apply_slider_value_visibility
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
    # Stored in bytes, shown in MB. Nothing else needs a unit conversion.
    scale: int = 1


NUMBER_ROWS: tuple[NumberRow, ...] = (
    NumberRow(
        "preview_render_size",
        "Preview size",
        PREVIEW_SIZE_MIN,
        PREVIEW_SIZE_MAX,
        128,
        " px",
        "Long edge of the interactive canvas. Higher is sharper at 100% zoom, and more VRAM and CPU per frame.",
    ),
    NumberRow(
        "preview_cache_max_entries",
        "Preview cache",
        1,
        64,
        1,
        " photos",
        "How many recently-viewed photos stay decoded in memory for instant navigation.",
    ),
    NumberRow(
        "preview_cache_max_bytes",
        "Preview cache limit",
        128,
        32768,
        128,
        " MB",
        "Memory ceiling for that cache. Lower it on a machine with little RAM.",
        scale=1024 * 1024,
    ),
    NumberRow(
        "preview_cache_max_full_res_entries",
        "HQ buffers",
        1,
        16,
        1,
        " photos",
        "Full-resolution HQ buffers kept in memory. Each is hundreds of MB, and keeping the previous frame makes going back instant.",
    ),
    NumberRow(
        "render_memo_max_entries",
        "Rendered frames",
        1,
        64,
        1,
        " frames",
        "Rendered frames held for navigating back with no re-render.",
    ),
    NumberRow(
        "max_texture_size",
        "GPU texture cap",
        0,
        16384,
        1024,
        " px",
        "Largest GPU texture dimension, including HQ preview loads. 0 lets the hardware "
        "decide (a lower default applies automatically on integrated GPUs).",
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

    Live-apply: each row writes as it changes, the way the overflow menu entries it replaces
    did. Rows read at startup say so, and light the restart hint instead of applying.
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

        self.setWindowTitle("Preferences")
        self.resize(700, 720)
        self._init_ui()

    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        intro = QLabel("Settings for the whole application. Changes apply as you make them, except where a row says otherwise.")
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
            ("Interface", self._build_interface),
            ("Performance", self._build_performance),
            ("Session & Storage", self._build_storage),
        ):
            section = CollapsibleSection(title)
            section.set_content(builder())
            sections.addWidget(section)

        sections.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        footer = QHBoxLayout()
        footer.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def _grid(self) -> tuple[QWidget, QGridLayout]:
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(THEME.space_xl, THEME.space_md, THEME.space_xl, THEME.space_md)
        grid.setHorizontalSpacing(THEME.space_xl)
        grid.setVerticalSpacing(THEME.space_md)
        grid.setColumnStretch(1, 1)
        return host, grid

    def _add_checkbox(self, grid: QGridLayout, row: int, label: str, checked: bool, tooltip: str) -> QCheckBox:
        box = QCheckBox(label)
        box.setChecked(checked)
        box.setToolTip(tooltip)
        grid.addWidget(box, row, 0, 1, 2)
        return box

    def _build_interface(self) -> QWidget:
        host, grid = self._grid()
        state = self.session.state
        row = 0

        grid.addWidget(field_label("UI scale"), row, 0)
        self.scale_combo = QComboBox()
        for pct in UI_SCALES:
            self.scale_combo.addItem(f"{pct}%", pct / 100.0)
        current = float(self.repo.get_global_setting("ui_scale", 1.0) or 1.0)
        self.scale_combo.setCurrentIndex(min(range(len(UI_SCALES)), key=lambda i: abs(UI_SCALES[i] / 100.0 - current)))
        self.scale_combo.currentIndexChanged.connect(self._on_ui_scale_changed)
        self.scale_combo.setToolTip("Scale the whole interface")
        grid.addWidget(self.scale_combo, row, 1)
        row += 1

        grid.addWidget(field_label("Canvas background"), row, 0)
        pills = QHBoxLayout()
        pills.setSpacing(THEME.space_sm)
        self.canvas_group = QButtonGroup(self)
        self.canvas_group.setExclusive(True)
        self.canvas_pills: list[QPushButton] = []
        for idx, (hex_color, _, label) in enumerate(_canvas_colors()):
            pill = QPushButton()
            pill.setCheckable(True)
            pill.setChecked(idx == state.canvas_bg_index)
            pill.setFixedHeight(default_button_height())
            pill.setToolTip(label)
            pill.setStyleSheet(_pill_qss(hex_color))
            pill.clicked.connect(lambda _checked=False, i=idx: self._on_canvas_bg_changed(i))
            self.canvas_group.addButton(pill, idx)
            self.canvas_pills.append(pill)
            pills.addWidget(pill, 1)
        grid.addLayout(pills, row, 1)
        row += 1

        self.immersive_box = self._add_checkbox(
            grid, row, "Immersive canvas", state.immersive_canvas, "Toolbar overlaps the image, instead of sitting below it"
        )
        self.immersive_box.toggled.connect(self.session.set_immersive_canvas)
        row += 1

        self.sticky_zoom_box = self._add_checkbox(
            grid, row, "Sticky zoom", state.sticky_zoom, "Keep the zoom level when switching images, instead of resetting to fit"
        )
        self.sticky_zoom_box.toggled.connect(self.session.set_sticky_zoom)
        row += 1

        self.invert_zoom_box = self._add_checkbox(
            grid, row, "Reverse scroll zoom", state.invert_zoom_scroll, "Scroll up zooms out instead of in"
        )
        self.invert_zoom_box.toggled.connect(self.session.set_invert_zoom_scroll)
        row += 1

        self.slider_values_box = self._add_checkbox(
            grid,
            row,
            "Show slider values",
            bool(self.repo.get_global_setting("show_slider_values", default=False)),
            "Keep every slider's value box open, instead of revealing it on hover",
        )
        self.slider_values_box.toggled.connect(self._on_slider_values_changed)
        row += 1

        grid.addLayout(
            _button_row(
                (
                    ("Customize Shortcuts…", "fa5s.keyboard", self._open_shortcut_editor),
                    ("Edit Toolbar…", "fa5s.wrench", self._open_toolbar_editor),
                    ("Reset Panel Layout", "fa5s.thumbtack", self._reset_panel_layout),
                )
            ),
            row,
            0,
            1,
            2,
        )
        return host

    def _build_performance(self) -> QWidget:
        host, grid = self._grid()
        row = 0

        self.gpu_box = self._add_checkbox(
            grid, row, "GPU acceleration", self.session.state.gpu_enabled and self._gpu_available, "Render the pipeline on the GPU"
        )
        row += 1
        if self._gpu_available:
            self.gpu_box.toggled.connect(self._on_gpu_changed)
            grid.addWidget(hint_label(f"Active backend: {self._backend_name()}"), row, 0, 1, 2)
        else:
            self.gpu_box.setEnabled(False)
            grid.addWidget(hint_label("No GPU available on this hardware — the CPU pipeline is in use."), row, 0, 1, 2)
        row += 1

        self.parallel_box = self._add_checkbox(
            grid,
            row,
            "Multi-core CPU rendering",
            parallel_enabled(),
            "Spread the CPU rendering kernels across cores. Experimental: turn it off if the app closes without warning.",
        )
        self.parallel_box.toggled.connect(self._on_parallel_changed)
        row += 1
        grid.addWidget(hint_label(f"{os.cpu_count() or '?'} cores available. Applies at once, no restart."), row, 0, 1, 2)
        row += 1

        self.low_vram_tiling_box = self._add_checkbox(
            grid,
            row,
            "Reduce export memory use",
            APP_CONFIG.low_vram_export_tiling,
            "Use smaller tiles and less pipelining during export, at some cost to export speed. "
            "Turn this on if exporting crashes on your GPU (typically an older or "
            "memory-constrained integrated one).",
        )
        if "low_vram_export_tiling" in self._pinned:
            self.low_vram_tiling_box.setEnabled(False)
            self.low_vram_tiling_box.setToolTip("Set in override.toml, which wins over this dialog")
        else:
            self.low_vram_tiling_box.toggled.connect(self._on_low_vram_tiling_changed)
        row += 1
        note = "Applies to the next export started, no restart needed." + (
            " Set in override.toml." if "low_vram_export_tiling" in self._pinned else ""
        )
        grid.addWidget(hint_label(note), row, 0, 1, 2)
        row += 1

        for spec in NUMBER_ROWS:
            grid.addWidget(field_label(spec.label), row, 0)
            spin = QSpinBox()
            spin.setRange(spec.minimum, spec.maximum)
            spin.setSingleStep(spec.step)
            spin.setSuffix(spec.suffix)
            spin.setValue(self._stored_number(spec))
            if spec.key in self._pinned:
                spin.setEnabled(False)
                spin.setToolTip("Set in override.toml, which wins over this dialog")
            else:
                spin.valueChanged.connect(lambda value, s=spec: self._on_number_changed(s, value))
            self._spins[spec.key] = spin
            grid.addWidget(spin, row, 1)
            row += 1
            note = spec.hint + (" Set in override.toml." if spec.key in self._pinned else "")
            grid.addWidget(hint_label(note), row, 0, 1, 2)
            row += 1

        return host

    def _build_storage(self) -> QWidget:
        host, grid = self._grid()
        grid.addLayout(
            _button_row(
                (
                    ("Persistent Settings…", "fa5s.thumbtack", self._open_sticky_dialog),
                    ("Manage Database…", "fa5s.database", self._open_database_dialog),
                )
            ),
            0,
            0,
            1,
            2,
        )
        grid.addWidget(
            hint_label("Persistent Settings chooses which edits carry onto the next file you open."),
            1,
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
        self._restart_hint.setText("Restart NegPy to apply the changed startup settings.")
        self._restart_hint.show()

    def _on_number_changed(self, spec: NumberRow, value: int) -> None:
        self.repo.save_global_setting(spec.key, int(value) * spec.scale)
        self._mark_restart()

    def _on_ui_scale_changed(self, index: int) -> None:
        self.repo.save_global_setting("ui_scale", float(self.scale_combo.itemData(index)))
        self._mark_restart()

    def _on_canvas_bg_changed(self, index: int) -> None:
        self.session.set_canvas_bg(index)
        canvas = getattr(self.controller, "canvas", None)
        if canvas is not None:
            _, (r, g, b), _ = _canvas_colors()[index]
            canvas.set_background_color(r, g, b)

    def _on_slider_values_changed(self, checked: bool) -> None:
        self.repo.save_global_setting("show_slider_values", bool(checked))
        if self._window is not None:
            apply_slider_value_visibility(self._window, bool(checked))

    def _on_gpu_changed(self, checked: bool) -> None:
        if checked != self.session.state.gpu_enabled:
            self.session.set_gpu_enabled(checked)

    def _on_parallel_changed(self, checked: bool) -> None:
        """Takes effect at once: every kernel is compiled both ways and dispatched per
        call, so there is nothing to recompile and no restart to wait for."""
        set_parallel_enabled(bool(checked))
        self.repo.save_global_settings({"cpu_parallel": bool(checked), "cpu_parallel_active": bool(checked)})
        self.controller.set_status(
            "Multi-core CPU rendering on — turn it off if the app closes unexpectedly" if checked else "Multi-core CPU rendering off",
            5000,
        )

    def _on_low_vram_tiling_changed(self, checked: bool) -> None:
        APP_CONFIG.low_vram_export_tiling = bool(checked)
        self.repo.save_global_setting("low_vram_export_tiling", bool(checked))

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


def _pill_qss(hex_color: str) -> str:
    """A swatch that reads as the colour it sets. Checked state is an accent outline: a tick
    or a text label would have to sit on top of the swatch and fight it for contrast."""
    return (
        f"QPushButton {{background: {hex_color}; border: 1px solid {THEME.border_color}; "
        f"border-radius: {THEME.radius_md}px;}}"
        f"QPushButton:checked {{border: 2px solid {THEME.accent_secondary};}}"
    )


def _button_row(items) -> QHBoxLayout:
    """Equal thirds across the row: a settings dialog has no reason to rag its actions left."""
    layout = QHBoxLayout()
    layout.setSpacing(THEME.space_md)
    for label, icon, handler in items:
        btn = QPushButton(qta.icon(icon, color=THEME.text_primary), f" {label}")
        btn.clicked.connect(handler)
        layout.addWidget(btn, 1)
    return layout


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
