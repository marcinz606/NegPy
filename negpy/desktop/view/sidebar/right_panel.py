import sys
from typing import Any, Dict

import qtawesome as qta
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.controller import AppController
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.controls_panel import ControlsPanel
from negpy.desktop.view.sidebar.export import ExportSidebar
from negpy.desktop.view.sidebar.history import HistoryPanel
from negpy.desktop.view.sidebar.metadata import MetadataSidebar
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.charts import HistogramWidget, PhotometricCurveWidget
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.stats import NegativeStatsWidget


class RightPanel(QWidget):
    """
    Right sidebar panel: a sticky (collapsible) Analysis section pinned at the top,
    above an icon-only tab switcher hosting the workflow control groups
    (Setup / Tone / Color / Finish) plus Export / Metadata / Scan.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sticky Analysis section (collapsible, pinned at top)
        analysis_content = QWidget()
        analysis_layout = QVBoxLayout(analysis_content)
        analysis_layout.setContentsMargins(5, 5, 5, 5)

        self.hist_widget = HistogramWidget()
        self.curve_widget = PhotometricCurveWidget()
        self.stats_widget = NegativeStatsWidget()

        repo = self.controller.session.repo
        self.hist_widget.set_log_scale(bool(repo.get_global_setting("histogram_log_scale")))
        self.hist_widget.scale_changed.connect(lambda enabled: repo.save_global_setting("histogram_log_scale", bool(enabled)))

        analysis_layout.addWidget(self.hist_widget, 1)
        analysis_layout.addWidget(self.curve_widget, 1)
        analysis_layout.addWidget(self.stats_widget, 0)

        repo = self.controller.session.repo
        persisted = repo.get_global_setting("section_expanded_analysis")
        analysis_expanded = bool(persisted) if persisted is not None else THEME.sidebar_expanded_defaults.get("analysis", True)
        self.analysis_section = CollapsibleSection(
            "Analysis",
            expanded=analysis_expanded,
            icon=qta.icon("fa5s.chart-bar", color="#aaa"),
        )
        self.analysis_section.set_content(analysis_content)
        self.analysis_section.expanded_changed.connect(lambda checked: repo.save_global_setting("section_expanded_analysis", checked))

        def wrap_scroll(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            scroll.setStyleSheet("QScrollArea { border: none; }")
            return scroll

        # Tab content widgets
        self.controls_panel = ControlsPanel(self.controller)
        self.export_sidebar = ExportSidebar(self.controller)
        self.metadata_sidebar = MetadataSidebar(self.controller)
        self.history_panel = HistoryPanel(self.controller)

        from negpy.desktop.view.sidebar.scan import ScanSidebar, _ScanUnsupportedPlaceholder

        if sys.platform == "win32":
            self.scan_sidebar = _ScanUnsupportedPlaceholder()
        else:
            self.scan_sidebar = ScanSidebar(self.controller)

        # Tab descriptors: workflow control-group pages first, then Export / Metadata / Scan.
        # (key, icon_name, tooltip, content_widget, [section_attrs])
        tab_specs = [
            (page["key"], page["icon_name"], page["tooltip"], page["widget"], page["sections"]) for page in self.controls_panel.pages
        ]
        tab_specs += [
            ("history", "fa5s.history", "History", self.history_panel, []),
            ("export", "fa5s.file-export", "Export", self.export_sidebar, []),
            ("metadata", "fa5s.tags", "Metadata", self.metadata_sidebar, []),
            ("scan", "fa5s.camera-retro", "Scan", self.scan_sidebar, []),
        ]

        # Icon-only tab switcher
        switcher_layout = QHBoxLayout()
        switcher_layout.setContentsMargins(0, 0, 0, 0)
        switcher_layout.setSpacing(0)

        self.stack = QStackedWidget()
        self.stack.setContentsMargins(0, 8, 0, 0)

        self._tab_buttons: list[QPushButton] = []
        self._tab_keys: list[str] = []
        self._tab_icons: list[str] = []
        self._tab_tooltips: list[str] = []
        self._section_tab_index: dict[str, int] = {}
        self._tab_sections: dict[int, list[str]] = {}
        self._tab_edited: list[bool] = []
        self._active_index = 0
        self._scan_index = -1

        tab_style = """
            QPushButton {
                background-color: #0D0D0D;
                border: none;
                border-bottom: 1px solid #262626;
                border-right: 1px solid #262626;
            }
            QPushButton:hover {
                background-color: #262626;
            }
            QPushButton:checked {
                background-color: #222222;
                border-bottom: none;
            }
        """

        for i, (key, icon_name, tooltip, content, section_attrs) in enumerate(tab_specs):
            btn = QPushButton()
            btn.setIcon(qta.icon(icon_name, color=THEME.text_secondary))
            btn.setIconSize(QSize(18, 18))
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedHeight(38)
            btn.setStyleSheet(tab_style)
            btn.clicked.connect(lambda _checked=False, idx=i: self._switch_tab(idx))
            switcher_layout.addWidget(btn, 1)

            self.stack.addWidget(wrap_scroll(content))
            self._tab_buttons.append(btn)
            self._tab_keys.append(key)
            self._tab_icons.append(icon_name)
            self._tab_tooltips.append(tooltip)
            self._tab_edited.append(False)
            if section_attrs:
                self._tab_sections[i] = section_attrs
            for attr in section_attrs:
                self._section_tab_index[attr] = i
            if key == "scan":
                self._scan_index = i

        # Tabs (switcher + stack) live in the bottom splitter pane
        tabs_container = QWidget()
        tabs_vbox = QVBoxLayout(tabs_container)
        tabs_vbox.setContentsMargins(0, 0, 0, 0)
        tabs_vbox.setSpacing(0)
        tabs_vbox.addLayout(switcher_layout)
        tabs_vbox.addWidget(self.stack, 1)

        # Vertical splitter lets the user resize Analysis vs. the tabs below
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.analysis_section)
        self.splitter.addWidget(tabs_container)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        saved_sizes = repo.get_global_setting("analysis_splitter_sizes")
        if isinstance(saved_sizes, list) and len(saved_sizes) == 2:
            self.splitter.setSizes([int(s) for s in saved_sizes])
        else:
            self.splitter.setSizes([320, 600])
        self.splitter.splitterMoved.connect(lambda *_: repo.save_global_setting("analysis_splitter_sizes", self.splitter.sizes()))

        layout.addWidget(self.splitter, 1)

        self.apply_shortcut_tooltips()

        # Default tab (Setup)
        self._switch_tab(0)

    def apply_shortcut_tooltips(self) -> None:
        """Append the current keyboard shortcut (action id `tab_<key>`) to each tab tooltip."""
        for btn, key, base in zip(self._tab_buttons, self._tab_keys, self._tab_tooltips):
            btn.setToolTip(tooltip_with_shortcut(base, f"tab_{key}"))

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self._update_analysis)
        self.controller.metrics_available.connect(self._on_metrics_available)
        self.controller.config_updated.connect(self.export_sidebar.sync_ui)
        self.controls_panel.modified_synced.connect(self._sync_tab_edited)

    def _sync_tab_edited(self) -> None:
        """Mark control-group tabs whose sections have edits (yellow icon, like edited sliders)."""
        for i, attrs in self._tab_sections.items():
            self._tab_edited[i] = any(getattr(getattr(self.controls_panel, a), "modified_count", 0) for a in attrs)
        self._refresh_tab_icons()

    def _refresh_tab_icons(self) -> None:
        for i, btn in enumerate(self._tab_buttons):
            if i == self._active_index:
                color = "white"
            elif self._tab_edited[i]:
                color = THEME.accent_edited
            else:
                color = THEME.text_secondary
            btn.setIcon(qta.icon(self._tab_icons[i], color=color))

    def _switch_tab(self, index: int) -> None:
        self._active_index = index
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._tab_buttons):
            btn.setChecked(i == index)
        self._refresh_tab_icons()

        # Trigger device detection when the Scan tab is selected
        if index == self._scan_index and hasattr(self.scan_sidebar, "on_activated"):
            self.scan_sidebar.on_activated()

    def reveal_section(self, section_attr: str) -> None:
        """Switch to the tab containing the given ControlsPanel section."""
        idx = self._section_tab_index.get(section_attr)
        if idx is not None:
            self._switch_tab(idx)

    def show_tab_by_key(self, key: str) -> None:
        if key in self._tab_keys:
            self._switch_tab(self._tab_keys.index(key))

    def scroll_to(self, widget: QWidget) -> None:
        """Ensure *widget* is visible within its enclosing scroll area."""
        parent = widget.parent()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(widget)
                return
            parent = parent.parent()

    def _on_metrics_available(self, metrics: Dict[str, Any]) -> None:
        hist_data = metrics.get("histogram_raw")
        if hist_data is not None:
            self.hist_widget.update_data(hist_data)

    def _update_analysis(self) -> None:
        metrics = self.controller.session.state.last_metrics

        hist_data = metrics.get("histogram_raw")
        if hist_data is not None:
            self.hist_widget.update_data(hist_data)
        else:
            buffer = metrics.get("analysis_buffer")
            if buffer is None:
                buffer = metrics.get("base_positive")
            if buffer is not None:
                self.hist_widget.update_data(buffer)

        from negpy.features.exposure.logic import effective_grade_range, normalized_shadow_refs, per_channel_curve_params
        from negpy.features.exposure.papers import effective_paper_profile

        config = self.controller.session.state.config.exposure
        process_mode = self.controller.session.state.config.process.process_mode
        paper = effective_paper_profile(config.paper_profile, process_mode)

        # While peeking the flat master, plot the flat curve so the chart matches
        # what the canvas is showing.
        if self.controller.state.flat_peek:
            from negpy.domain.models import flat_master_config
            from negpy.features.exposure.logic import flat_curve_params

            flat_cfg = flat_master_config(self.controller.session.state.config).exposure
            gain, lift = flat_curve_params()
            # Flat log master has no print grade — the ISO-R contrast stat reads N/A.
            slope, density_range = None, None
            self.curve_widget.update_curve(flat_cfg, slope=gain, pivot=lift, flat=True)
        else:
            # Mirror PhotometricProcessor so the plotted curve matches the render under
            # the Auto Grade / Auto Density / Cast Removal toggles. CPU stores
            # "final_bounds", GPU stores "log_bounds".
            anchor = metrics.get("metered_anchor") if config.auto_exposure else None
            density_range = effective_grade_range(
                config.auto_normalize_contrast,
                metrics.get("norm_density_range"),
                metrics.get("textural_range"),
            )
            d_min = paper.d_min if config.paper_dmin else 0.0
            bounds = metrics.get("final_bounds") or metrics.get("log_bounds")
            shadow_refs_norm = normalized_shadow_refs(bounds, metrics.get("shadow_log_refs"))
            slopes, pivots, _ = per_channel_curve_params(
                config.grade,
                config.density,
                config.auto_normalize_contrast,
                config.cast_removal_strength,
                metrics.get("norm_density_range"),
                shadow_refs_norm,
                metrics.get("textural_range"),
                d_min=d_min,
                anchor=anchor,
                paper=paper,
            )
            # Green channel is the base curve (white reference + stats slope).
            slope, pivot = slopes[1], pivots[1]
            self.curve_widget.update_curve(config, slope=slope, pivot=pivot, slopes=slopes, pivots=pivots, process_mode=process_mode)

        from negpy.features.exposure.stats import negative_statistics

        clip_low, clip_high = self.hist_widget.clip_fractions()
        # Flat peek bypasses the print curve — the darkroom-units row reads N/A.
        flat = self.controller.state.flat_peek
        self.stats_widget.update_stats(
            negative_statistics(
                metrics.get("norm_density_range"),
                metrics.get("metered_anchor"),
                slope,
                clip_low,
                clip_high,
                effective_range=density_range,
                density=None if flat else config.density,
                wb_cmy=None if flat else (config.wb_cyan, config.wb_magenta, config.wb_yellow),
                scan_clip=metrics.get("scan_clip_fractions"),
            )
        )
