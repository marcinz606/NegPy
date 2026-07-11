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
from negpy.desktop.view.styles.templates import EditedDot
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

        from negpy.desktop.view.sidebar.scanlight import ScanlightSidebar

        self.scanlight_sidebar = ScanlightSidebar(self.controller)

        # One "Scan" tab hosting both the SANE scanner and the RGB-Scan capture as
        # collapsible sections (mirrors the "Color — Lab, Toning" tab).
        self.scan_page = self._build_scan_page()

        # Tab descriptors: workflow control-group pages first, then Export / Metadata / Scan.
        # (key, icon_name, tooltip, content_widget, [section_attrs])
        tab_specs = [
            (page["key"], page["icon_name"], page["tooltip"], page["widget"], page["sections"]) for page in self.controls_panel.pages
        ]
        tab_specs += [
            ("history", "fa5s.history", "History", self.history_panel, []),
            ("export", "fa5s.file-export", "Export", self.export_sidebar, []),
            ("metadata", "fa5s.tags", "Metadata", self.metadata_sidebar, []),
            ("scan", "fa5s.camera-retro", "Scan", self.scan_page, []),
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

        for i, (key, icon_name, tooltip, content, section_attrs) in enumerate(tab_specs):
            btn = QPushButton()
            btn.setObjectName("right_tab_btn")
            btn.setIcon(qta.icon(icon_name, color=THEME.text_secondary))
            btn.setIconSize(QSize(18, 18))
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedHeight(38)
            btn.edited_dot = EditedDot(btn)
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

        # Collapsing the Analysis section should hand its splitter space back to the tabs
        # below (pinning the header at the top) instead of leaving a large empty pane.
        self._analysis_expanded_size = self.splitter.sizes()[0]
        self.analysis_section.expanded_changed.connect(self._resize_splitter_for_analysis)
        if not analysis_expanded:
            self._resize_splitter_for_analysis(False)

        layout.addWidget(self.splitter, 1)

        self.apply_shortcut_tooltips()

        # Default tab (Setup)
        self._switch_tab(0)

    def _resize_splitter_for_analysis(self, expanded: bool) -> None:
        """Pin the collapsed Analysis header at the top: shrink pane 0 to the header and
        give the rest to the tabs; restore the prior size when re-expanded."""
        sizes = self.splitter.sizes()
        total = sum(sizes)
        if total <= 0:
            return
        if expanded:
            top = min(max(self._analysis_expanded_size, 120), max(120, total - 120))
        else:
            self._analysis_expanded_size = sizes[0]
            top = max(1, self.analysis_section.sizeHint().height())
        self.splitter.setSizes([top, max(0, total - top)])

    def _build_scan_page(self) -> QWidget:
        """The 'Scan' tab hosts two collapsible sections (like Color's Lab / Toning): the
        SANE flatbed/film scanner on top, the RGB-Scan trichromatic capture below."""
        repo = self.controller.session.repo

        def make(title: str, key: str, icon_name: str, content: QWidget, default_expanded: bool) -> CollapsibleSection:
            persisted = repo.get_global_setting(f"section_expanded_{key}")
            expanded = bool(persisted) if persisted is not None else default_expanded
            section = CollapsibleSection(title, expanded=expanded, icon=qta.icon(icon_name, color="#aaa"))
            section.set_content(content)
            section.expanded_changed.connect(lambda checked, k=key: repo.save_global_setting(f"section_expanded_{k}", checked))
            return section

        self.scan_sane_section = make("Scanner (SANE)", "scan_sane", "fa5s.camera-retro", self.scan_sidebar, False)
        self.scan_rgb_section = make("Camera Scanning", "scan_rgb", "fa5s.camera", self.scanlight_sidebar, True)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)
        page_layout.addWidget(self.scan_sane_section)
        page_layout.addWidget(self.scan_rgb_section)
        return page

    def apply_shortcut_tooltips(self) -> None:
        """Append the current keyboard shortcut (action id `tab_<key>`) to each tab tooltip."""
        for btn, key, base in zip(self._tab_buttons, self._tab_keys, self._tab_tooltips):
            btn.setToolTip(tooltip_with_shortcut(base, f"tab_{key}"))

    def _connect_signals(self) -> None:
        self.controller.image_updated.connect(self._update_analysis)
        self.controller.metrics_available.connect(self._on_metrics_available)
        self.controller.pixel_readout_rgb.connect(self.hist_widget.set_marker)
        self.controller.tone_drag_changed.connect(self.curve_widget.set_active_param)
        self.controller.config_updated.connect(self.export_sidebar.sync_ui)
        self.controller.config_updated.connect(self.metadata_sidebar.sync_ui)
        self.controls_panel.modified_synced.connect(self._sync_tab_edited)

    def _sync_tab_edited(self) -> None:
        """Mark control-group tabs whose sections have edits (corner dot, like edited sliders)."""
        for i, attrs in self._tab_sections.items():
            self._tab_edited[i] = any(getattr(getattr(self.controls_panel, a), "modified_count", 0) for a in attrs)
        self._refresh_tab_icons()

    def _refresh_tab_icons(self) -> None:
        for i, btn in enumerate(self._tab_buttons):
            color = "white" if i == self._active_index else THEME.text_secondary
            btn.setIcon(qta.icon(self._tab_icons[i], color=color))
            btn.edited_dot.set_active(self._tab_edited[i])

    def _switch_tab(self, index: int) -> None:
        self._active_index = index
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self._tab_buttons):
            btn.setChecked(i == index)
        self._refresh_tab_icons()

        # Trigger device detection + gating refresh when the Scan tab is selected — it now
        # hosts both the SANE scanner and the RGB-Scan capture as collapsible sections.
        if index == self._scan_index:
            if hasattr(self.scan_sidebar, "on_activated"):
                self.scan_sidebar.on_activated()
            if hasattr(self.scanlight_sidebar, "on_activated"):
                self.scanlight_sidebar.on_activated()

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
                grade_trims=(config.grade_trim_red, config.grade_trim_green, config.grade_trim_blue),
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
