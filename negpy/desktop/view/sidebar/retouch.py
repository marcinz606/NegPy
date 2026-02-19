from PyQt6.QtWidgets import QPushButton, QHBoxLayout
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider, SignalSlider
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.session import ToolMode
from negpy.desktop.view.styles.theme import THEME


class RetouchSidebar(BaseSidebar):
    """
    Panel for dust removal and healing.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(10)
        conf = self.state.config.retouch


        auto_row = QHBoxLayout()
        self.threshold_slider = CompactSlider("Threshold", 0.01, 1.0, conf.dust_threshold)
        self.auto_size_slider = CompactSlider("Auto Size", 3.0, 8.0, float(conf.dust_size), step=1.0, precision=1)
        auto_row.addWidget(self.threshold_slider)
        auto_row.addWidget(self.auto_size_slider)
        self.layout.addLayout(auto_row)

        buttons_row = QHBoxLayout()
        self.auto_dust_btn = QPushButton(" Auto Dust")
        self.auto_dust_btn.setCheckable(True)
        self.auto_dust_btn.setChecked(conf.dust_remove)
        self.auto_dust_btn.setIcon(qta.icon("fa5s.magic", color=THEME.text_primary))
        
        self.pick_dust_btn = QPushButton(" Heal Tool")
        self.pick_dust_btn.setCheckable(True)
        self.pick_dust_btn.setIcon(qta.icon("fa5s.eye-dropper", color=THEME.text_primary))
        
        buttons_row.addWidget(self.auto_dust_btn)
        buttons_row.addWidget(self.pick_dust_btn)
        self.layout.addLayout(buttons_row)

        self.manual_size_slider = SignalSlider("Brush Size", 2.0, 16.0, float(conf.manual_dust_size), step=1.0, precision=1)
        self.layout.addWidget(self.manual_size_slider)

        actions_row = QHBoxLayout()
        self.undo_btn = QPushButton(" Undo Last")
        self.undo_btn.setIcon(qta.icon("fa5s.undo", color=THEME.text_primary))

        self.clear_btn = QPushButton(" Clear All")
        self.clear_btn.setIcon(qta.icon("fa5s.trash-alt", color=THEME.text_primary))

        actions_row.addWidget(self.undo_btn)
        actions_row.addWidget(self.clear_btn)
        self.layout.addLayout(actions_row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.auto_dust_btn.toggled.connect(lambda c: self.update_config_section("retouch", dust_remove=c))
        self.threshold_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", readback_metrics=False, dust_threshold=v)
        )
        self.auto_size_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", readback_metrics=False, dust_size=int(v))
        )
        self.pick_dust_btn.toggled.connect(self._on_pick_toggled)
        self.manual_size_slider.valueChanged.connect(
            lambda v: self.update_config_section("retouch", render=False, persist=True, manual_dust_size=int(v))
        )
        self.undo_btn.clicked.connect(self.controller.undo_last_retouch)
        self.clear_btn.clicked.connect(self.controller.clear_retouch)

    def _on_pick_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(ToolMode.DUST_PICK if checked else ToolMode.NONE)

    def sync_ui(self) -> None:
        conf = self.state.config.retouch
        self.block_signals(True)
        try:
            self.auto_dust_btn.setChecked(conf.dust_remove)
            self.threshold_slider.setValue(conf.dust_threshold)
            self.auto_size_slider.setValue(float(conf.dust_size))
            self.manual_size_slider.setValue(float(conf.manual_dust_size))
            self.pick_dust_btn.setChecked(self.state.active_tool == ToolMode.DUST_PICK)

            has_spots = len(conf.manual_dust_spots) > 0
            self.undo_btn.setEnabled(has_spots)
            self.clear_btn.setEnabled(has_spots)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        widgets = [
            self.auto_dust_btn,
            self.threshold_slider,
            self.auto_size_slider,
            self.manual_size_slider,
            self.pick_dust_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
