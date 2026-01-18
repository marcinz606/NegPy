from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from src.desktop.view.widgets.sliders import CompactSlider, SignalSlider
from src.desktop.controller import AppController
from src.desktop.session import ToolMode


class RetouchSidebar(QWidget):
    """
    Panel for dust removal and healing.
    """

    def __init__(self, controller: AppController):
        super().__init__()
        self.controller = controller
        self.state = controller.state

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 0, 5, 5)
        layout.setSpacing(10)

        conf = self.state.config.retouch

        # 1. Automatic Section
        layout.addWidget(QLabel("<b>Automatic Dust Removal</b>"))
        self.auto_dust_check = QPushButton("Enable Auto Removal")
        self.auto_dust_check.setCheckable(True)
        self.auto_dust_check.setChecked(conf.dust_remove)
        layout.addWidget(self.auto_dust_check)

        # Auto Controls (2 columns)
        auto_row = QHBoxLayout()
        self.threshold_slider = CompactSlider(
            "Threshold", 0.01, 1.0, conf.dust_threshold
        )
        self.auto_size_slider = CompactSlider(
            "Auto Size", 1.0, 10.0, float(conf.dust_size)
        )
        auto_row.addWidget(self.threshold_slider)
        auto_row.addWidget(self.auto_size_slider)
        layout.addLayout(auto_row)

        # 2. Manual Section
        layout.addWidget(QLabel("<b>Manual Healing</b>"))
        self.pick_dust_btn = QPushButton("Heal Dust (Manual Tool)")
        self.pick_dust_btn.setCheckable(True)
        layout.addWidget(self.pick_dust_btn)

        self.manual_size_slider = SignalSlider(
            "Brush Size", 2.0, 100.0, float(conf.manual_dust_size), step=1.0
        )
        layout.addWidget(self.manual_size_slider)

        self.clear_btn = QPushButton("Clear All Retouching")
        layout.addWidget(self.clear_btn)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.auto_dust_check.toggled.connect(self._on_auto_toggled)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.auto_size_slider.valueChanged.connect(self._on_auto_size_changed)
        self.pick_dust_btn.toggled.connect(self._on_pick_toggled)
        self.manual_size_slider.valueChanged.connect(self._on_manual_size_changed)
        self.clear_btn.clicked.connect(self.controller.clear_retouch)

    def _on_auto_toggled(self, checked: bool) -> None:
        from dataclasses import replace

        new_ret = replace(self.state.config.retouch, dust_remove=checked)
        self.controller.session.update_config(
            replace(self.state.config, retouch=new_ret)
        )
        self.controller.request_render()

    def _on_threshold_changed(self, val: float) -> None:
        from dataclasses import replace

        new_ret = replace(self.state.config.retouch, dust_threshold=val)
        self.controller.session.update_config(
            replace(self.state.config, retouch=new_ret)
        )
        self.controller.request_render()

    def _on_auto_size_changed(self, val: float) -> None:
        from dataclasses import replace

        new_ret = replace(self.state.config.retouch, dust_size=int(val))
        self.controller.session.update_config(
            replace(self.state.config, retouch=new_ret)
        )
        self.controller.request_render()

    def _on_manual_size_changed(self, val: float) -> None:
        from dataclasses import replace

        new_ret = replace(self.state.config.retouch, manual_dust_size=int(val))
        self.controller.session.update_config(
            replace(self.state.config, retouch=new_ret), persist=True
        )
        # No immediate re-render needed as this only affects FUTURE heal spots

    def _on_pick_toggled(self, checked: bool) -> None:
        self.controller.set_active_tool(
            ToolMode.DUST_PICK if checked else ToolMode.NONE
        )

    def sync_ui(self) -> None:
        conf = self.state.config.retouch
        self.auto_dust_check.blockSignals(True)
        self.threshold_slider.blockSignals(True)
        self.auto_size_slider.blockSignals(True)
        self.manual_size_slider.blockSignals(True)

        try:
            self.auto_dust_check.setChecked(conf.dust_remove)
            self.threshold_slider.setValue(conf.dust_threshold)
            self.auto_size_slider.setValue(float(conf.dust_size))
            self.manual_size_slider.setValue(float(conf.manual_dust_size))
        finally:
            self.auto_dust_check.blockSignals(False)
            self.threshold_slider.blockSignals(False)
            self.auto_size_slider.blockSignals(False)
            self.manual_size_slider.blockSignals(False)
