from PyQt6.QtWidgets import (
    QPushButton,
    QComboBox,
    QHBoxLayout,
    QInputDialog,
)
import qtawesome as qta
from negpy.desktop.view.widgets.sliders import CompactSlider
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.features.process.models import ProcessMode


class ProcessSidebar(BaseSidebar):
    """
    Panel for core film processing, normalization, and roll management.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.process

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([m.value for m in ProcessMode])
        self.mode_combo.setCurrentText(conf.process_mode)
        self.layout.addWidget(self.mode_combo)

        sliders_row = QHBoxLayout()
        self.analysis_buffer_slider = CompactSlider("Analysis Buffer", 0.0, 0.25, conf.analysis_buffer)
        self.shadow_cast_strength_slider = CompactSlider("Cast removal", 0.0, 1.0, conf.shadow_cast_strength)
        
        sliders_row.addWidget(self.analysis_buffer_slider)
        sliders_row.addWidget(self.shadow_cast_strength_slider)
        self.layout.addLayout(sliders_row)

        wp_bp_row = QHBoxLayout()
        self.white_point_slider = CompactSlider("White Point", -0.5, 0.5, conf.white_point_offset, has_neutral=True)
        self.black_point_slider = CompactSlider("Black Point", -0.5, 0.5, conf.black_point_offset, has_neutral=True)
        wp_bp_row.addWidget(self.white_point_slider)
        wp_bp_row.addWidget(self.black_point_slider)
        self.layout.addLayout(wp_bp_row)

        self.normalize_e6_btn = QPushButton(" Normalize")
        self.normalize_e6_btn.setFixedHeight(35)
        self.normalize_e6_btn.setCheckable(True)
        self.normalize_e6_btn.setIcon(qta.icon("fa5s.magic", color=THEME.text_primary))
        self.normalize_e6_btn.setChecked(conf.e6_normalize)
        self.normalize_e6_btn.setToolTip("Automatically stretch the histogram to full dynamic range")
        self.layout.addWidget(self.normalize_e6_btn)

        btns_row = QHBoxLayout()
        self.analyze_roll_btn = QPushButton(" Batch Analysis")
        self.analyze_roll_btn.setFixedHeight(35)
        self.analyze_roll_btn.setIcon(qta.icon("fa5s.search", color=THEME.text_primary))

        self.use_roll_avg_btn = QPushButton(" Use Roll Average")
        self.use_roll_avg_btn.setFixedHeight(35)
        self.use_roll_avg_btn.setCheckable(True)
        self.use_roll_avg_btn.setIcon(qta.icon("mdi6.film", color=THEME.text_primary))

        btns_row.addWidget(self.analyze_roll_btn)
        btns_row.addWidget(self.use_roll_avg_btn)
        self.layout.addLayout(btns_row)

        self.roll_combo = QComboBox()
        self.roll_combo.setPlaceholderText("Select Roll...")
        self._refresh_rolls()
        self.layout.addWidget(self.roll_combo)

        roll_actions = QHBoxLayout()
        self.load_roll_btn = QPushButton(" Load")
        self.load_roll_btn.setIcon(qta.icon("fa5s.upload", color=THEME.text_primary))

        self.save_roll_btn = QPushButton(" Save")
        self.save_roll_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))

        self.delete_roll_btn = QPushButton(" Delete")
        self.delete_roll_btn.setIcon(qta.icon("fa5s.trash", color=THEME.text_primary))

        roll_actions.addWidget(self.load_roll_btn)
        roll_actions.addWidget(self.save_roll_btn)
        roll_actions.addWidget(self.delete_roll_btn)
        self.layout.addLayout(roll_actions)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.analysis_buffer_slider.valueChanged.connect(self._on_buffer_changed)
        self.shadow_cast_strength_slider.valueChanged.connect(self._on_shadow_cast_strength_changed)
        self.white_point_slider.valueChanged.connect(self._on_white_point_changed)
        self.black_point_slider.valueChanged.connect(self._on_black_point_changed)
        self.normalize_e6_btn.toggled.connect(self._on_normalize_e6_toggled)
        self.analyze_roll_btn.clicked.connect(self.controller.request_batch_normalization)
        self.use_roll_avg_btn.toggled.connect(self._on_use_roll_average_toggled)

        self.load_roll_btn.clicked.connect(self._on_load_roll)
        self.save_roll_btn.clicked.connect(self._on_save_roll)
        self.delete_roll_btn.clicked.connect(self._on_delete_roll)

    def _on_shadow_cast_strength_changed(self, val: float) -> None:
        self.update_config_section("process", shadow_cast_strength=val, persist=True)

    def _on_white_point_changed(self, val: float) -> None:
        self.update_config_section("process", white_point_offset=val, persist=True)

    def _on_black_point_changed(self, val: float) -> None:
        self.update_config_section("process", black_point_offset=val, persist=True)

    def _on_mode_changed(self, mode: str) -> None:
        self.update_config_section("process", process_mode=mode, persist=True)
        self.sync_ui()

    def _on_normalize_e6_toggled(self, checked: bool) -> None:
        self.update_config_section("process", e6_normalize=checked, persist=True)

    def _on_buffer_changed(self, val: float) -> None:
        """
        Updates analysis buffer and forces local re-analysis.
        """
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            analysis_buffer=val,
            local_floors=(0.0, 0.0, 0.0),
            local_ceils=(0.0, 0.0, 0.0),
        )

    def _on_use_roll_average_toggled(self, checked: bool) -> None:
        """
        Toggles between Roll-wide baseline and Local auto-exposure.
        Forcing re-analysis when switching to Local.
        """
        if not checked:
            self.update_config_section(
                "process",
                persist=True,
                render=True,
                use_roll_average=False,
                local_floors=(0.0, 0.0, 0.0),
                local_ceils=(0.0, 0.0, 0.0),
                roll_name=None,
            )
        else:
            self.update_config_section("process", persist=True, render=True, use_roll_average=True)

    def _refresh_rolls(self) -> None:
        """
        Populates roll dropdown from database.
        """
        current = self.roll_combo.currentText()
        self.roll_combo.blockSignals(True)
        self.roll_combo.clear()
        rolls = self.controller.session.repo.list_normalization_rolls()
        self.roll_combo.addItems(rolls)
        if current in rolls:
            self.roll_combo.setCurrentText(current)
        else:
            self.roll_combo.setCurrentIndex(-1)
        self.roll_combo.blockSignals(False)

    def _on_load_roll(self) -> None:
        """
        Applies selected roll to session.
        """
        name = self.roll_combo.currentText()
        if name:
            self.controller.apply_normalization_roll(name)

    def _on_save_roll(self) -> None:
        """
        Prompts user for name and saves current normalization.
        """
        name, ok = QInputDialog.getText(self, "Save Roll", "Enter name for this roll:")
        if ok and name:
            self.controller.save_current_normalization_as_roll(name)
            self._refresh_rolls()
            self.roll_combo.setCurrentText(name)

    def _on_delete_roll(self) -> None:
        """
        Removes selected roll from DB.
        """
        name = self.roll_combo.currentText()
        if name:
            self.controller.session.repo.delete_normalization_roll(name)
            self._refresh_rolls()

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            self.mode_combo.setCurrentText(conf.process_mode)
            self.analysis_buffer_slider.setValue(conf.analysis_buffer)
            self.shadow_cast_strength_slider.setValue(conf.shadow_cast_strength)
            self.white_point_slider.setValue(conf.white_point_offset)
            self.black_point_slider.setValue(conf.black_point_offset)

            is_e6 = conf.process_mode == ProcessMode.E6
            self.normalize_e6_btn.setVisible(is_e6)
            self.normalize_e6_btn.setChecked(conf.e6_normalize)

            self.use_roll_avg_btn.setChecked(conf.use_roll_average)
            self._refresh_rolls()
            if conf.roll_name:
                self.roll_combo.setCurrentText(conf.roll_name)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all sliders and buttons.
        """
        widgets = [
            self.mode_combo,
            self.analysis_buffer_slider,
            self.shadow_cast_strength_slider,
            self.white_point_slider,
            self.black_point_slider,
            self.normalize_e6_btn,
            self.analyze_roll_btn,
            self.use_roll_avg_btn,
            self.roll_combo,
            self.load_roll_btn,
            self.save_roll_btn,
            self.delete_roll_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
