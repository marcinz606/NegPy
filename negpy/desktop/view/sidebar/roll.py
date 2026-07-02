import qtawesome as qta
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QPushButton,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.features.process.models import invalidate_local_bounds


class RollAnalysisSidebar(BaseSidebar):
    """
    Roll-wide normalization: batch analysis, roll-average axes, saved rolls.
    """

    def _init_ui(self) -> None:
        self.layout.setSpacing(12)
        conf = self.state.config.process

        self.layout.addWidget(section_subheader("BATCH"))

        btns_row = QHBoxLayout()
        self.analyze_roll_btn = QPushButton(" Batch Analysis")
        self.analyze_roll_btn.setIcon(qta.icon("fa5s.search", color=THEME.text_primary))
        self.analyze_roll_btn.setToolTip("Scan every loaded file and compute a roll-wide average density and colour balance baseline")

        btns_row.addWidget(self.analyze_roll_btn)
        self.layout.addLayout(btns_row)

        avg_row = QHBoxLayout()
        self.use_luma_avg_btn = QPushButton(" Use Luma Average")
        self.use_luma_avg_btn.setCheckable(True)
        self.use_luma_avg_btn.setChecked(conf.use_luma_average)
        self.use_luma_avg_btn.setIcon(qta.icon("mdi6.film", color=THEME.text_primary))
        self.use_luma_avg_btn.setToolTip(
            "Take the tonal-range (black/white-point) baseline from Batch Analysis; colour still re-derives per frame"
        )

        self.use_colour_avg_btn = QPushButton(" Use Colour Average")
        self.use_colour_avg_btn.setCheckable(True)
        self.use_colour_avg_btn.setChecked(conf.use_colour_average)
        self.use_colour_avg_btn.setIcon(qta.icon("mdi6.film", color=THEME.text_primary))
        self.use_colour_avg_btn.setToolTip(
            "Take the per-channel colour-balance baseline from Batch Analysis; luma range still re-derives per frame"
        )

        avg_row.addWidget(self.use_luma_avg_btn)
        avg_row.addWidget(self.use_colour_avg_btn)
        self.layout.addLayout(avg_row)

        self.layout.addWidget(section_subheader("ROLL"))

        self.roll_combo = QComboBox()
        self.roll_combo.setPlaceholderText("Select Roll...")
        self.roll_combo.setToolTip("Previously saved roll normalization baselines")
        self._refresh_rolls()
        self.layout.addWidget(self.roll_combo)

        roll_actions = QHBoxLayout()
        self.load_roll_btn = QPushButton(" Load")
        self.load_roll_btn.setIcon(qta.icon("fa5s.upload", color=THEME.text_primary))
        self.load_roll_btn.setToolTip("Apply the selected roll's bounds and balance to the current workspace")

        self.save_roll_btn = QPushButton(" Save")
        self.save_roll_btn.setIcon(qta.icon("fa5s.save", color=THEME.text_primary))
        self.save_roll_btn.setToolTip("Save the current Batch Analysis result as a named reusable roll")

        self.delete_roll_btn = QPushButton(" Delete")
        self.delete_roll_btn.setIcon(qta.icon("fa5s.trash", color=THEME.text_primary))
        self.delete_roll_btn.setToolTip("Remove the selected roll from the database")

        roll_actions.addWidget(self.load_roll_btn)
        roll_actions.addWidget(self.save_roll_btn)
        roll_actions.addWidget(self.delete_roll_btn)
        self.layout.addLayout(roll_actions)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.analyze_roll_btn.clicked.connect(self.controller.request_batch_normalization)
        self.use_luma_avg_btn.toggled.connect(self._on_use_luma_average_toggled)
        self.use_colour_avg_btn.toggled.connect(self._on_use_colour_average_toggled)

        self.load_roll_btn.clicked.connect(self._on_load_roll)
        self.save_roll_btn.clicked.connect(self._on_save_roll)
        self.delete_roll_btn.clicked.connect(self._on_delete_roll)
        self.sync_ui()

    def _on_use_luma_average_toggled(self, checked: bool) -> None:
        """Toggle the roll-wide luma (tonal-range) baseline for this axis only."""
        self._toggle_roll_axis(use_luma_average=checked)

    def _on_use_colour_average_toggled(self, checked: bool) -> None:
        """Toggle the roll-wide colour-balance baseline for this axis only."""
        self._toggle_roll_axis(use_colour_average=checked)

    def _toggle_roll_axis(self, **axis: bool) -> None:
        """
        Flip one roll-average axis. The other axis re-derives per frame, so we clear
        the cached local bounds to force a fresh analysis, and drop roll_name (the
        baseline is no longer applied as a named whole).
        """
        self.update_config_section(
            "process",
            persist=True,
            render=True,
            roll_name=None,
            **axis,
            **invalidate_local_bounds(self.state.config.process),
        )
        self.sync_ui()

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
            self.use_luma_avg_btn.setChecked(conf.use_luma_average)
            self.use_colour_avg_btn.setChecked(conf.use_colour_average)

            self._refresh_rolls()
            if conf.roll_name:
                self.roll_combo.setCurrentText(conf.roll_name)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all buttons.
        """
        widgets = [
            self.analyze_roll_btn,
            self.use_luma_avg_btn,
            self.use_colour_avg_btn,
            self.roll_combo,
            self.load_roll_btn,
            self.save_roll_btn,
            self.delete_roll_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
