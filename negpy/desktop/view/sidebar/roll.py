from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
)

from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import labeled_action
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.features.process.models import invalidate_local_bounds

# A control character keeps this impossible to collide with a user's saved roll name.
_CURRENT_ROLL_ID = "\x00current"
_CURRENT_ROLL_LABEL = "Current Roll"


class RollAnalysisSidebar(BaseSidebar):
    """
    Roll-wide normalization: a searchable picker over saved baselines and the
    current session, one Apply action, then the per-axis average toggles.
    """

    def _init_ui(self) -> None:
        conf = self.state.config.process

        self.roll_combo = SearchableGearCombo(placeholder="Search rolls…")
        self.roll_combo.setToolTip("Current Roll scans the loaded files fresh; a saved name applies its stored baseline.")
        self.layout.addWidget(self.roll_combo)

        roll_actions = QHBoxLayout()
        self.apply_roll_btn = labeled_action(
            "fa5s.check",
            " Apply",
            "Apply the picked roll: Current Roll re-scans the loaded files for a fresh baseline, "
            "a saved name loads its stored bounds and balance",
            primary=True,
        )
        self.save_roll_btn = self._labeled_action(
            "fa5s.save", " Save", "Save the current roll baseline (bounds and balance) under a name, to reuse later"
        )
        self.delete_roll_btn = self._labeled_action("fa5s.trash", " Delete", "Remove the selected saved roll from the database")

        roll_actions.addWidget(self.apply_roll_btn)
        roll_actions.addWidget(self.save_roll_btn)
        roll_actions.addWidget(self.delete_roll_btn)
        self.layout.addLayout(roll_actions)

        self._roll_sync_key = None
        self._refresh_rolls(force=True)

        avg_row = QHBoxLayout()
        self.use_luma_avg_btn = self._small_toggle(
            "mdi6.film",
            "Use Luma Average",
            conf.use_luma_average,
            "Take the tonal-range (black/white-point) baseline from the picked roll; color still re-derives per frame",
        )

        self.use_color_avg_btn = self._small_toggle(
            "mdi6.film",
            "Use Color Average",
            conf.use_color_average,
            "Take the per-channel color-balance baseline from the picked roll; luma range still re-derives per frame",
        )

        avg_row.addWidget(self.use_luma_avg_btn)
        avg_row.addWidget(self.use_color_avg_btn)
        self.layout.addLayout(avg_row)

        self.layout.addStretch()

    def _connect_signals(self) -> None:
        self.apply_roll_btn.clicked.connect(self._on_apply_roll)
        self.use_luma_avg_btn.toggled.connect(self._on_use_luma_average_toggled)
        self.use_color_avg_btn.toggled.connect(self._on_use_color_average_toggled)

        self.save_roll_btn.clicked.connect(self._on_save_roll)
        self.delete_roll_btn.clicked.connect(self._on_delete_roll)
        self.roll_combo.selection_changed.connect(self._update_delete_enabled)
        self.sync_ui()

    def _on_use_luma_average_toggled(self, checked: bool) -> None:
        """Toggle the roll-wide luma (tonal-range) baseline for this axis only."""
        self._toggle_roll_axis(use_luma_average=checked)

    def _on_use_color_average_toggled(self, checked: bool) -> None:
        """Toggle the roll-wide color-balance baseline for this axis only."""
        self._toggle_roll_axis(use_color_average=checked)

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

    def _on_apply_roll(self) -> None:
        """Current Roll re-runs Batch Analysis on the loaded files; a saved name loads its baseline."""
        roll_id = self.roll_combo.selected_id()
        if roll_id and roll_id != _CURRENT_ROLL_ID:
            self.controller.apply_normalization_roll(roll_id)
        else:
            self.controller.request_batch_normalization()

    def _refresh_rolls(self, *, force: bool = False) -> None:
        """
        Rebuilds the picker from the saved-roll table, skipping a rebuild mid-search
        (SearchableGearCombo.is_editing) and one the roll set and selection don't need.
        """
        if not force and self.roll_combo.is_editing():
            return
        names = self.controller.session.repo.list_normalization_rolls()
        conf = self.state.config.process
        selected = conf.roll_name if conf.roll_name in names else _CURRENT_ROLL_ID
        key = (tuple(names), selected)
        if not force and key == self._roll_sync_key:
            return
        self._roll_sync_key = key
        entries = [(_CURRENT_ROLL_LABEL, _CURRENT_ROLL_ID)] + [(name, name) for name in names]
        self.roll_combo.set_labeled_items(entries, selected)
        self._update_delete_enabled()

    def _update_delete_enabled(self, *_args) -> None:
        """Current Roll is not a saved row, so Delete only applies to a real selection."""
        roll_id = self.roll_combo.selected_id()
        self.delete_roll_btn.setEnabled(bool(roll_id) and roll_id != _CURRENT_ROLL_ID)

    def _on_save_roll(self) -> None:
        """
        Prompts user for name and saves current normalization.
        """
        name, ok = QInputDialog.getText(self, "Save Roll", "Enter name for this roll:")
        if not ok or not name:
            return
        if name.strip().casefold() == _CURRENT_ROLL_LABEL.casefold():
            QMessageBox.warning(self, "Roll Name", f'"{_CURRENT_ROLL_LABEL}" is reserved for the loaded files. Choose another name.')
            return
        self.controller.save_current_normalization_as_roll(name)
        self._refresh_rolls(force=True)
        self.roll_combo.set_selected_id(name)

    def _on_delete_roll(self) -> None:
        """
        Removes selected roll from DB.
        """
        name = self.roll_combo.selected_id()
        if not name or name == _CURRENT_ROLL_ID:
            return
        if confirm_delete_named(
            self,
            "Roll",
            name,
            informative="The frames keep their current look; only the saved roll baseline goes.",
        ):
            self.controller.session.repo.delete_normalization_roll(name)
            self._refresh_rolls(force=True)

    def sync_ui(self) -> None:
        conf = self.state.config.process
        self.block_signals(True)
        try:
            self.use_luma_avg_btn.setChecked(conf.use_luma_average)
            self.use_color_avg_btn.setChecked(conf.use_color_average)
            self._refresh_rolls()
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all buttons.
        """
        widgets = [
            self.apply_roll_btn,
            self.use_luma_avg_btn,
            self.use_color_avg_btn,
            self.roll_combo,
            self.save_roll_btn,
            self.delete_roll_btn,
        ]
        for w in widgets:
            w.blockSignals(blocked)
