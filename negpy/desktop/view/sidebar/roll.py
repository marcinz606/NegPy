from typing import Optional

from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, section_subheader, set_hint_kind, wrap_tooltip
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.services.assets import rolls

# Prefixes a roll's label once it has a saved baseline -- the field itself is the
# "analyzed" indicator, so the search text ignores it (built from the plain name).
_TICK = "✓ "

# Batch Analysis is the action, Roll Baseline the field it fills; the Library's own
# menu item runs the same action, so it reads these rather than restating them.
BATCH_ANALYSIS_TOOLTIP = "Batch Analysis — measures every loaded frame's exposure and saves the average as this roll's baseline"
BATCH_ANALYSIS_DISABLED_TOOLTIP = "Open this roll first — Batch Analysis measures the files currently loaded."
FROM_FRAME_TOOLTIP = (
    "Use This Frame — save the current frame's bounds as this roll's baseline, in place of a measured "
    "average. Every frame on Use Luma/Color Average follows it, including one loaded later."
)
FROM_FRAME_DISABLED_TOOLTIP = "Open this roll first — the baseline is written onto the files currently loaded."


class RollAnalysisSidebar(BaseSidebar):
    """
    Which roll's Batch Analysis baseline this frame's Use Luma/Color Average axes
    borrow: a searchable picker over every roll in your library, ticked once it has
    one. Picking a roll loads its baseline immediately -- there is no separate Apply.
    Reanalyze, beside the picker, runs Batch Analysis itself (the metering pass that
    fills the tick in) -- the same action the Library's own "Batch Analysis" offers,
    reachable here too since this is where you notice a roll has never been measured.
    Use This Frame, beside it, writes the current frame's own bounds there instead, for
    a roll that wants one chosen frame as its reference rather than an average. Both are
    enabled only for the loaded roll, since both act on the files currently open, not on
    whichever one this picker happens to show.
    """

    def _init_ui(self) -> None:
        self.layout.addWidget(section_subheader("ROLL BASELINE"))
        self.roll_combo = SearchableGearCombo(placeholder="Search rolls…")
        self.roll_combo.setToolTip(wrap_tooltip("Picking a roll loads its saved baseline onto the loaded files."))
        self.layout.addWidget(self.roll_combo)

        row = QHBoxLayout()
        self.reanalyze_btn = self._labeled_action("fa5s.tachometer-alt", " Reanalyze", BATCH_ANALYSIS_TOOLTIP)
        self.from_frame_btn = self._labeled_action("fa5s.crosshairs", " Use This Frame", FROM_FRAME_TOOLTIP)
        for btn in (self.reanalyze_btn, self.from_frame_btn):
            row.addWidget(btn, 1)
        self.layout.addLayout(row)
        # Lock Bounds is adopted into this row (between the two) once ControlsPanel wires
        # it in -- see insert_lock_button.
        self._picker_row = row

        self.roll_status_hint = hint_label("", "muted")
        self.layout.addWidget(self.roll_status_hint)

        self._roll_sync_key = None
        self._refresh_rolls(force=True)
        self.layout.addStretch()

    def insert_lock_button(self, lock_bounds_btn) -> None:
        """Adopts ProcessSidebar's Lock Bounds toggle into the button row, between
        Reanalyze and Use This Frame. Lock Bounds is specifically about this frame's
        relationship to Batch Analysis, so it belongs beside the actions it exempts
        the frame from."""
        self._picker_row.insertWidget(1, lock_bounds_btn, 1)

    def _connect_signals(self) -> None:
        self.roll_combo.selection_changed.connect(self._on_roll_picked)
        self.reanalyze_btn.clicked.connect(self.controller.request_batch_normalization)
        self.from_frame_btn.clicked.connect(self._on_from_frame_clicked)
        self.sync_ui()

    def _on_roll_picked(self, roll_id: str) -> None:
        """Picking a roll is the whole action: it loads that roll's saved baseline
        onto the currently loaded files. A no-op if it has never been analyzed."""
        if roll_id:
            self.controller.apply_normalization_roll(roll_id)
        active_id = self.controller.state.active_roll_id
        self._update_roll_status_hint(active_id, roll_id)
        self._update_reanalyze_btn(active_id, roll_id)

    def _on_from_frame_clicked(self) -> None:
        active_id = self.controller.state.active_roll_id
        if active_id:
            self.controller.set_roll_baseline_from_frame(active_id)

    def _name_for_id(self, roll_id: str) -> str:
        entry = rolls.roll_for_id(self.controller.session.repo, roll_id)
        return entry["name"] if entry else roll_id

    def _refresh_rolls(self, *, force: bool = False) -> None:
        """
        Rebuilds the picker from every library roll, skipping a rebuild mid-search
        (SearchableGearCombo.is_editing) and one the roll set and selection don't need.
        The loaded roll is pinned first in the dropdown, ahead of the alphabetical
        rest, since it is the default choice.
        """
        if not force and self.roll_combo.is_editing():
            return
        repo = self.controller.session.repo
        active_id = self.controller.state.active_roll_id
        listed = rolls.all_rolls_sorted(repo)
        analyzed = {rid for rid, _entry in listed if rolls.roll_normalization(repo, rid)}
        conf = self.state.config.process
        selected = active_id or ""
        if conf.roll_name:
            matching = next((rid for rid, entry in listed if entry.get("name") == conf.roll_name), None)
            if matching:
                selected = matching
        key = (tuple(rid for rid, _entry in listed), tuple(sorted(analyzed)), selected, active_id)
        if not force and key == self._roll_sync_key:
            return
        self._roll_sync_key = key
        ordered = [pair for pair in listed if pair[0] == active_id] + [pair for pair in listed if pair[0] != active_id]
        entries = [(f"{_TICK}{entry.get('name', '')}" if rid in analyzed else entry.get("name", ""), rid) for rid, entry in ordered]
        self.roll_combo.set_labeled_items(entries, selected, search_fn=lambda _label, item_id: self._name_for_id(item_id))
        self._update_roll_status_hint(active_id, selected)
        self._update_reanalyze_btn(active_id, selected)

    def _update_roll_status_hint(self, active_id: Optional[str], selected_id: str) -> None:
        """Flags a baseline picked from a roll other than the one loaded."""
        if active_id and selected_id and selected_id != active_id:
            set_hint_kind(self.roll_status_hint, "warning")
            self.roll_status_hint.setText(f"Using “{self._name_for_id(selected_id)}”, a baseline saved for a different roll")
        else:
            self.roll_status_hint.setText("")

    def _update_reanalyze_btn(self, active_id: Optional[str], selected_id: str) -> None:
        """Both ways of filling a baseline write onto the loaded roll, same as the
        Library's own "Batch Analysis" -- grayed out otherwise, with the same
        explanation."""
        is_active = bool(selected_id) and selected_id == active_id
        for btn, on, off in (
            (self.reanalyze_btn, BATCH_ANALYSIS_TOOLTIP, BATCH_ANALYSIS_DISABLED_TOOLTIP),
            (self.from_frame_btn, FROM_FRAME_TOOLTIP, FROM_FRAME_DISABLED_TOOLTIP),
        ):
            btn.setEnabled(is_active)
            btn.setToolTip(wrap_tooltip(on if is_active else off))

    def sync_ui(self) -> None:
        self.block_signals(True)
        try:
            self._refresh_rolls()
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        """
        Helper to block/unblock all buttons.
        """
        self.roll_combo.blockSignals(blocked)
