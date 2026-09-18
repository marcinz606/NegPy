from typing import Optional

from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, section_subheader, set_hint_kind
from negpy.desktop.view.widgets.searchable_gear_combo import SearchableGearCombo
from negpy.services.assets import rolls

# Prefixes a roll's label once it has a saved Batch Analysis baseline -- the field
# itself is the "analyzed" indicator, so the search text ignores it (built from the
# plain name).
_TICK = "✓ "


class RollAnalysisSidebar(BaseSidebar):
    """
    Which roll's Batch Analysis baseline this frame's Use Luma/Color Average axes
    borrow: a searchable picker over every roll in your library, ticked once it has
    one. Picking a roll loads its baseline immediately -- there is no separate Apply.
    Reanalyze, beside the picker, runs Batch Analysis itself (the metering pass that
    fills the tick in) -- the same action the Library's own "Analyze Roll…" offers,
    reachable here too since this is where you notice a roll has never been measured.
    Enabled only for the loaded roll, since Batch Analysis measures the files
    currently open, not just whichever one this picker happens to show.
    """

    def _init_ui(self) -> None:
        self.layout.addWidget(section_subheader("Batch Analysis"))
        row = QHBoxLayout()
        self.roll_combo = SearchableGearCombo(placeholder="Search rolls…")
        self.roll_combo.setToolTip("Picking a roll loads its saved Batch Analysis baseline onto the loaded files.")
        row.addWidget(self.roll_combo, 1)
        self.reanalyze_btn = self._icon_action(
            "fa5s.play",
            "Analyze Roll — measures every loaded frame's exposure and saves the average as this roll's baseline",
        )
        row.addWidget(self.reanalyze_btn)
        self.layout.addLayout(row)
        # Lock Bounds is adopted into this same row (between the combo and Reanalyze) once
        # ControlsPanel wires it in -- see insert_lock_button.
        self._picker_row = row

        self.roll_status_hint = hint_label("", "muted")
        self.layout.addWidget(self.roll_status_hint)

        self._roll_sync_key = None
        self._refresh_rolls(force=True)
        self.layout.addStretch()

    def insert_lock_button(self, lock_bounds_btn) -> None:
        """Adopts ProcessSidebar's Lock Bounds toggle into this row, between the roll
        picker and Reanalyze. Lock Bounds is specifically about this frame's
        relationship to Batch Analysis, so it belongs beside the action it exempts
        the frame from -- not the Analysis Buffer row it used to share."""
        self._picker_row.insertWidget(1, lock_bounds_btn)

    def _connect_signals(self) -> None:
        self.roll_combo.selection_changed.connect(self._on_roll_picked)
        self.reanalyze_btn.clicked.connect(self.controller.request_batch_normalization)
        self.sync_ui()

    def _on_roll_picked(self, roll_id: str) -> None:
        """Picking a roll is the whole action: it loads that roll's saved baseline
        onto the currently loaded files. A no-op if it has never been analyzed."""
        if roll_id:
            self.controller.apply_normalization_roll(roll_id)
        active_id = self.controller.state.active_roll_id
        self._update_roll_status_hint(active_id, roll_id)
        self._update_reanalyze_btn(active_id, roll_id)

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
            self.roll_status_hint.setText(f'Using "{self._name_for_id(selected_id)}", a baseline saved for a different roll')
        else:
            self.roll_status_hint.setText("")

    def _update_reanalyze_btn(self, active_id: Optional[str], selected_id: str) -> None:
        """Reanalyze only ever measures the loaded roll, same as the Library's own
        "Analyze Roll…" -- grayed out otherwise, with the same explanation."""
        is_active = bool(selected_id) and selected_id == active_id
        self.reanalyze_btn.setEnabled(is_active)
        self.reanalyze_btn.setToolTip(
            "Analyze Roll — measures every loaded frame's exposure and saves the average as this roll's baseline"
            if is_active
            else "Open this roll first — Batch Analysis measures the files currently loaded."
        )

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
