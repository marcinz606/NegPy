from negpy.kernel.system.i18n import tr
from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.view.confirm import confirm_assembly_mode, confirm_undiptych
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, wrap_tooltip
from negpy.desktop.view.widgets.sliders import SliderGroup
from negpy.services.assets import rolls

_FOLLOWS_ROLLS_HINT = tr("Each of these frames follows its own roll's Half Frame. Open the roll to change it.")
_TOGGLE_TIP = tr("Split each scan into two frames, for half-frame cameras.")


class HalfFrameSidebar(BaseSidebar):
    """
    Half-frame splitting for this roll: whether each scan becomes two frames, and the
    split geometry for the odd frame the batch detection gets wrong. Each roll
    remembers its own state.
    """

    def _init_ui(self) -> None:
        self.enable_btn = self._small_toggle(
            "mdi.view-split-vertical",
            tr("Half Frame Mode"),
            self.controller.half_frame_mode_for_roll(self.state.active_roll_id),
            "",
            align_left=True,
        )
        self.layout.addWidget(self.enable_btn)

        self.adjust_btn = self._labeled_action(
            "mdi.tune-variant",
            tr("Adjust…"),
            wrap_tooltip(tr("Set this scan's crop and where it splits.")),
        )
        self.auto_btn = self._labeled_action(
            "fa5s.layer-group",
            tr("Detect All"),
            wrap_tooltip(tr("Find the crop and the split again on every loaded scan.")),
        )
        self.unsplit_btn = self._labeled_action(
            "fa5s.object-ungroup",
            tr("Unsplit"),
            wrap_tooltip(tr("Turn this diptych back into one plain scan, deleting both halves' edits")),
        )
        btn_row = QHBoxLayout()
        for btn in (self.adjust_btn, self.auto_btn, self.unsplit_btn):
            btn_row.addWidget(btn, 1)
        self.hint = hint_label("", "warning")
        self.layout.addWidget(SliderGroup(btn_row, self.hint))

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(self._on_toggled)
        self.controller.half_frame_mode_changed.connect(self._follow_mode)
        self.adjust_btn.clicked.connect(self._on_adjust)
        self.auto_btn.clicked.connect(self.controller.auto_detect_all_half_frame_splits)
        self.unsplit_btn.clicked.connect(self._on_unsplit)
        self.sync_ui()

    def _on_toggled(self, checked: bool) -> None:
        """No editor pops up: past the confirmation, turning it on splits every loaded
        scan at its auto-detected gutter directly, and the odd frame it gets wrong is
        fixed afterward with Adjust. With nothing loaded there is nothing to split and
        nothing to confirm."""
        loaded = len(self.state.uploaded_files)
        if checked and loaded and not confirm_assembly_mode(self, tr("Half Frame"), loaded):
            self._follow_mode(False)
            return
        self.controller.set_half_frame_mode(checked)
        if checked and loaded:
            self.controller.auto_detect_all_half_frame_splits()

    def _follow_mode(self, enabled: bool) -> None:
        """Follow the active roll's own toggle state. Signals are blocked because
        request_asset_discovery already applied it for this roll; letting toggled
        through would ask for it a second time and re-run discovery."""
        self.enable_btn.blockSignals(True)
        self.enable_btn.setChecked(enabled)
        self.enable_btn.blockSignals(False)
        self.sync_ui()

    def _on_adjust(self) -> None:
        """Open the half-frame rectangle editor on the current image; its own
        Apply ▾ picks what the result gets saved to."""
        path, file_hash = self.controller.current_base_file()
        if not path or not file_hash:
            return
        result = self.controller.open_half_frame_dialog(path, file_hash, selected_hashes=self.controller.selected_base_hashes())
        if result is not None:
            self.controller.reload_after_half_frame_change()

    def _on_unsplit(self) -> None:
        if confirm_undiptych(self):
            self.controller.request_undiptych()

    def _follows_rolls(self) -> bool:
        """With no roll open, whether a loaded frame belongs to a roll: it then splits by that
        roll's toggle, which this one cannot change. Cached on the loaded paths and the rolls."""
        if self.state.active_roll_id:
            return False
        repo = self.controller.session.repo
        key = (tuple(f.get("path", "") for f in self.state.uploaded_files), rolls.saved_rolls(repo))
        if getattr(self, "_follows_key", None) != key:
            paths = key[0]
            self._follows_key = key
            self._follows = any(rolls.rolls_containing_path(repo, p) for p in dict.fromkeys(paths) if p)
        return self._follows

    def sync_ui(self) -> None:
        """Each roll keeps its own toggle; with no roll open it is this session's choice for
        frames in no roll. A frame from a roll follows that roll, so with one loaded and no
        roll open the toggle grays out under a hint."""
        usable = not self._follows_rolls()
        has_files = bool(self.state.uploaded_files)
        self.block_signals(True)
        try:
            self.enable_btn.setChecked(self.controller.half_frame_mode_for_roll(self.state.active_roll_id))
            self.enable_btn.setEnabled(usable)
            self.enable_btn.setToolTip(wrap_tooltip(_TOGGLE_TIP if usable else _FOLLOWS_ROLLS_HINT))
            self.adjust_btn.setEnabled(usable and has_files)
            self.auto_btn.setEnabled(usable and has_files)
            self.unsplit_btn.setEnabled(self.controller.active_diptych() is not None)
            self.hint.setText("" if usable else _FOLLOWS_ROLLS_HINT)
            self.hint.setVisible(not usable)
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.enable_btn, self.adjust_btn, self.auto_btn, self.unsplit_btn):
            w.blockSignals(blocked)
