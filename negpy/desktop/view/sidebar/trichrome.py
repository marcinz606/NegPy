from negpy.kernel.system.i18n import tr
import os

from PyQt6.QtWidgets import QHBoxLayout

from negpy.desktop.session import can_edit_triplet, is_plain_triplet, trichrome_merge_indices
from negpy.desktop.view.confirm import confirm_assembly_mode
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, wrap_tooltip
from negpy.desktop.view.widgets.rgb_triplet_dialog import open_triplet_dialog
from negpy.features.rgbscan.models import is_rgb_triplet


class TrichromeSidebar(BaseSidebar):
    """Trichrome Mode per roll, and the three exposures the open frame is assembled from."""

    def _init_ui(self) -> None:
        self.enable_btn = self._small_toggle(
            "mdi.google-circles-communities",
            tr("Trichrome Mode"),
            self.controller.rgb_scan_mode_for_roll(self.state.active_roll_id),
            wrap_tooltip(tr("Make each frame from three exposures (red, green and blue), a setting that each roll keeps.")),
            align_left=True,
        )
        self.merge_btn = self._icon_action(
            "fa5s.object-group", tr("Merge Exposures (Trichrome): join the three selected frames into one, finding red, green and blue.")
        )
        self.split_btn = self._icon_action("fa5s.object-ungroup", tr("Split Trichrome: turn this frame back into its three exposures."))
        self.edit_btn = self._icon_action("fa5s.pen", tr("Edit Trichrome…: pick this frame's three exposures by hand."))
        row = QHBoxLayout()
        row.addWidget(self.enable_btn, 1)
        row.addWidget(self.merge_btn)
        row.addWidget(self.split_btn)
        row.addWidget(self.edit_btn)
        self.layout.addLayout(row)

        self.hint = hint_label("")
        self.layout.addWidget(self.hint)

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(self._on_toggled)
        self.controller.rgb_scan_mode_changed.connect(self._follow_mode)
        self.edit_btn.clicked.connect(lambda: open_triplet_dialog(self, self.controller))
        self.merge_btn.clicked.connect(self.controller.request_trichrome_merge)
        self.split_btn.clicked.connect(self.controller.request_split_triplet)
        self.controller.session.files_changed.connect(self.sync_ui)
        self.controller.session.state_changed.connect(self._sync_buttons)
        self.sync_ui()

    def _on_toggled(self, checked: bool) -> None:
        """Turning the mode on regroups the loaded roll into triplets, which reads and
        thumbnails every frame again, so it asks first. With nothing loaded there is
        nothing to regroup and nothing to confirm."""
        loaded = len(self.state.uploaded_files)
        if checked and loaded and not confirm_assembly_mode(self, tr("Trichrome"), loaded):
            self._follow_mode(False)
            return
        self.controller.set_rgb_scan_mode(checked)

    def _follow_mode(self, enabled: bool) -> None:
        """Follow a mode change this card did not make. Signals are blocked because the
        controller has already applied it; letting toggled through would ask for it a
        second time and re-run discovery."""
        self.enable_btn.blockSignals(True)
        self.enable_btn.setChecked(enabled)
        self.enable_btn.blockSignals(False)

    def sync_ui(self) -> None:
        conf = self.state.config.rgbscan
        self.block_signals(True)
        try:
            self.enable_btn.setChecked(self.controller.rgb_scan_mode_for_roll(self.state.active_roll_id))
            self._sync_buttons()
            if is_rgb_triplet(conf):
                parts = [os.path.basename(p) for p in (self.state.current_file_path or "", conf.green_path, conf.blue_path) if p]
                self.hint.setText(tr("Assembled from {others} and {last}.", others=", ".join(parts[:-1]), last=parts[-1]))
            else:
                self.hint.setText(tr("This frame is one exposure, not a Trichrome triplet."))
        finally:
            self.block_signals(False)

    def _sync_buttons(self) -> None:
        """Runs on every selection change, so it reads only the loaded files."""
        files = self.state.uploaded_files
        idx = self.state.selected_file_idx
        self.edit_btn.setEnabled(0 <= idx < len(files) and can_edit_triplet(files[idx]))
        self.split_btn.setEnabled(0 <= idx < len(files) and is_plain_triplet(files[idx]))
        self.merge_btn.setEnabled(bool(trichrome_merge_indices(files, self.state.selected_indices)))

    def block_signals(self, blocked: bool) -> None:
        for w in (self.enable_btn, self.edit_btn, self.merge_btn, self.split_btn):
            w.blockSignals(blocked)
