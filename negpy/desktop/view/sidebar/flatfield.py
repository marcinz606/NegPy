import os

import qtawesome as qta
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QPushButton,
)

from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import section_subheader
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import CompactSlider

_NONE_LABEL = "— None —"
_FILE_FILTER = "Reference images (*.dng *.tif *.tiff *.cr2 *.cr3 *.nef *.arw *.raf *.rw2 *.jpg *.jpeg *.png);;All files (*)"


class FlatFieldSidebar(BaseSidebar):
    """
    Flat-field / falloff correction. Manages named reference profiles (the bare
    light-source scan) and a per-image enable toggle.
    """

    def _init_ui(self) -> None:
        self.enable_btn = self._small_toggle(
            "fa5s.lightbulb",
            "Flatfield Correction",
            False,
            "Apply the active flat-field reference to this image",
        )
        self.layout.addWidget(self.enable_btn)

        self.layout.addWidget(section_subheader("REFERENCE PROFILE"))

        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip("Saved flat-field reference profiles (scan of the bare light source)")
        self.layout.addWidget(self.profile_combo)

        actions = QHBoxLayout()
        self.add_btn = QPushButton(" Add…")
        self.add_btn.setIcon(qta.icon("fa5s.plus", color=THEME.text_primary))
        self.add_btn.setToolTip("Pick a reference image and save it as a named profile")

        self.delete_btn = QPushButton(" Delete")
        self.delete_btn.setIcon(qta.icon("fa5s.trash", color=THEME.text_primary))
        self.delete_btn.setToolTip("Remove the selected profile")

        actions.addWidget(self.add_btn)
        actions.addWidget(self.delete_btn)
        self.layout.addLayout(actions)

        self.layout.addWidget(section_subheader("LENS DISTORTION"))
        self.k1_slider = CompactSlider("Distortion", -0.25, 0.25, self.state.config.flatfield.k1, step=0.005, has_neutral=True)
        self.k1_slider.setToolTip(
            "Radial lens-distortion correction for the copy-stand rig.\nSaved with the profile. Use the film rebate as a straight reference."
        )
        self.layout.addWidget(self.k1_slider)

        self.layout.addStretch()
        self._refresh_profiles()

    def _connect_signals(self) -> None:
        self.enable_btn.toggled.connect(self.controller.set_flatfield_enabled)
        self.profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        self.add_btn.clicked.connect(self._on_add)
        self.delete_btn.clicked.connect(self._on_delete)
        # Drag = live preview only; commit writes k1 to the profile (its real home).
        self.k1_slider.valueChanged.connect(
            lambda v: self.update_config_section("flatfield", render=True, persist=False, readback_metrics=False, k1=v)
        )
        self.k1_slider.valueChanged.connect(lambda _v: self.controller.show_rotation_guide())
        self.k1_slider.valueCommitted.connect(self.controller.set_flatfield_k1)
        self.sync_ui()

    def _refresh_profiles(self) -> None:
        # Preserve the caller's block state: unblocking here would let sync_ui's
        # setCurrentIndex re-fire _on_profile_selected and loop into update_config.
        prev = self.profile_combo.signalsBlocked()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem(_NONE_LABEL, "")
        for name in self.controller.session.repo.list_flatfield_profiles():
            self.profile_combo.addItem(name, name)
        self.profile_combo.blockSignals(prev)

    def _on_profile_selected(self, _idx: int) -> None:
        name = self.profile_combo.currentData() or ""
        active = self.controller.session.repo.get_global_setting("flatfield_active_profile") or ""
        if name == active:
            return
        self.controller.set_active_flatfield_profile(name)
        self.sync_ui()

    def _on_add(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select flat-field reference", "", _FILE_FILTER)
        if not path:
            return
        default_name = os.path.splitext(os.path.basename(path))[0]
        name, ok = QInputDialog.getText(self, "Save Flat-Field Profile", "Profile name:", text=default_name)
        if ok and name:
            self.controller.save_flatfield_profile(name, path)
            self._refresh_profiles()
            self.sync_ui()

    def _on_delete(self) -> None:
        name = self.profile_combo.currentData()
        if name:
            self.controller.delete_flatfield_profile(name)
            self._refresh_profiles()
            self.sync_ui()

    def sync_ui(self) -> None:
        conf = self.state.config.flatfield
        active = self.controller.session.repo.get_global_setting("flatfield_active_profile") or ""

        self.block_signals(True)
        try:
            self._refresh_profiles()
            idx = self.profile_combo.findData(active)
            self.profile_combo.setCurrentIndex(idx if idx >= 0 else 0)

            self.enable_btn.setChecked(conf.apply)
            self.enable_btn.setEnabled(bool(conf.reference_path))
            self.k1_slider.setValue(conf.k1)
            self.k1_slider.setEnabled(bool(conf.reference_path))
        finally:
            self.block_signals(False)

    def block_signals(self, blocked: bool) -> None:
        for w in (self.enable_btn, self.profile_combo, self.add_btn, self.delete_btn, self.k1_slider):
            w.blockSignals(blocked)
