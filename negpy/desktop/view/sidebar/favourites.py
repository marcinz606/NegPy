import qtawesome as qta
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from negpy.desktop.controller import AppController
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.slider_shortcut_groups import SLIDER_GROUPS
from negpy.desktop.view.slider_targets import SLIDER_ATTRS, slider_widget_map
from negpy.desktop.view.styles.templates import hint_label
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import hidden_by_gating
from negpy.desktop.view.widgets.favourites_dialog import FavouritesDialog
from negpy.desktop.view.widgets.sliders import clone_slider

_SETTING_KEY = "favourite_sliders"


def load_favourites(repo) -> list[str]:
    """Drop ids that no longer exist so a retired slider degrades quietly."""
    stored = repo.get_global_setting(_SETTING_KEY)
    if not isinstance(stored, list):
        return []
    return [slider_id for slider_id in stored if slider_id in SLIDER_ATTRS]


class FavouritesSidebar(BaseSidebar):
    """User-chosen sliders gathered in one tab. Each favourite is a *mirror* of the real
    control, not the control itself — a QWidget has one parent, so re-parenting would tear
    the slider out of its home section. The mirror forwards to the original, which keeps its
    existing binding, mode-gating and channel-retargeting intact."""

    SIDE_MARGIN = 5

    def __init__(self, controller: AppController, controls):
        self.controls = controls
        self._mirrors: list[tuple[object, object]] = []
        super().__init__(controller)

    def _init_ui(self) -> None:
        row = QHBoxLayout()
        self.edit_btn = QPushButton("  Edit Favourites")
        self.edit_btn.setIcon(qta.icon("fa5s.sliders-h", color=THEME.text_primary))
        self.edit_btn.setToolTip("Choose which sliders appear here, and in what order")
        self.edit_btn.clicked.connect(self._open_editor)
        row.addWidget(self.edit_btn)
        row.addStretch()
        self.layout.addLayout(row)

        self.empty_hint = hint_label("No favourites yet — use Edit Favourites to pick the sliders you reach for most.")
        self.empty_hint.setWordWrap(True)
        self.layout.addWidget(self.empty_hint)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(THEME.space_sm)
        self.layout.addWidget(self._container)
        self.layout.addStretch(1)

        self._rebuild()

    def _connect_signals(self) -> None:
        # Not controller.config_updated: sidebar syncing is debounced 150ms and this signal
        # fires at the end of it, so the originals already hold the fresh values.
        self.controls.modified_synced.connect(self.sync_ui)

    def _choices(self) -> list[tuple[str, str, str]]:
        widgets = slider_widget_map(self.controls)
        return [(group.id, group.category, widgets[group.id]().label.text()) for group in SLIDER_GROUPS]

    def _open_editor(self) -> None:
        repo = self.controller.session.repo
        dlg = FavouritesDialog(self, self._choices(), load_favourites(repo))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            repo.save_global_setting(_SETTING_KEY, dlg.selected_ids())
            self._rebuild()

    def _rebuild(self) -> None:
        for clone, _ in self._mirrors:
            clone.setParent(None)
            clone.deleteLater()
        self._mirrors.clear()

        widgets = slider_widget_map(self.controls)
        for slider_id in load_favourites(self.controller.session.repo):
            src = widgets[slider_id]()
            clone = clone_slider(src)
            clone.valueChanged.connect(lambda v, s=src: s.mirror_value(v, commit=False))
            clone.valueCommitted.connect(lambda v, s=src: s.mirror_value(v, commit=True))
            self._container_layout.addWidget(clone)
            self._mirrors.append((clone, src))

        self.empty_hint.setVisible(not self._mirrors)
        self.sync_ui()

    def sync_ui(self) -> None:
        for clone, src in self._mirrors:
            clone.setValue(src.value())
            # Only mode gating should hide a mirror; a collapsed section or an off-screen
            # tab must not. B&W hides the whole Colour section, so isHidden() alone misses it.
            clone.setVisible(not hidden_by_gating(src))
            clone.setEnabled(src.isEnabled())
