"""Simple and Advanced modes, and the Roll and Export menus of the menu bar.

Advanced is NegPy as it always was: every tab of the controls panel. Simple keeps three tabs,
the way Lightroom keeps a few modules:

* **Library**
* **Edit**: one column of Filtration, Tone, Paper Response, Geometry (Crop and Straighten),
  Lab, Retouch and Finishing
* **Scan**

The Roll tab's cards move to the menu bar's **Roll** menu, one item per card, and each opens
the card in a pop-up. Export moves to an **Export** menu: Frame, Selected and Roll export with
the current settings, and Preferences… opens the whole Advanced Export panel in a pop-up.

Both menus are on the bar in either mode. On macOS they join the global menu bar; on Windows
and Linux the window gets a menu bar of its own, which also carries the mode switch.

A pop-up shows the real card, not a copy (see ``widgets.borrow``), so a change there is the
same edit the Roll tab would make, and the card goes back to its tab when the pop-up closes.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QKeySequence
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QMenu, QMenuBar, QPushButton, QScrollArea, QStackedWidget, QVBoxLayout, QWidget

from negpy.desktop.view.shortcut_registry import key_for
from negpy.desktop.view.styles.templates import pin_dialog_default
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.borrow import borrow
from negpy.desktop.view.widgets.collapsible import CollapsibleSection
from negpy.desktop.view.widgets.dialog_geometry import remember_dialog_geometry
from negpy.kernel.system.i18n import tr

if TYPE_CHECKING:
    from negpy.desktop.view.main_window import MainWindow

UI_MODE_SETTING = "ui_mode"
SIMPLE = "simple"
ADVANCED = "advanced"

# The Roll tab's cards, in the tab's own order: (ControlsPanel attribute, menu label).
ROLL_CARDS: tuple[tuple[str, str], ...] = (
    ("film_section", tr("Setup")),
    ("baseline_section", tr("Roll Analysis")),
    ("process_section", tr("Metering")),
    ("fade_section", tr("Fade Restoration")),
    ("demosaic_section", tr("Raw Decode")),
    ("optics_section", tr("Optics")),
    ("sensor_section", tr("Calibration")),
    ("assembly_section", tr("Frame Assembly")),
)

# Export menu: (scope, label). Picking one exports it and makes it what the Export button and
# the export key (Ctrl+E, ⌘E) export from then on, the way the button's own chevron does.
EXPORT_SCOPES: tuple[tuple[str, str], ...] = (
    ("current", tr("Frame")),
    ("selected", tr("Selected")),
    ("all", tr("Roll")),
)


def retired_by_mode(widget: QWidget) -> bool:
    """True when the film mode or the config hid *widget* (a card that does not apply), not
    when it is merely on a tab that has never been shown."""
    return widget.isHidden() and widget.testAttribute(Qt.WidgetAttribute.WA_WState_ExplicitShowHide)


class CardPopup(QDialog):
    """A non-modal pop-up that holds a card from the controls panel while it is open. The
    canvas stays live beside it, so a change shows on the frame as it is made."""

    def __init__(self, title: str, widget: QWidget, parent: QWidget, *, repo=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.resize(420, 560)
        self._widget = widget
        # A stack hides the pages behind the front one explicitly, so a page shows only if told to.
        from_stack = isinstance(widget.parentWidget(), QStackedWidget)
        self._restore: Optional[Callable[[], None]] = borrow(widget)
        self._was_expanded: Optional[bool] = None
        if isinstance(widget, CollapsibleSection) and widget.collapsible:
            self._was_expanded = widget.toggle_button.isChecked()
            widget.expand()

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(THEME.space_lg, THEME.space_lg, THEME.space_lg, THEME.space_lg)
        body_layout.setSpacing(THEME.space_lg)
        body_layout.addWidget(widget)
        body_layout.addStretch(1)
        if from_stack:
            widget.show()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        close_btn = QPushButton(tr("Close"))
        close_btn.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, THEME.space_md)
        layout.addWidget(scroll, 1)
        layout.addLayout(buttons)
        buttons.setContentsMargins(THEME.space_lg, 0, THEME.space_lg, 0)

        # No Enter target: Enter in a card's field must not close the pop-up around it.
        pin_dialog_default(None, close_btn, scope=self)

        self.finished.connect(self._give_back)
        remember_dialog_geometry(self, repo, "card_popup")

    def _give_back(self, *_args) -> None:
        if self._restore is None:
            return
        restore, self._restore = self._restore, None
        restore()
        if self._was_expanded is not None and isinstance(self._widget, CollapsibleSection):
            self._widget.set_expanded(self._was_expanded)
        self.deleteLater()

    def closeEvent(self, event) -> None:
        # A window close does not emit finished on its own for a dialog that was show()n.
        super().closeEvent(event)
        self._give_back()


class UiMode(QObject):
    """Owns the Simple/Advanced choice, the pop-ups and the Roll and Export menus."""

    changed = pyqtSignal(bool)  # True for Simple

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self._window = window
        self._repo = window.controller.session.repo
        self._popups: dict[str, CardPopup] = {}
        self._mode_actions: list[tuple[QAction, QAction]] = []
        self._keyed: list[tuple[QAction, str]] = []
        window.right_panel.popup_requested.connect(self.open_popup)
        self.set_simple(self._repo.get_global_setting(UI_MODE_SETTING, ADVANCED) == SIMPLE, persist=False)

    # --- mode -----------------------------------------------------------------------------

    @property
    def simple(self) -> bool:
        return self._window.right_panel.simple_mode

    def set_simple(self, simple: bool, persist: bool = True) -> None:
        self._window.right_panel.set_simple_mode(simple)
        if persist:
            self._repo.save_global_setting(UI_MODE_SETTING, SIMPLE if simple else ADVANCED)
        for simple_act, advanced_act in self._mode_actions:
            simple_act.setChecked(simple)
            advanced_act.setChecked(not simple)
        self.changed.emit(simple)

    def toggle(self) -> None:
        self.set_simple(not self.simple)

    # --- pop-ups --------------------------------------------------------------------------

    def open_popup(self, key: str) -> None:
        """ "roll:<section attr>" for a Roll card, "export" for Export Preferences."""
        existing = self._popups.get(key)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            return
        if key == "export":
            widget = self._window.right_panel.export_sidebar
            title = tr("Export Preferences")
        elif key.startswith("roll:"):
            attr = key.removeprefix("roll:")
            labels = dict(ROLL_CARDS)
            if attr not in labels:
                return
            widget = getattr(self._window.controls_panel, attr)
            if retired_by_mode(widget):
                self._window.controller.set_status(tr("{value} not available", value=labels[attr]), 1500, kind="warning")
                return
            title = tr("Roll: {value}", value=labels[attr])
        else:
            return
        popup = CardPopup(title, widget, self._window, repo=self._repo)
        popup.destroyed.connect(lambda _obj=None, k=key: self._popups.pop(k, None))
        popup.finished.connect(lambda _result=0, k=key: self._popups.pop(k, None))
        self._popups[key] = popup
        popup.show()

    def close_popups(self) -> None:
        """Hand every borrowed card back, before the window closes and takes the pop-ups with it."""
        for popup in list(self._popups.values()):
            popup.close()

    def open_export_preferences(self) -> None:
        self.open_popup("export")

    def export(self, scope: str) -> None:
        """Export *scope* and keep it as the choice the Export button and Ctrl+E use."""
        sidebar = self._window.right_panel.export_sidebar
        sidebar.set_export_scope(scope)
        sidebar.run_export(scope)

    # --- menus ----------------------------------------------------------------------------

    def _action(self, menu: QMenu, text: str, slot: Callable[[], None], action_id: str = "") -> QAction:
        action = QAction(text, menu)
        action.setMenuRole(QAction.MenuRole.NoRole)
        action.triggered.connect(lambda _checked=False, run=slot: run())
        if action_id:
            self._keyed.append((action, action_id))
        menu.addAction(action)
        return action

    def build_mode_menu(self, parent: QWidget) -> QMenu:
        """The Mode menu: Simple or Advanced, and which keyboard shortcut set is in use."""
        menu = QMenu(tr("Mode"), parent)
        self.build_mode_items(menu)
        menu.addSeparator()
        menu.addMenu(self.build_shortcut_set_menu(menu))
        return menu

    def build_mode_items(self, menu: QMenu) -> None:
        """Two radio items, Simple Mode and Advanced Mode."""
        group = QActionGroup(menu)
        group.setExclusive(True)
        # Not keyed: on macOS a menu key equivalent outranks the QShortcut, and this item only
        # turns Simple on, where the toggle_simple_mode key switches both ways.
        simple_act = self._action(menu, tr("Simple Mode"), lambda: self.set_simple(True))
        advanced_act = self._action(menu, tr("Advanced Mode"), lambda: self.set_simple(False))
        for act in (simple_act, advanced_act):
            act.setCheckable(True)
            group.addAction(act)
        simple_act.setChecked(self.simple)
        advanced_act.setChecked(not self.simple)
        self._mode_actions.append((simple_act, advanced_act))

    def build_shortcut_set_menu(self, parent: QWidget) -> QMenu:
        """Switch every key to NegPy's own set or the Lightroom one. Checked is the set in use;
        neither is checked once the user has customized keys of their own."""
        from negpy.desktop.view.shortcut_presets import PRESETS, matching_preset

        menu = QMenu(tr("Keyboard Shortcuts"), parent)
        group = QActionGroup(menu)
        group.setExclusive(False)
        actions = {}
        manager = self._window.shortcut_manager
        for name, (label, _overrides) in PRESETS.items():
            act = self._action(menu, tr("{value} Shortcuts", value=label), lambda n=name: self._use_shortcut_set(n))
            act.setCheckable(True)
            group.addAction(act)
            actions[name] = act

        def sync() -> None:
            current = matching_preset(manager.bindings)
            for name, act in actions.items():
                act.setChecked(name == current)

        menu.aboutToShow.connect(sync)
        menu.addSeparator()
        self._action(menu, tr("Customize Shortcuts…"), lambda: manager.open_editor(self._window))
        return menu

    def build_roll_menu(self, parent: QWidget) -> QMenu:
        menu = QMenu(tr("Roll"), parent)
        cards: list[tuple[QAction, str]] = []
        for attr, label in ROLL_CARDS:
            act = self._action(menu, label + "…", lambda a=attr: self.open_popup(f"roll:{a}"))
            cards.append((act, attr))
        menu.addSeparator()
        run = self._window.shortcut_manager.action_for
        self._action(menu, tr("Import Folder as a Roll…"), run("import_roll"), "import_roll")
        self._action(menu, tr("Run Roll Analysis"), run("roll_batch_analysis"), "roll_batch_analysis")
        self._action(menu, tr("Roll Settings…"), run("roll_settings"), "roll_settings")

        def sync() -> None:
            # A card the film mode retires (Fade Restoration off a slide, Raw Decode off a scan)
            # is grayed out, as it would be missing from the Roll tab.
            for act, attr in cards:
                act.setEnabled(not retired_by_mode(getattr(self._window.controls_panel, attr)))

        menu.aboutToShow.connect(sync)
        return menu

    def build_export_menu(self, parent: QWidget) -> QMenu:
        menu = QMenu(tr("Export"), parent)
        group = QActionGroup(menu)
        group.setExclusive(True)
        scope_actions: dict[str, QAction] = {}
        for scope, label in EXPORT_SCOPES:
            # Not keyed: Ctrl+E runs whichever of these is checked, through its window shortcut.
            act = self._action(menu, label, lambda s=scope: self.export(s))
            act.setCheckable(True)
            group.addAction(act)
            scope_actions[scope] = act
        menu.addSeparator()
        self._action(menu, tr("Preferences…"), self.open_export_preferences, "export_preferences")

        def sync() -> None:
            # The check, and on Windows and Linux the export key beside it, follow the Export
            # button's choice, which its own chevron can change too.
            chosen = self._window.right_panel.export_sidebar.export_scope
            key = key_for("export")
            native = QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText) if key else ""
            for scope, label in EXPORT_SCOPES:
                act = scope_actions[scope]
                act.setChecked(scope == chosen)
                if sys.platform != "darwin":
                    act.setText(f"{label}\t{native}" if scope == chosen and native else label)

        menu.aboutToShow.connect(sync)
        return menu

    def _use_shortcut_set(self, name: str) -> None:
        """Switch every key to the set *name*, after asking when that would drop keys the user
        set by hand."""
        from PyQt6.QtWidgets import QMessageBox

        from negpy.desktop.view.shortcut_presets import PRESETS, matching_preset, preset_bindings

        manager = self._window.shortcut_manager
        current = matching_preset(manager.bindings)
        if current == name:
            return
        if not current:
            answer = QMessageBox.question(
                self._window,
                tr("Keyboard Shortcuts"),
                tr("Replace your customized shortcuts with the {value} set?", value=PRESETS[name][0]),
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        manager.update_bindings(preset_bindings(name))
        self._window.controller.set_status(tr("{value} Shortcuts", value=PRESETS[name][0]), 2000)

    def sync_shortcuts(self) -> None:
        """Show each item's key, after a rebind too. On macOS a menu key equivalent outranks
        a QShortcut, so there only ⌘ combinations go on the item (see mac_menu_bar)."""
        mac = sys.platform == "darwin"
        for action, action_id in self._keyed:
            key = key_for(action_id)
            if mac and "Ctrl+" not in key:
                key = ""
            if mac:
                action.setShortcut(QKeySequence(key))
            else:
                # The window's QShortcut already runs the key; the menu only names it.
                action.setText(
                    action.text().split("\t")[0]
                    + (f"\t{QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)}" if key else "")
                )


def install_ui_mode(window: "MainWindow") -> UiMode:
    """The Mode, Roll and Export menus. Call after the shortcut manager and, on
    macOS, after the global menu bar exist."""
    mode = UiMode(window)
    mac_bar: Optional[QMenuBar] = getattr(window, "mac_menus", None)
    if mac_bar is not None:
        before = mac_bar.window_menu.menuAction()
        mac_bar.insertMenu(before, mode.build_mode_menu(mac_bar))
        mac_bar.insertMenu(before, mode.build_roll_menu(mac_bar))
        mac_bar.insertMenu(before, mode.build_export_menu(mac_bar))
    else:
        bar = window.menuBar()
        app_menu = bar.addMenu(tr("NegPy"))
        mode._action(app_menu, tr("Preferences…"), window.shortcut_manager.action_for("open_preferences"), "open_preferences")
        bar.addMenu(mode.build_mode_menu(bar))
        bar.addMenu(mode.build_roll_menu(bar))
        bar.addMenu(mode.build_export_menu(bar))
    mode.sync_shortcuts()
    return mode
