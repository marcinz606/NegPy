from typing import Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QCheckBox,
    QMenu,
    QPushButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from negpy.desktop.view.styles.templates import HEADER_BUTTON_SIZE, HEADER_HEIGHT, HEADER_ICON_SIZE, wrap_tooltip
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.sliders import align_slider_columns
import qtawesome as qta

# Said by every card's Roll half where no roll spans the loaded frames, so the disabled
# control carries the reason once rather than each sidebar wording it again.
NO_ROLL_SCOPE_HINT = "These frames are not one roll, so there is no roll to hold a shared value. Save as Roll to make one."


def roll_revert_icon(color: str) -> QIcon:
    """Reset to Roll's icon, shared by the section headers and the frame-wide menu items:
    the reset arrow at full height, a film frame cut into its lower right. The cut-out is
    painted in bg_header, the header's and the menu's background."""
    film_offset = (0.188, 0.2)
    return qta.icon(
        "fa5s.undo",
        "fa5s.circle",
        "fa5s.film",
        options=[
            {"color": color},
            {"color": THEME.bg_header, "scale_factor": 0.924, "offset": film_offset},
            {"color": color, "scale_factor": 0.66, "offset": film_offset},
        ],
    )


class CollapsibleSection(QWidget):
    """
    A simple collapsible container with a header button and configurable initial state.
    """

    reset_requested = pyqtSignal()
    roll_revert_requested = pyqtSignal()
    expanded_changed = pyqtSignal(bool)
    info_requested = pyqtSignal()
    selection_toggled = pyqtSignal(bool)
    scope_selected = pyqtSignal(str)  # "frame" | "roll"

    def __init__(
        self,
        title: str,
        expanded: bool = True,
        icon: Optional[QIcon] = None,
        background_widget: Optional[QWidget] = None,
        info: bool = False,
        select: bool = False,
        collapsible: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._title_text = title
        self.collapsible = collapsible
        if not collapsible:
            expanded = True

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.toggle_button = QPushButton()
        if collapsible:
            self.toggle_button.setCheckable(True)
            self.toggle_button.setChecked(expanded)
            self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.setFixedHeight(HEADER_HEIGHT)

        # Styled by the QPushButton#collapsible_header rules in modern_dark.qss. overlay="true"
        # means the header is stacked over a preview widget, on a translucent background.
        self.toggle_button.setObjectName("collapsible_header")
        self.toggle_button.setProperty("overlay", "true" if background_widget else "false")

        btn_layout = QHBoxLayout(self.toggle_button)
        btn_layout.setContentsMargins(THEME.space_xl, 8, THEME.space_xl, 8)
        btn_layout.setSpacing(10)
        # Kept typed (toggle_button.layout() widens to QLayout | None), for set_actions_menu
        # to insert into later.
        self._header_row = btn_layout

        # Nested in the header button like reset_btn, so ticking the section does not also
        # collapse it. Tristate is for display only: a click always resolves to all or none.
        self.select_box: Optional[QCheckBox] = None
        if select:
            self.select_box = QCheckBox()
            self.select_box.setTristate(True)
            self.select_box.setCursor(Qt.CursorShape.PointingHandCursor)
            self.select_box.setToolTip(f"Select every {title} setting")
            self.select_box.clicked.connect(self._on_select_clicked)
            btn_layout.addWidget(self.select_box)

        if icon:
            icon_label = QLabel()
            icon_label.setPixmap(icon.pixmap(14, 14))
            btn_layout.addWidget(icon_label)

        self.title_label = QLabel(self._title_text)
        self.title_label.setStyleSheet(
            f"font-weight: {THEME.weight_semibold}; font-size: {THEME.font_size_header}px; letter-spacing: 0.01em; background: transparent;"
        )
        btn_layout.addWidget(self.title_label)

        btn_layout.addStretch()

        self.info_btn: Optional[QPushButton] = None
        if info:
            # Nested in the header button, like reset_btn: it eats its own clicks, so opening the help
            # does not also collapse the section.
            self.info_btn = QPushButton()
            self.info_btn.setIcon(qta.icon("fa5s.info-circle", color=THEME.text_muted))
            self.info_btn.setFixedSize(HEADER_BUTTON_SIZE, HEADER_BUTTON_SIZE)
            self.info_btn.setIconSize(QSize(HEADER_ICON_SIZE, HEADER_ICON_SIZE))
            self.info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.info_btn.setToolTip(f"What am I looking at? — {title} guide")
            self.info_btn.setObjectName("collapsible_reset_btn")
            self.info_btn.clicked.connect(self.info_requested)
            btn_layout.addWidget(self.info_btn)

        self.reset_btn = QPushButton()
        self.reset_btn.setIcon(qta.icon("fa5s.undo", color=THEME.text_muted))
        self.reset_btn.setFixedSize(HEADER_BUTTON_SIZE, HEADER_BUTTON_SIZE)
        self.reset_btn.setIconSize(QSize(HEADER_ICON_SIZE, HEADER_ICON_SIZE))
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setToolTip(f"Reset {title} to defaults")
        self.reset_btn.setVisible(False)
        self.reset_btn.setObjectName("collapsible_reset_btn")
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        btn_layout.addWidget(self.reset_btn)

        # Hidden until set_roll_revert: most cards on most frames already follow the roll.
        self.roll_revert_btn = self._header_button(roll_revert_icon(THEME.text_muted), f"Reset {title} to the roll's settings")
        self.roll_revert_btn.setVisible(False)
        self.roll_revert_btn.clicked.connect(self.roll_revert_requested)
        btn_layout.addWidget(self.roll_revert_btn)
        self.roll_revert_available = False

        # Lazily built by set_actions_menu(): most sections have nothing that belongs
        # here, so no button exists until one asks for it.
        self.actions_btn: Optional[QPushButton] = None

        # Lazily built by set_scope_buttons(): a section owning no settings has no scope
        # to choose between.
        self.frame_btn: Optional[QPushButton] = None
        self.roll_btn: Optional[QPushButton] = None
        self._scope = "frame"
        self._scope_visible = False
        self.modified_count = 0

        self.chevron_label: Optional[QLabel] = None
        if collapsible:
            self.chevron_label = QLabel()
            self.chevron_label.setStyleSheet("background: transparent;")
            self._update_chevron(expanded)
            btn_layout.addWidget(self.chevron_label)

        if background_widget:
            background_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            header_container = QWidget()
            header_container.setFixedHeight(HEADER_HEIGHT)
            stacked = QStackedLayout(header_container)
            stacked.setStackingMode(QStackedLayout.StackingMode.StackAll)
            stacked.setContentsMargins(0, 0, 0, 0)
            stacked.addWidget(background_widget)
            stacked.addWidget(self.toggle_button)
            self.main_layout.addWidget(header_container)
        else:
            self.main_layout.addWidget(self.toggle_button)

        self.content_area = QFrame(self)
        self.content_area.setObjectName("collapsible_content")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(THEME.space_xl, 4, THEME.space_xl, 8)  # same inset as header
        self.content_layout.setSpacing(4)
        self.content_area.setVisible(expanded)

        self.main_layout.addWidget(self.content_area)

        if collapsible:
            self.toggle_button.toggled.connect(self._on_toggle)

    def _header_button(self, icon: QIcon, tooltip: str) -> QPushButton:
        """The header's own button size and look, the one reset and the scope pair use."""
        btn = QPushButton()
        btn.setIcon(icon)
        btn.setFixedSize(HEADER_BUTTON_SIZE, HEADER_BUTTON_SIZE)
        btn.setIconSize(QSize(HEADER_ICON_SIZE, HEADER_ICON_SIZE))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setObjectName("collapsible_reset_btn")
        btn.setToolTip(wrap_tooltip(tooltip))
        return btn

    def set_content(self, widget: QWidget) -> None:
        # Plain QWidget content is painted #0D0D0D by the global `QWidget {}` QSS rule, covering
        # the #121212 card frame, since custom subclasses are not auto-painted. The objectName
        # rule forces it transparent either way.
        widget.setObjectName("collapsible_content_body")
        self.content_layout.addWidget(widget)
        align_slider_columns(widget)

    def _update_chevron(self, expanded: bool) -> None:
        if self.chevron_label is None:
            return
        if expanded:
            self.chevron_label.setPixmap(qta.icon("fa5s.chevron-down", color=THEME.text_secondary).pixmap(12, 12))
        else:
            self.chevron_label.setPixmap(qta.icon("fa5s.chevron-right", color=THEME.text_secondary).pixmap(12, 12))

    def set_modified(self, count: int) -> None:
        """Append count to title when non-zero; show reset button. Unrelated to
        the scope pair's roll-override state -- keeping this title to only what it
        has always meant (how far from NegPy's own defaults) instead of chaining a
        second, unrelated fact onto the same "· count" reading."""
        self.modified_count = count
        visible = count > 0
        self.reset_btn.setVisible(visible)
        self.title_label.setText(f"{self._title_text} · {count}" if visible else self._title_text)
        self._refresh_scope_stripe()

    def set_roll_revert(self, available: bool) -> None:
        """Show Reset to Roll while this frame holds something other than the roll's value
        on this card."""
        self.roll_revert_available = available
        self.roll_revert_btn.setVisible(available)

    def set_selection_state(self, checked: int, total: int) -> None:
        """Reflect how many of the section's rows are ticked. Emits nothing."""
        if self.select_box is None:
            return
        if checked == 0:
            state = Qt.CheckState.Unchecked
        elif checked == total:
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
        self.select_box.blockSignals(True)
        self.select_box.setCheckState(state)
        self.select_box.blockSignals(False)

    def _on_select_clicked(self) -> None:
        if self.select_box is None:
            return
        want = self.select_box.checkState() != Qt.CheckState.Checked
        self.select_box.setCheckState(Qt.CheckState.Checked if want else Qt.CheckState.Unchecked)
        self.selection_toggled.emit(want)

    def _on_reset_clicked(self) -> None:
        self.reset_requested.emit()

    def _on_toggle(self, checked: bool) -> None:
        self.content_area.setVisible(checked)
        self._update_chevron(checked)
        self.expanded_changed.emit(checked)

    def expand(self) -> None:
        if self.collapsible and not self.toggle_button.isChecked():
            self.toggle_button.setChecked(True)

    def set_expanded(self, expanded: bool) -> None:
        if self.collapsible:
            self.toggle_button.setChecked(expanded)

    def add_header_toggle(self, icon_name: str, tooltip: str) -> QPushButton:
        btn = self._header_button(qta.icon(icon_name, color=THEME.text_muted, color_on=THEME.text_on_accent), tooltip)
        btn.setCheckable(True)
        self._header_row.insertWidget(self._header_row.count() - 1, btn)
        return btn

    def set_actions_menu(self, menu: QMenu, tooltip: str) -> None:
        """An always-visible header menu button, for section-level actions that reach
        past the section's own settings -- Film Strip's New Roll and its roll-wide
        reset, for one. reset_btn stays the affordance for resetting this section."""
        if self.actions_btn is None:
            self.actions_btn = QPushButton()
            self.actions_btn.setIcon(qta.icon("fa5s.ellipsis-v", color=THEME.text_muted))
            self.actions_btn.setFixedSize(HEADER_BUTTON_SIZE, HEADER_BUTTON_SIZE)
            self.actions_btn.setIconSize(QSize(HEADER_ICON_SIZE, HEADER_ICON_SIZE))
            self.actions_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.actions_btn.setObjectName("collapsible_reset_btn")
            self._header_row.insertWidget(self._header_row.count() - 1, self.actions_btn)
        self.actions_btn.setToolTip(tooltip)
        self.actions_btn.setMenu(menu)

    def set_scope_buttons(
        self, visible: bool, scope: str, roll_tooltip: str = "", frame_tooltip: str = "", roll_enabled: bool = True
    ) -> None:
        """The header's Frame/Roll pair: which scope this card's values live at, and the
        one click that moves them to the other. The active half is colored and checked,
        the other is the affordance; clicking the active one does nothing. A Roll-tab card
        reads its lock here, a frame card is always Frame and uses Roll as a one-shot
        push.

        `roll_enabled=False` keeps the pair readable where there is no roll to move values
        to: the frames are not one roll, which is a fact worth showing rather than hiding
        the pair and leaving the scope unsaid."""
        if self.frame_btn is None:
            self.frame_btn = self._build_scope_button("fa5s.image", "frame")
            self.roll_btn = self._build_scope_button("fa5s.film", "roll")
        self._scope = scope
        for btn, key, color in (
            (self.frame_btn, "frame", THEME.warn_amber),
            (self.roll_btn, "roll", THEME.accent_secondary),
        ):
            active = key == scope
            btn.setVisible(visible)
            btn.setChecked(active)
            btn.setIcon(qta.icon(btn.property("scope_icon"), color=color if active else THEME.text_muted))
        self.roll_btn.setEnabled(roll_enabled)
        self.frame_btn.setToolTip(frame_tooltip or f"{self._title_text} is this frame's own")
        self.roll_btn.setToolTip(roll_tooltip or f"Give the roll this frame's {self._title_text}…")
        self._scope_visible = visible
        self._refresh_scope_stripe()

    def _refresh_scope_stripe(self) -> None:
        """The header carries the lit button's own colour as a stripe (QSS [scope] rules),
        and only on a card holding something other than its defaults: a stripe down every
        untouched card says nothing. The body stays plain, like every other sidebar's."""
        striped = self._scope_visible and self.modified_count > 0
        self.toggle_button.setProperty("scope", self._scope if striped else "")
        style = self.toggle_button.style()
        style.unpolish(self.toggle_button)
        style.polish(self.toggle_button)

    def _build_scope_button(self, icon_name: str, scope: str) -> QPushButton:
        btn = QPushButton()
        btn.setCheckable(True)
        btn.setProperty("scope_icon", icon_name)
        btn.setFixedSize(HEADER_BUTTON_SIZE, HEADER_BUTTON_SIZE)
        btn.setIconSize(QSize(HEADER_ICON_SIZE, HEADER_ICON_SIZE))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setObjectName("collapsible_reset_btn")
        # Checked is display only: the pair is a readout as much as a control, so a click
        # on the half already active must not un-check it into a third, meaningless state.
        btn.clicked.connect(lambda: self._on_scope_clicked(scope))
        self._header_row.insertWidget(self._header_row.count() - 1, btn)
        return btn

    def _on_scope_clicked(self, scope: str) -> None:
        if scope == self._scope:
            self.frame_btn.setChecked(self._scope == "frame")
            self.roll_btn.setChecked(self._scope == "roll")
            return
        self.scope_selected.emit(scope)


def make_section(
    repo,
    title: str,
    key: str,
    content: QWidget,
    icon_name: str,
    default_expanded: bool = False,
    background_widget: Optional[QWidget] = None,
    collapsible: bool = True,
) -> CollapsibleSection:
    """The one way a sidebar builds a section: persisted under section_expanded_{key}, and the
    ⓘ guide present exactly when docs/USER_GUIDE.md carries a `panel:{key}` marker. The help
    dialog is parented to the section, so it centres on the window the section is in.
    collapsible=False always expands and never reads or writes the persisted setting; repo=None
    keeps the section but not its expanded state."""
    from negpy.desktop.view.widgets.section_help_dialog import SectionHelpDialog, has_guide

    persist = collapsible and repo is not None
    if collapsible:
        setting = f"section_expanded_{key}"
        persisted = repo.get_global_setting(setting) if persist else None
        expanded = default_expanded if persisted is None else bool(persisted)
    else:
        expanded = True
    section = CollapsibleSection(
        title,
        expanded=expanded,
        icon=qta.icon(icon_name, color=THEME.text_hint),
        background_widget=background_widget,
        info=has_guide(key),
        collapsible=collapsible,
    )
    section.set_content(content)
    if persist:
        section.expanded_changed.connect(lambda checked: repo.save_global_setting(setting, checked))
    if section.info_btn:
        section.info_requested.connect(lambda: SectionHelpDialog(key, title, section).exec())
    return section


def hidden_by_gating(widget: QWidget) -> bool:
    """True when the mode or config retired *widget*, as opposed to it being off-screen.

    Not isVisible(): a collapsed section, an off-screen tab and a closed dock all read as
    invisible. Gating hides the widget, a block in its sidebar or the whole section, never
    the section's content_area — the collapse flag — so the walk stops there.
    """
    section = widget.parentWidget()
    while section is not None and not isinstance(section, CollapsibleSection):
        section = section.parentWidget()
    if section is None:
        return False
    return section.isHidden() or not widget.isVisibleTo(section.content_area)
