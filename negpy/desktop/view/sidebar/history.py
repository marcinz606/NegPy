from negpy.kernel.system.i18n import tr
import qtawesome as qta
from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from negpy.desktop.view.confirm import confirm_delete_named
from negpy.desktop.view.shortcut_registry import tooltip_with_shortcut
from negpy.desktop.view.sidebar.base import BaseSidebar
from negpy.desktop.view.styles.templates import hint_label, labeled_action
from negpy.desktop.view.styles.theme import THEME
from negpy.desktop.view.widgets.collapsible import make_section
from negpy.desktop.view.widgets.flat_rows import ICON, RowList, more_rect, paint_fill, paint_more, paint_pill
from negpy.kernel.system.config import APP_CONFIG

_INDEX_ROLE = Qt.ItemDataRole.UserRole
_DETAIL_ROLE = Qt.ItemDataRole.UserRole + 1
_NUMBER_ROLE = Qt.ItemDataRole.UserRole + 2
# "now", "undone" or "": the pill at a row's right end.
_TAG_ROLE = Qt.ItemDataRole.UserRole + 3

_ROW_HEIGHT = 30

_WORK_PRINT_TOOLTIP = tr("Named versions of this frame: click one to use it, or right-click to export, rename or delete it.")


class _RowDelegate(QStyledItemDelegate):
    """A flat row: a step number or an icon, the name and a muted detail, a pill, and ⋮ on hover.
    The current step takes the selected fill and a Now pill; an undone one is struck through."""

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), _ROW_HEIGHT)

    def paint(self, p: QPainter, option, index) -> None:
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect.adjusted(0, 1, 0, -1)
        tag = index.data(_TAG_ROLE) or ""
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        paint_fill(p, rect, current=tag == "now", hover=hover)

        font = QFont(option.font)
        font.setPixelSize(THEME.font_size_base)
        small = QFont(font)
        small.setPixelSize(THEME.font_size_small)
        x = rect.left() + THEME.space_lg
        number = index.data(_NUMBER_ROLE)
        if number is not None:
            width = QFontMetrics(small).horizontalAdvance("100")
            p.setFont(small)
            p.setPen(QColor(THEME.text_hint))
            p.drawText(QRect(x, rect.top(), width, rect.height()), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(number))
            x += width + THEME.space_lg
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if icon is not None:
            icon.paint(p, QRect(x, rect.center().y() - ICON // 2, ICON, ICON))
            x += ICON + THEME.space_lg

        # The ⋮ slot stays reserved, so a pill does not move when the row is hovered.
        right = more_rect(option.rect).left() - THEME.space_md
        if tag:
            text = tr("Now") if tag == "now" else tr("Undone")
            right = paint_pill(p, right, rect.center().y(), text, small, accent=tag == "now") - THEME.space_md

        name_font = QFont(font)
        name_font.setWeight(QFont.Weight(THEME.weight_semibold if tag == "now" else THEME.weight_regular))
        name_font.setStrikeOut(tag == "undone")
        name = QFontMetrics(name_font).elidedText(index.data(Qt.ItemDataRole.DisplayRole) or "", Qt.TextElideMode.ElideRight, right - x)
        p.setFont(name_font)
        p.setPen(QColor(THEME.text_hint if tag == "undone" else THEME.text_primary))
        p.drawText(QRect(x, rect.top(), right - x, rect.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)
        x += QFontMetrics(name_font).horizontalAdvance(name)
        detail = index.data(_DETAIL_ROLE)
        if detail and x < right:
            font.setStrikeOut(tag == "undone")
            p.setFont(font)
            p.setPen(QColor(THEME.text_hint))
            text = QFontMetrics(font).elidedText(f" · {detail}", Qt.TextElideMode.ElideRight, right - x)
            p.drawText(QRect(x, rect.top(), right - x, rect.height()), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
        if hover:
            paint_more(p, option.rect)
        p.restore()


class HistoryPanel(BaseSidebar):
    """Work prints (named versions, kept) above the edit-history steps (pruned), each a card."""

    def _init_ui(self) -> None:
        repo = self.controller.session.repo

        self.work_prints = RowList(_RowDelegate, _ROW_HEIGHT)
        self.work_prints.setToolTip(_WORK_PRINT_TOOLTIP)
        self.work_prints_hint = hint_label(tr("Keep this edit as a named version."))
        self.save_btn = labeled_action(
            "mdi6.bookmark-plus-outline",
            tr("Save Work Print…"),
            tooltip_with_shortcut(tr("Save Work Print: keep the current edit as a named version"), "save_work_print"),
        )
        # Global setting, default off.
        self.apply_metadata_btn = self._labeled_toggle(
            "",
            tr("Load Metadata"),
            self.controller.session.work_print_apply_metadata(),
            tr("Also restore the metadata saved with a work print when it loads."),
        )
        foot = QHBoxLayout()
        foot.setContentsMargins(0, 0, 0, 0)
        foot.addWidget(self.save_btn)
        foot.addStretch(1)
        foot.addWidget(self.apply_metadata_btn)
        prints_body = QWidget()
        prints_col = QVBoxLayout(prints_body)
        prints_col.setContentsMargins(0, 0, 0, 0)
        prints_col.setSpacing(THEME.space_sm)
        prints_col.addWidget(self.work_prints)
        prints_col.addWidget(self.work_prints_hint)
        prints_col.addLayout(foot)
        self.work_prints_section = make_section(
            repo, tr("Work Prints"), "work_prints", prints_body, "mdi6.bookmark-outline", default_expanded=True
        )

        self.list = RowList(_RowDelegate, _ROW_HEIGHT)
        self.list.setToolTip(tr("Click a step to jump to it (last 100 edits kept)."))
        history_body = QWidget()
        history_col = QVBoxLayout(history_body)
        history_col.setContentsMargins(0, 0, 0, 0)
        history_col.setSpacing(THEME.space_sm)
        history_col.addWidget(self.list)
        history_col.addWidget(
            hint_label(
                tr(
                    "Click a step to go back to it; ⋮ or right-click exports it. The last {count} edits are kept.",
                    count=APP_CONFIG.max_history_steps,
                )
            )
        )
        self.history_section = make_section(repo, tr("Edit History"), "edit_history", history_body, "fa5s.history", default_expanded=True)
        self.undo_btn = self.history_section.add_header_button("mdi.undo", tooltip_with_shortcut(tr("Undo"), "undo"))
        self.redo_btn = self.history_section.add_header_button("mdi.redo", tooltip_with_shortcut(tr("Redo"), "redo"))

        self.layout.addWidget(self.work_prints_section)
        self.layout.addWidget(self.history_section)
        self.layout.addStretch(1)

        self.refresh()

    def _connect_signals(self) -> None:
        self.list.itemClicked.connect(self._on_item_clicked)
        self.list.menu_requested.connect(self._on_context_menu)
        self.work_prints.itemClicked.connect(lambda item: self.controller.session.load_work_print(item.text()))
        self.work_prints.menu_requested.connect(self._on_work_print_menu)
        self.save_btn.clicked.connect(self.save_work_print)
        self.apply_metadata_btn.toggled.connect(self._on_apply_metadata_toggled)
        self.undo_btn.clicked.connect(self.controller.session.undo)
        self.redo_btn.clicked.connect(self.controller.session.redo)
        self.controller.session.history_changed.connect(self.refresh)
        self.controller.session.work_prints_changed.connect(self.refresh)
        self.controller.session.file_selected.connect(lambda _: self.refresh())

    def refresh(self) -> None:
        session = self.controller.session
        state = self.controller.state
        names = session.work_prints()
        self.work_prints.clear()
        icon = qta.icon("mdi6.bookmark-outline", color=THEME.text_secondary)
        for name in names:
            self.work_prints.addItem(QListWidgetItem(icon, name))
        self.work_prints.fit()
        self.work_prints.setVisible(bool(names))
        self.work_prints_hint.setVisible(not names)
        self.apply_metadata_btn.setVisible(bool(names))
        self.save_btn.setEnabled(bool(state.current_file_hash))

        self.list.clear()
        for row in reversed(self.controller.history_steps()):  # newest on top
            item = QListWidgetItem(row["name"])
            item.setData(_INDEX_ROLE, row["index"])
            item.setData(_NUMBER_ROLE, row["index"])
            item.setData(_DETAIL_ROLE, row["detail"])
            item.setData(_TAG_ROLE, "now" if row["is_current"] else "undone" if row["undone"] else "")
            self.list.addItem(item)
        self.list.fit()
        self.undo_btn.setEnabled(state.undo_index > 0)
        self.redo_btn.setEnabled(state.undo_index < state.max_history_index)

    def save_work_print(self) -> None:
        session = self.controller.session
        if not self.controller.state.current_file_hash:
            return
        name, ok = QInputDialog.getText(self, tr("Save Work Print"), tr("Name:"), text=session.next_work_print_name())
        name = name.strip()
        if not (ok and name):
            return
        if name in session.work_prints():
            replace = QMessageBox.question(self, tr("Replace Work Print"), tr("“{name}” already exists. Replace it?", name=name))
            if replace != QMessageBox.StandardButton.Yes:
                return
        session.save_work_print(name)

    def _on_apply_metadata_toggled(self, checked: bool) -> None:
        self.controller.session.set_work_print_apply_metadata(checked)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        self.controller.jump_to_history_step(item.data(_INDEX_ROLE))

    def _on_context_menu(self, pos) -> None:
        item = self.list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        # The current step is the After half, so comparing against it shows nothing.
        compare_action = None if item.data(_TAG_ROLE) == "now" else menu.addAction(tr("Compare With This Step"))
        export_action = menu.addAction(tr("Export This Version…"))
        chosen = menu.exec(self.list.mapToGlobal(pos))
        if chosen is export_action:
            self.controller.export_history_step(item.data(_INDEX_ROLE))
        elif chosen is not None and chosen is compare_action:
            self.controller.compare_with_history_step(item.data(_INDEX_ROLE))

    def _on_work_print_menu(self, pos) -> None:
        item = self.work_prints.itemAt(pos)
        if item is None:
            return
        name = item.text()
        menu = QMenu(self)
        compare_action = menu.addAction(tr("Compare With This Version"))
        export_action = menu.addAction(tr("Export This Version…"))
        rename_action = menu.addAction(tr("Rename…"))
        delete_action = menu.addAction(tr("Delete…"))
        chosen = menu.exec(self.work_prints.mapToGlobal(pos))
        if chosen is compare_action:
            self.controller.compare_with_work_print(name)
        elif chosen is export_action:
            self.controller.export_work_print(name)
        elif chosen is rename_action:
            new_name, ok = QInputDialog.getText(self, tr("Rename Work Print"), tr("Name:"), text=name)
            new_name = new_name.strip()
            if not (ok and new_name) or new_name == name:
                return
            exists = new_name in self.controller.session.work_prints()
            if (
                exists
                and QMessageBox.question(self, tr("Replace Work Print"), tr("“{new_name}” already exists. Replace it?", new_name=new_name))
                != QMessageBox.StandardButton.Yes
            ):
                return
            self.controller.session.rename_work_print(name, new_name, replace_existing=exists)
        elif chosen is delete_action:
            if confirm_delete_named(
                self,
                tr("Work Print"),
                name,
                informative=tr("Later edits never delete a work print; this is the only way to delete one."),
            ):
                self.controller.session.delete_work_print(name)
