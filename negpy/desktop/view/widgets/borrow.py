"""Lend a widget to another place in the window and put it back exactly where it was.

Simple mode moves the everyday Frame cards onto one Edit page, and the Roll and Export pop-ups
show cards that live on a tab. Each needs the one real widget, since a card is wired to its
config section and a copy would be a second, unsynced control. ``borrow`` leaves a hidden
placeholder in the widget's slot, so the slot keeps its place among its neighbours, and returns
the call that swaps the widget back in.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtWidgets import QStackedWidget, QWidget


def _detach(widget: QWidget) -> None:
    """Take *widget* out of whatever layout holds it now, so it can be placed again cleanly."""
    parent = widget.parentWidget()
    if isinstance(parent, QStackedWidget):
        parent.removeWidget(widget)
    elif parent is not None and parent.layout() is not None:
        parent.layout().removeWidget(widget)


def borrow(widget: QWidget) -> Callable[[], None]:
    """Free *widget* to be added to another layout; the returned call puts it back.

    A page of a QStackedWidget keeps its index (the stack's page order is how its tabs find
    their pages), and the page that was in front stays in front. A widget in a box or grid
    layout keeps its position. The caller adds the widget wherever it is going next.
    """
    parent = widget.parentWidget()
    if parent is None:
        raise ValueError("borrow() needs a widget that sits in a parent")

    placeholder = QWidget(parent)
    # Explicitly hidden, so it takes no room and no layout shows it.
    placeholder.hide()

    if isinstance(parent, QStackedWidget):
        stack = parent
        front = stack.currentWidget()
        stack.insertWidget(stack.indexOf(widget), placeholder)
        stack.removeWidget(widget)
        stack.setCurrentWidget(placeholder if front is widget else front)

        def restore() -> None:
            _detach(widget)
            front_now = stack.currentWidget()
            stack.insertWidget(stack.indexOf(placeholder), widget)
            stack.removeWidget(placeholder)
            stack.setCurrentWidget(widget if front_now is placeholder else front_now)
            placeholder.deleteLater()

        return restore

    layout = parent.layout()
    if layout is None or layout.replaceWidget(widget, placeholder) is None:
        placeholder.deleteLater()
        raise ValueError(f"borrow() could not find {widget!r} in its parent's layout")

    def restore() -> None:
        _detach(widget)
        # replaceWidget searches nested layouts too, so the placeholder is found wherever it sits.
        layout.replaceWidget(placeholder, widget)
        placeholder.deleteLater()

    return restore
