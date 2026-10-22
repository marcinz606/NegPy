"""Export panel's Printing Notes: the canvas preview and the export share the Alongside
Exports card, and the preview toggle mirrors the controller without echoing back."""

from types import SimpleNamespace

from negpy.desktop.view.sidebar.export import ExportSidebar
from tests.conftest import FakeController, FakeRepo


def _sidebar() -> ExportSidebar:
    controller = FakeController(repo=FakeRepo())
    return ExportSidebar(controller)


def test_the_preview_and_the_export_share_one_card() -> None:
    sidebar = _sidebar()

    assert sidebar.alongside_section._title_text == "Alongside Exports"
    content = sidebar.alongside_section.findChildren(type(sidebar.printing_notes_btn))
    assert sidebar.printing_notes_btn in content
    assert sidebar.printing_notes_preview_btn in content


def test_the_preview_toggle_drives_the_controller() -> None:
    sidebar = _sidebar()

    sidebar.printing_notes_preview_btn.click()

    sidebar.controller.toggle_printing_notes.assert_called_once_with(force=True)


def test_the_export_button_asks_for_the_sheet() -> None:
    sidebar = _sidebar()

    sidebar.printing_notes_btn.click()

    sidebar.controller.request_printing_notes_export.assert_called_once()


def test_a_state_sync_does_not_echo_back_as_a_toggle() -> None:
    stub = SimpleNamespace(printing_notes_preview_btn=_sidebar().printing_notes_preview_btn)
    calls: list = []
    stub.printing_notes_preview_btn.toggled.connect(lambda checked: calls.append(checked))

    ExportSidebar._on_printing_notes_changed(stub, True)

    assert stub.printing_notes_preview_btn.isChecked()
    assert calls == []


def test_sync_ui_does_not_reset_cursor_in_focused_contact_sheet_path(qapp) -> None:
    """sync_ui() runs on every debounced AppState resync; its setText() on the Contact
    Sheet output path field must not steal the caret from a user mid-edit — see #1071,
    which was the same bug in the Filename field."""
    from PyQt6.QtTest import QTest

    sidebar = _sidebar()
    sidebar.show()
    QTest.qWaitForWindowActive(sidebar)

    sidebar.cs_output_path_edit.setFocus()
    qapp.processEvents()
    assert sidebar.cs_output_path_edit.hasFocus()

    sidebar.cs_output_path_edit.setText("prefix_suffix")
    sidebar.cs_output_path_edit.setCursorPosition(6)

    sidebar.sync_ui()

    assert sidebar.cs_output_path_edit.cursorPosition() == 6
    sidebar.hide()
