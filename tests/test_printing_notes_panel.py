"""Export panel's Printing Notes section: one collapsible home for the canvas preview
and the export, and the preview toggle mirrors the controller without echoing back."""

from types import SimpleNamespace

from negpy.desktop.view.sidebar.export import ExportSidebar
from tests.conftest import FakeController, FakeRepo


def _sidebar() -> ExportSidebar:
    controller = FakeController(repo=FakeRepo())
    return ExportSidebar(controller)


def test_the_preview_and_the_export_share_one_section() -> None:
    sidebar = _sidebar()

    assert sidebar.printing_notes_section._title_text == "Printing Notes"
    content = sidebar.printing_notes_section.findChildren(type(sidebar.printing_notes_btn))
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
