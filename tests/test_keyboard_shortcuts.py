from unittest.mock import MagicMock, call


def test_keep_and_reject_shortcuts_ignore_the_roll_preview() -> None:
    from negpy.desktop.view.keyboard_shortcuts import _toggle_contact_sheet_mark

    controller = MagicMock()
    window = MagicMock()
    window.central_workspace.is_showing_roll_preview.return_value = True

    _toggle_contact_sheet_mark(controller, window, "keeper")
    _toggle_contact_sheet_mark(controller, window, "excluded")

    controller.session.toggle_mark.assert_not_called()


def test_keep_and_reject_shortcuts_mark_the_loaded_asset() -> None:
    from negpy.desktop.view.keyboard_shortcuts import _toggle_contact_sheet_mark

    controller = MagicMock()
    window = MagicMock()
    window.central_workspace.is_showing_roll_preview.return_value = False

    _toggle_contact_sheet_mark(controller, window, "keeper")
    _toggle_contact_sheet_mark(controller, window, "excluded")

    assert controller.session.toggle_mark.call_args_list == [
        call("keeper"),
        call("excluded"),
    ]
