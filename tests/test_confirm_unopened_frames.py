from unittest.mock import MagicMock

from negpy.desktop.controller import AppController
from negpy.desktop.session import AppState


def test_unopened_frames_are_counted_without_parsing_any_edit() -> None:
    controller = MagicMock()
    controller.state = AppState()
    controller.session.repo.saved_hashes.return_value = {"a"}
    controller._confirm_bulk_export.return_value = True

    assert AppController._confirm_unopened_frames(controller, [{"hash": "a"}, {"hash": "b"}]) is True

    controller.session.repo.load_file_settings_many.assert_not_called()
    assert "1 of 2 frames" in controller._confirm_bulk_export.call_args.args[0]
