from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.view.widgets.granular_settings_dialog import open_paste_dialog
from negpy.domain.models import WorkspaceConfig


def _controller(clipboard, rows=None):
    state = SimpleNamespace(current_file_hash="h", clipboard=clipboard, clipboard_rows=rows)
    return SimpleNamespace(session=MagicMock(state=state), set_status=MagicMock())


def test_an_empty_clipboard_says_so():
    controller = _controller(None)
    open_paste_dialog(None, controller)
    controller.set_status.assert_called_once_with("Nothing to paste", 2000)
    controller.session.apply_pasted_fields.assert_not_called()


def test_a_card_copy_opens_the_picker_on_that_card_alone(monkeypatch):
    from types import SimpleNamespace as NS

    from negpy.desktop.view.widgets import granular_settings_dialog as gsd

    made = []

    class FakeDialog:
        def __init__(self, *_a, **kw):
            self.kw, self.limited = kw, None
            made.append(self)

        def limit_to_rows(self, ids):
            self.limited = list(ids)

        def exec(self):
            return gsd.QDialog.DialogCode.Accepted

        def selected(self):
            return ["picked"]

        def paste_bounds(self):
            return False

    monkeypatch.setattr(gsd, "GranularSettingsDialog", FakeDialog)
    rows = [NS(id="lab.saturation"), NS(id="lab.skin_protection")]
    controller = _controller(WorkspaceConfig(), rows)

    open_paste_dialog(None, controller)

    assert made[0].limited == ["lab.saturation", "lab.skin_protection"]
    assert made[0].kw["bounds_mode"] == ""
    controller.session.apply_pasted_fields.assert_called_once_with(["picked"], include_bounds=False)
