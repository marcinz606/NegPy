from types import SimpleNamespace
from unittest.mock import MagicMock

from negpy.desktop.view import main_window
from negpy.desktop.view.main_window import MainWindow


def _window():
    return SimpleNamespace(canvas=SimpleNamespace(hud=MagicMock()))


def test_a_failed_batch_lists_every_failed_file(monkeypatch):
    box_cls = MagicMock()
    monkeypatch.setattr(main_window, "QMessageBox", box_cls)
    errors = ["Export failed for a.tif: disk full", "Export failed for b.tif: disk full"]

    MainWindow._on_export_finished(_window(), 3.0, errors)

    assert box_cls.call_args.args[2] == "2 files could not be exported."
    box_cls.return_value.setDetailedText.assert_called_once_with("\n".join(errors))


def test_a_clean_batch_asks_nothing(monkeypatch):
    box_cls = MagicMock()
    monkeypatch.setattr(main_window, "QMessageBox", box_cls)
    win = _window()

    MainWindow._on_export_finished(win, 3.0, [])

    box_cls.assert_not_called()
    assert win.canvas.hud.showMessage.call_args.kwargs["kind"] == "info"


def test_a_clean_batch_names_its_folder(monkeypatch):
    monkeypatch.setattr(main_window, "QMessageBox", MagicMock())
    win = _window()

    MainWindow._on_export_finished(win, 3.0, [], "/scans/export")

    assert win.canvas.hud.showMessage.call_args.args[0] == "Exported to /scans/export in 3.00s"


def test_show_library_shows_the_tab_before_the_first_import():
    calls = []
    win = SimpleNamespace(
        controller=SimpleNamespace(has_rolls=lambda: False),
        drawer=SimpleNamespace(isVisible=lambda: True),
        right_panel=SimpleNamespace(
            show_tab_by_key=calls.append,
            library_tree=SimpleNamespace(prompt_import_folder=lambda: calls.append("import")),
        ),
    )

    MainWindow.show_library(win)

    assert calls == ["library", "import"]
