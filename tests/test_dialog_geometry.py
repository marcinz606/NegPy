"""Every resizable dialog opens at the size and position it had when it last closed.

The static walks keep a new dialog, or a new place that builds one, from shipping without the
store. The behavior tests cover each way a dialog hides.
"""

import ast
import re
from pathlib import Path

import pytest
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QHideEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog, QWidget

from negpy.desktop.view.widgets.dialog_geometry import _GeometryKeeper, remember_dialog_geometry
from tests.conftest import FakeController, FakeRepo, dialog_classes

NEGPY = Path(__file__).resolve().parents[1] / "negpy"
KEY = "dialog_geometry_probe"
# Windows that fix their own size: there is nothing for the user to resize.
FIXED_SIZE = {"ProgressDialog", "CommandPalette"}
# Callables that pass the store on to a dialog they build.
FORWARDERS = {"resolve_other_gear_pick", "GearItemsPanel", "GearPresetsPanel"}
# Calls on the dialog that set its default size or its window flags.
SIZING = {"resize", "setMinimumSize", "setMinimumWidth", "setWindowFlags", "float_over_app"}


def _init(cls: ast.ClassDef) -> ast.FunctionDef | None:
    return next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)


def _calls(node: ast.AST | None) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)] if node is not None else []


def _callee(call: ast.Call) -> str:
    func = call.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def _on_self(call: ast.Call) -> bool:
    """self.resize(...) and float_over_app(self): the call acts on the dialog itself."""
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else call.args[0] if call.args else None
    return isinstance(receiver, ast.Name) and receiver.id == "self"


def _remember_call(cls: ast.ClassDef) -> ast.Call | None:
    return next((c for c in _calls(_init(cls)) if _callee(c) == "remember_dialog_geometry"), None)


def test_every_resizable_dialog_remembers_its_geometry():
    dialogs = {cls.name: (path, cls) for path, cls in dialog_classes()}
    assert {"PreferencesDialog", "ShortcutEditorDialog", "CrosstalkEditorDialog", "RgbTripletDialog", "LiveViewWindow"} <= set(dialogs)
    assert FIXED_SIZE <= set(dialogs)
    missing = []
    for name, (path, cls) in dialogs.items():
        if name in FIXED_SIZE:
            fixed = any(_callee(c) in ("setFixedWidth", "setFixedSize") and _on_self(c) for c in _calls(cls))
            assert fixed, f"{name} no longer fixes its own size, so it needs remember_dialog_geometry"
        elif _remember_call(cls) is None:
            missing.append(f"{path.name}: {name}")
    assert not missing, f"dialogs that do not call remember_dialog_geometry in __init__: {missing}"


def test_dialog_geometry_names_are_unique_snake_case():
    names = []
    for _, cls in dialog_classes():
        call = _remember_call(cls)
        if call is None:
            continue
        first, name = call.args[0], call.args[-1]
        assert isinstance(first, ast.Name) and first.id == "self", cls.name
        assert isinstance(name, ast.Constant) and re.fullmatch(r"[a-z][a-z0-9_]*", str(name.value)), cls.name
        names.append(name.value)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"dialogs that share one stored geometry: {duplicates}"


def test_restore_runs_after_the_default_size():
    """A default resize() or a window-flag change after the restore overrides it."""
    early = []
    for path, cls in dialog_classes():
        call = _remember_call(cls)
        if call is None:
            continue
        sizing = [c.lineno for c in _calls(_init(cls)) if _callee(c) in SIZING and _on_self(c)]
        if sizing and max(sizing) > call.lineno:
            early.append(f"{path.name}: {cls.name}")
    assert not early, f"remember_dialog_geometry runs before the default size or flags: {early}"


def test_every_dialog_call_site_passes_the_store():
    """A dialog built without repo= keeps its default size and saves nothing."""
    takes_repo = set(FORWARDERS)
    for _, cls in dialog_classes():
        init = _init(cls)
        if init is not None and "repo" in {a.arg for a in init.args.kwonlyargs}:
            takes_repo.add(cls.name)
    seen, missing = set(), []
    for path in sorted(NEGPY.rglob("*.py")):
        for call in _calls(ast.parse(path.read_text())):
            name = _callee(call)
            if name not in takes_repo:
                continue
            seen.add(name)
            if not any(kw.arg == "repo" for kw in call.keywords):
                missing.append(f"{path.relative_to(NEGPY.parent)}:{call.lineno} {name}")
    assert not missing, f"built without repo=: {missing}"
    assert FORWARDERS <= seen, f"no call to {sorted(FORWARDERS - seen)}: update FORWARDERS"


@pytest.fixture
def host(qapp):
    widget = QWidget()
    widget.resize(700, 700)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()


def _dialog(repo, parent=None, cls=QDialog) -> QDialog:
    dlg = cls(parent)
    dlg.resize(300, 200)
    remember_dialog_geometry(dlg, repo, "probe")
    return dlg


def _shown(dlg: QDialog, qapp) -> QDialog:
    dlg.show()
    qapp.processEvents()
    return dlg


def test_size_and_position_survive_reopen(qapp, host):
    repo = FakeRepo()
    first = _shown(_dialog(repo, host), qapp)
    first.resize(480, 320)
    first.move(60, 70)
    qapp.processEvents()
    expected = first.geometry()
    first.close()

    second = _dialog(repo, host)
    assert second.geometry() == expected
    _shown(second, qapp)
    assert second.geometry() == expected
    assert second.size() == QSize(480, 320)
    second.close()


@pytest.mark.parametrize("stored", [None, "", "not valid geometry", "QUFBQQ==", 123, ["x"]])
def test_stored_garbage_keeps_the_default(qapp, stored):
    repo = FakeRepo(dialog_geometry_probe=stored)
    dlg = _shown(_dialog(repo), qapp)
    assert dlg.size() == QSize(300, 200)
    dlg.close()
    assert isinstance(repo.data[KEY], str) and repo.data[KEY] != stored


def test_without_a_store_nothing_is_wired(qapp):
    dlg = _shown(_dialog(None), qapp)
    assert dlg.findChildren(_GeometryKeeper) == []
    assert dlg.size() == QSize(300, 200)
    dlg.close()


class _Overriding(QDialog):
    """The override shapes in the tree: each does its own work, then calls super."""

    def accept(self) -> None:
        super().accept()

    def reject(self) -> None:
        super().reject()

    def done(self, a0: int) -> None:
        super().done(a0)

    def closeEvent(self, a0) -> None:  # noqa: N802
        super().closeEvent(a0)


CLOSE_PATHS = {
    "accept": QDialog.accept,
    "reject": QDialog.reject,
    "close": QDialog.close,
    "done": lambda dlg: dlg.done(2),
    "hide": QDialog.hide,
    "escape": lambda dlg: QTest.keyClick(dlg, Qt.Key.Key_Escape),
}


@pytest.mark.parametrize("cls", [QDialog, _Overriding])
@pytest.mark.parametrize("close", CLOSE_PATHS.values(), ids=CLOSE_PATHS.keys())
def test_every_close_path_saves(qapp, host, cls, close):
    repo = FakeRepo()
    dlg = _shown(_dialog(repo, host, cls), qapp)
    close(dlg)
    qapp.processEvents()
    assert not dlg.isVisible()
    assert KEY in repo.data


def test_minimize_does_not_save(qapp, host):
    repo = FakeRepo()
    dlg = _shown(_dialog(repo, host), qapp)
    dlg.showMinimized()
    qapp.processEvents()
    assert KEY not in repo.data
    dlg.close()
    qapp.processEvents()
    assert KEY not in repo.data


class _SpontaneousHide(QHideEvent):
    def spontaneous(self) -> bool:
        return True


def test_spontaneous_hide_is_ignored(qapp):
    repo = FakeRepo()
    dlg = _dialog(repo)
    keeper = dlg.findChild(_GeometryKeeper)
    keeper.eventFilter(dlg, _SpontaneousHide())
    assert KEY not in repo.data
    keeper.eventFilter(dlg, QHideEvent())
    assert KEY in repo.data


class _FailingRepo(FakeRepo):
    def save_global_setting(self, key, value):
        raise RuntimeError("disk full")


def test_a_failing_store_does_not_raise(qapp, caplog):
    """An exception inside a Qt event filter aborts the process."""
    dlg = _shown(_dialog(_FailingRepo()), qapp)
    dlg.close()
    qapp.processEvents()
    assert not dlg.isVisible()
    assert "Failed to persist dialog geometry" in caplog.text


def test_cached_window_keeps_its_size_across_hide_and_show(qapp):
    repo = FakeRepo()
    dlg = _shown(_dialog(repo), qapp)
    dlg.resize(500, 400)
    qapp.processEvents()
    dlg.hide()
    _shown(dlg, qapp)
    assert dlg.size() == QSize(500, 400)
    dlg.hide()
    assert _dialog(repo).size() == QSize(500, 400)


def _reopens_at_last_geometry(qapp, make, key: str) -> None:
    repo = FakeRepo()
    first = _shown(make(repo), qapp)
    first.resize(560, 420)
    first.move(30, 40)
    qapp.processEvents()
    expected = first.geometry()
    first.hide()
    assert key in repo.data
    second = make(repo)
    assert second.geometry() == expected
    _shown(second, qapp)
    assert second.geometry() == expected
    second.hide()


def test_rgb_triplet_dialog_reopens_at_its_last_geometry(qapp):
    from negpy.desktop.view.widgets.rgb_triplet_dialog import RgbTripletDialog

    _reopens_at_last_geometry(qapp, lambda repo: RgbTripletDialog(None, "", "", "", repo=repo), "dialog_geometry_rgb_triplet")


def test_live_view_window_reopens_at_its_last_geometry(qapp):
    """The Scanlight panel closes this window with a bare hide(), which finished never reports."""
    from negpy.desktop.view.sidebar.live_view_window import LiveViewWindow

    _reopens_at_last_geometry(qapp, lambda repo: LiveViewWindow(None, repo=repo), "dialog_geometry_live_view")


def test_preferences_dialog_reopens_at_its_last_geometry(qapp):
    from negpy.desktop.view.widgets.preferences_dialog import PreferencesDialog

    _reopens_at_last_geometry(
        qapp, lambda repo: PreferencesDialog(FakeController(repo), None, pinned_keys=set()), "dialog_geometry_preferences"
    )


def test_shortcut_editor_reopens_at_its_last_geometry(qapp):
    from negpy.desktop.view.shortcut_registry import default_bindings
    from negpy.desktop.view.widgets.shortcut_editor import ShortcutEditorDialog

    _reopens_at_last_geometry(
        qapp, lambda repo: ShortcutEditorDialog(default_bindings(), None, None, repo=repo), "dialog_geometry_shortcut_editor"
    )
