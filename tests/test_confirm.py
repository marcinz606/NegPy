"""warn_targets_affect_unbaked: the Set-Targets change warning and its dismissal."""

from conftest import FakeRepo

import negpy.desktop.view.confirm as confirm


class _Icon:
    Information = "information"
    Warning = "warning"
    Question = "question"


class _ButtonRole:
    AcceptRole = "accept"
    RejectRole = "reject"


class _FakeMessageBox:
    """A stand-in QMessageBox that records how many times it was shown and drives the
    dismissal checkbox from a class flag, so the test needs no modal."""

    Icon = _Icon
    ButtonRole = _ButtonRole

    def __init__(self, parent=None):
        type(self).shown += 1

    @classmethod
    def reset(cls, *, check=False):
        cls.shown = 0
        cls.check = check

    def setIcon(self, *a):
        pass

    def setWindowTitle(self, *a):
        pass

    def setText(self, *a):
        pass

    def setInformativeText(self, *a):
        pass

    def setCheckBox(self, cb):
        cb.setChecked(type(self).check)

    def addButton(self, text, role):
        return text

    def setDefaultButton(self, *a):
        pass

    def exec(self):
        pass


def test_warn_shows_when_not_dismissed(qapp, monkeypatch):
    _FakeMessageBox.reset()
    monkeypatch.setattr(confirm, "QMessageBox", _FakeMessageBox)

    confirm.warn_targets_affect_unbaked(None, FakeRepo())

    assert _FakeMessageBox.shown == 1


def test_warn_skips_when_dismissed(qapp, monkeypatch):
    _FakeMessageBox.reset()
    monkeypatch.setattr(confirm, "QMessageBox", _FakeMessageBox)

    confirm.warn_targets_affect_unbaked(None, FakeRepo(exposure_targets_warn_dismissed=True))

    assert _FakeMessageBox.shown == 0


def test_warn_saves_dismiss_when_checked(qapp, monkeypatch):
    _FakeMessageBox.reset(check=True)
    monkeypatch.setattr(confirm, "QMessageBox", _FakeMessageBox)

    repo = FakeRepo()
    confirm.warn_targets_affect_unbaked(None, repo)

    assert repo.get_global_setting("exposure_targets_warn_dismissed", False) is True


def test_warn_does_not_save_dismiss_when_unchecked(qapp, monkeypatch):
    _FakeMessageBox.reset(check=False)
    monkeypatch.setattr(confirm, "QMessageBox", _FakeMessageBox)

    repo = FakeRepo()
    confirm.warn_targets_affect_unbaked(None, repo)

    assert "exposure_targets_warn_dismissed" not in repo.data
