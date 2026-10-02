"""The output both scanners share: where frames go, and which roll they make."""

import os

from negpy.desktop.view.sidebar.scan_output import SETTINGS_KEY, ScanOutputPanel, load_scan_output_settings


class _Repo:
    def __init__(self, store=None):
        self.store = dict(store or {})

    def get_global_setting(self, key, default=None):
        return self.store.get(key, default)

    def save_global_setting(self, key, value):
        self.store[key] = value


def test_first_run_takes_the_folder_and_roll_the_scanners_kept_and_saves_them():
    repo = _Repo({"scanlight_settings": {"output_folder": "/cam", "roll_name": "Portra"}, "scanner_settings": {"output_folder": "/scan"}})
    settings = load_scan_output_settings(repo)
    assert (settings.output_folder, settings.roll_name) == ("/cam", "Portra")
    assert repo.store[SETTINGS_KEY]["output_folder"] == "/cam"


def test_the_film_scanner_folder_is_used_when_the_camera_kept_none():
    repo = _Repo({"scanner_settings": {"output_folder": "/scan"}})
    assert load_scan_output_settings(repo).output_folder == "/scan"


def test_folder_as_roll_writes_into_the_folder_and_names_the_roll_after_it(qapp):
    panel = ScanOutputPanel(_Repo())
    panel.folder_edit.setText("/scans/Portra 400/")
    assert panel.folder_is_roll()
    assert panel.target_folder() == "/scans/Portra 400/"
    assert panel.roll_name() == "Portra 400"
    assert not panel.roll_edit.isEnabled()


def test_a_roll_subfolder_takes_the_roll_name_and_refuses_an_unsafe_one(qapp):
    panel = ScanOutputPanel(_Repo())
    panel.folder_edit.setText("/scans")
    panel.folder_roll_btn.setChecked(False)
    panel.roll_edit.setText("Roll 7")
    assert panel.roll_edit.isEnabled()
    assert panel.target_folder() == os.path.join("/scans", "Roll 7")
    panel.roll_edit.setText("../out")
    assert panel.roll_name() is None and panel.target_folder() is None


def test_edits_persist(qapp):
    repo = _Repo()
    panel = ScanOutputPanel(repo)
    panel.as_roll_btn.setChecked(False)
    assert repo.store[SETTINGS_KEY]["scan_as_roll"] is False
    assert ScanOutputPanel(repo).as_roll() is False
