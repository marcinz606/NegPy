"""The per-folder record of Blend and Single Exposure captures."""

import pytest

from negpy.services.assets.capture_modes import BLEND, SINGLE, capture_modes, record_capture_mode


class _Repo:
    def __init__(self):
        self.settings: dict = {}
        self.saves = 0

    def get_global_setting(self, key, default=None):
        return self.settings.get(key, default)

    def save_global_setting(self, key, value):
        self.settings[key] = value
        self.saves += 1


def test_a_recorded_capture_reads_back_by_path():
    repo = _Repo()
    record_capture_mode(repo, "/scans/roll/Roll_Frame001.ARW", BLEND)
    record_capture_mode(repo, "/scans/roll/Roll_Frame002.ARW", SINGLE)
    record_capture_mode(repo, "/scans/other/Other_Frame001.ARW", SINGLE)
    paths = [
        "/scans/roll/Roll_Frame001.ARW",
        "/scans/roll/Roll_Frame002.ARW",
        "/scans/roll/Roll_Frame003.ARW",
        "/scans/other/Other_Frame001.ARW",
    ]
    assert capture_modes(repo, paths) == {
        "/scans/roll/Roll_Frame001.ARW": BLEND,
        "/scans/roll/Roll_Frame002.ARW": SINGLE,
        "/scans/other/Other_Frame001.ARW": SINGLE,
    }
    assert len(repo.settings) == 2  # one setting per folder


def test_a_retake_replaces_the_mode_and_an_unchanged_one_writes_nothing():
    repo = _Repo()
    record_capture_mode(repo, "/r/F1.ARW", SINGLE)
    record_capture_mode(repo, "/r/F1.ARW", SINGLE)
    assert repo.saves == 1
    record_capture_mode(repo, "/r/F1.ARW", BLEND)
    assert capture_modes(repo, ["/r/F1.ARW"]) == {"/r/F1.ARW": BLEND}


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        record_capture_mode(_Repo(), "/r/F1.ARW", "trichrome")


def test_a_fresh_blend_frame_opens_as_its_capture_imported_it():
    from negpy.desktop.session import resolve_asset_capture
    from negpy.features.process.models import ProcessMode, scan_setup_values
    from negpy.kernel.system.config import DEFAULT_WORKSPACE_CONFIG

    linear_raw, paper = scan_setup_values("camera", "narrowband")
    for mode in (BLEND, SINGLE):
        cfg = resolve_asset_capture(DEFAULT_WORKSPACE_CONFIG, {"capture": mode})
        assert cfg.process.process_mode == ProcessMode.C41
        assert cfg.process.linear_raw is linear_raw
        assert cfg.exposure.paper_profile == paper
    assert resolve_asset_capture(DEFAULT_WORKSPACE_CONFIG, {}) is DEFAULT_WORKSPACE_CONFIG
