import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from negpy.kernel.system import updater
from negpy.kernel.system.updater import (
    APP_BUNDLE,
    APPIMAGE,
    NSIS,
    UpdateError,
    UpdateInfo,
    appimage_script,
    apply_update,
    download_asset,
    find_update,
    macos_script,
    nsis_script,
    select_asset,
)


def _asset(name: str, size: int = 10) -> dict:
    return {"name": name, "browser_download_url": f"https://example.test/{name}", "size": size}


ASSETS = [
    _asset("NegPy-0.9.5-x86_64.AppImage"),
    _asset("NegPy-0.9.5-Win64-Setup.exe"),
    _asset("NegPy-0.9.5-macOS-arm64.dmg"),
    _asset("NegPy-0.9.5-macOS-x86_64.dmg"),
]


# --- asset selection ------------------------------------------------------


def test_each_install_kind_picks_its_own_asset():
    assert select_asset(ASSETS, APPIMAGE)["name"].endswith(".AppImage")
    assert select_asset(ASSETS, NSIS)["name"].endswith("-Setup.exe")


@pytest.mark.parametrize("machine, expected", [("arm64", "arm64"), ("aarch64", "arm64"), ("x86_64", "x86_64")])
def test_the_mac_build_follows_the_machine_architecture(monkeypatch, machine, expected):
    monkeypatch.setattr("platform.machine", lambda: machine)

    assert select_asset(ASSETS, APP_BUNDLE)["name"] == f"NegPy-0.9.5-macOS-{expected}.dmg"


def test_a_single_dmg_release_serves_any_mac(monkeypatch):
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    only = [_asset("NegPy-0.9.5-macOS-x86_64.dmg")]

    assert select_asset(only, APP_BUNDLE) is only[0]


def test_no_asset_for_an_install_kind_that_cannot_update_itself():
    assert select_asset(ASSETS, "") is None
    assert select_asset([_asset("source.zip")], APPIMAGE) is None


# --- finding the update ---------------------------------------------------


def _release(tag: str = "v0.9.5", assets: list | None = None) -> dict:
    return {
        "tag_name": tag,
        "body": "## Notes",
        "html_url": "https://example.test/release",
        "assets": ASSETS if assets is None else assets,
    }


@pytest.fixture
def frozen_appimage(monkeypatch, tmp_path):
    """A running copy that looks like an installed AppImage."""
    target = tmp_path / "NegPy.AppImage"
    target.write_bytes(b"old")
    monkeypatch.setattr(updater.sys, "frozen", True, raising=False)
    monkeypatch.setattr(updater.sys, "platform", "linux")
    monkeypatch.setenv("APPIMAGE", str(target))
    return target


def test_a_newer_release_reports_the_matching_asset(monkeypatch, frozen_appimage):
    monkeypatch.setattr(updater, "get_app_version", lambda: "0.9.0")
    monkeypatch.setattr(updater, "fetch_latest_release", lambda timeout=5.0: _release())

    info = find_update()

    assert info.version == "0.9.5"
    assert info.asset_name == "NegPy-0.9.5-x86_64.AppImage"
    assert info.can_self_install


def test_the_current_release_is_no_update(monkeypatch, frozen_appimage):
    monkeypatch.setattr(updater, "get_app_version", lambda: "0.9.5")
    monkeypatch.setattr(updater, "fetch_latest_release", lambda timeout=5.0: _release())

    assert find_update() is None


def test_an_unknown_version_never_offers_an_update(monkeypatch):
    monkeypatch.setattr(updater, "get_app_version", lambda: "unknown")

    assert find_update() is None


def test_an_unreachable_github_is_no_update(monkeypatch):
    monkeypatch.setattr(updater, "get_app_version", lambda: "0.9.0")
    monkeypatch.setattr(updater, "fetch_latest_release", lambda timeout=5.0: None)

    assert find_update() is None


def test_a_release_without_a_build_for_this_install_still_points_at_the_page(monkeypatch, frozen_appimage):
    monkeypatch.setattr(updater, "get_app_version", lambda: "0.9.0")
    monkeypatch.setattr(updater, "fetch_latest_release", lambda timeout=5.0: _release(assets=[_asset("source.zip")]))

    info = find_update()

    assert info.version == "0.9.5"
    assert not info.can_self_install
    assert info.page_url == "https://example.test/release"


# --- downloading ----------------------------------------------------------


def _urlopen_serving(payload: bytes) -> MagicMock:
    response = MagicMock()
    response.headers = {"Content-Length": str(len(payload))}
    stream = io.BytesIO(payload)
    response.read.side_effect = lambda n: stream.read(n)
    response.__enter__.return_value = response
    return MagicMock(return_value=response)


def _info(size: int = 6) -> UpdateInfo:
    return UpdateInfo(
        version="0.9.5",
        notes="",
        page_url="https://example.test/release",
        asset_name="NegPy-0.9.5-x86_64.AppImage",
        download_url="https://example.test/asset",
        size=size,
    )


def test_the_asset_lands_whole_and_progress_is_reported(tmp_path):
    seen: list[tuple[int, int]] = []

    with patch("urllib.request.urlopen", _urlopen_serving(b"payload")):
        path = download_asset(_info(7), tmp_path, on_progress=lambda d, t: seen.append((d, t)))

    assert path == tmp_path / "NegPy-0.9.5-x86_64.AppImage"
    assert path.read_bytes() == b"payload"
    assert seen[-1] == (7, 7)


def test_a_cancelled_download_leaves_nothing_behind(tmp_path):
    with patch("urllib.request.urlopen", _urlopen_serving(b"payload")):
        with pytest.raises(UpdateError):
            download_asset(_info(), tmp_path, is_cancelled=lambda: True)

    assert list(tmp_path.iterdir()) == []


def test_a_failed_download_leaves_nothing_behind(tmp_path):
    with patch("urllib.request.urlopen", side_effect=OSError("connection reset")):
        with pytest.raises(UpdateError):
            download_asset(_info(), tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_a_release_with_no_asset_cannot_be_downloaded(tmp_path):
    info = UpdateInfo(version="0.9.5", notes="", page_url="https://example.test/release")

    with pytest.raises(UpdateError):
        download_asset(info, tmp_path)


# --- the swap scripts -----------------------------------------------------


def test_the_appimage_script_waits_then_replaces_and_relaunches():
    script = appimage_script(Path("/tmp/new.AppImage"), Path("/home/u/NegPy.AppImage"), 4242)

    assert "kill -0 4242" in script
    assert "mv -f /tmp/new.AppImage /home/u/NegPy.AppImage" in script
    assert script.rstrip().endswith("exec /home/u/NegPy.AppImage")


def test_the_posix_scripts_wait_on_the_pipe_not_only_the_pid():
    """A zombie still answers `kill -0`, so the pipe's EOF is what starts the swap."""
    script = appimage_script(Path("/tmp/new"), Path("/tmp/old"), 1)

    assert script.index("cat >/dev/null") < script.index("kill -0 1")
    assert "waited -lt 20" in script  # the PID poll is only a grace period


def test_the_windows_script_does_not_wait_on_a_stuck_pid_forever():
    assert "Wait-Process -Id 1 -Timeout 240" in nsis_script(Path("s.exe"), Path("d"), Path("e.exe"), 1)


def test_the_macos_script_mounts_swaps_the_bundle_and_reopens():
    script = macos_script(Path("/tmp/n.dmg"), Path("/Applications/NegPy.app"), 7, Path("/tmp/mnt"), Path("/tmp/NegPy.app"))

    assert "kill -0 7" in script
    assert "hdiutil attach /tmp/n.dmg" in script
    assert "ditto /tmp/mnt/NegPy.app /tmp/NegPy.app" in script
    assert "rm -rf /Applications/NegPy.app" in script
    assert "open /Applications/NegPy.app" in script


def test_the_windows_script_installs_silently_over_the_current_install():
    script = nsis_script(Path(r"C:\Temp\Setup.exe"), Path(r"C:\Program Files\NegPy"), Path(r"C:\Program Files\NegPy\NegPy.exe"), 9)

    assert "Wait-Process -Id 9" in script
    # NSIS wants /D last and unquoted, spaces and all.
    assert r'start "" /wait "C:\Temp\Setup.exe" /S /D=C:\Program Files\NegPy' in script
    # Through Explorer: the script is elevated, the app it restarts must not be.
    assert r'explorer.exe "C:\Program Files\NegPy\NegPy.exe"' in script


def test_the_windows_script_switches_cmd_to_utf8_before_the_paths():
    """cmd parses a batch file in the OEM codepage, not the UTF-8 it is written in. The
    staging path carries the profile name, so a non-ASCII user name garbles every path
    unless the script switches the codepage first."""
    script = nsis_script(
        Path("C:\\Users\\José\\Temp\\Setup.exe"), Path(r"C:\Program Files\NegPy"), Path(r"C:\Program Files\NegPy\NegPy.exe"), 3
    )

    assert "chcp 65001 >nul" in script
    assert script.index("chcp 65001") < script.index("José")


# --- applying -------------------------------------------------------------


def test_a_source_checkout_refuses_to_swap_itself(monkeypatch, tmp_path):
    monkeypatch.setattr(updater.sys, "frozen", False, raising=False)

    with pytest.raises(UpdateError):
        apply_update(tmp_path / "new.AppImage", _info())


def test_an_appimage_in_a_read_only_folder_says_so(monkeypatch, frozen_appimage, tmp_path):
    monkeypatch.setattr(updater.os, "access", lambda path, mode: False)

    with pytest.raises(UpdateError, match="not writable"):
        apply_update(tmp_path / "new.AppImage", _info())


def test_applying_spawns_the_script_detached_and_touches_nothing_yet(monkeypatch, frozen_appimage, tmp_path):
    spawned = {}
    monkeypatch.setattr(updater.subprocess, "Popen", lambda cmd, **kw: spawned.update(cmd=cmd, kw=kw))
    downloaded = tmp_path / "new.AppImage"
    downloaded.write_bytes(b"new")

    apply_update(downloaded, _info())

    assert spawned["kw"]["start_new_session"] is True
    assert Path(spawned["cmd"][1]).read_text().count(frozen_appimage.as_posix()) >= 1
    assert frozen_appimage.read_bytes() == b"old"  # the running copy survives until it exits


def test_the_appimage_stages_next_to_the_target(frozen_appimage):
    assert updater.staging_dir() == frozen_appimage.parent


def test_the_release_payload_shape_matches_what_github_sends():
    """Guards the field names find_update() reads out of the API response."""
    sample = json.loads(json.dumps(_release()))

    assert {"tag_name", "body", "html_url", "assets"} <= sample.keys()
    assert {"name", "browser_download_url", "size"} <= sample["assets"][0].keys()
