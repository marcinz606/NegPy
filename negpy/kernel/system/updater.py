"""In-place self update.

The app finds the release asset that matches how *this* copy was installed,
downloads it, then hands a small detached script the job of swapping the
installation while NegPy is closed — a running program cannot overwrite itself.
The script waits for our PID to disappear, installs, and starts the new build.
"""

from __future__ import annotations

import os
import platform
import shlex
import stat
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.version import (
    RELEASES_PAGE,
    USER_AGENT,
    fetch_latest_release,
    get_app_version,
    is_newer,
)

logger = get_logger(__name__)

_CHUNK = 256 * 1024

# How the running copy was installed. "" means it cannot replace itself: a source
# checkout, or a bundle moved out of the layout its installer produced.
APPIMAGE = "appimage"
NSIS = "nsis"
APP_BUNDLE = "app_bundle"


class UpdateError(Exception):
    """A self update that cannot proceed, with a message fit for the user."""


@dataclass(frozen=True)
class UpdateInfo:
    """A newer release, and the asset that suits this install (if there is one)."""

    version: str
    notes: str
    page_url: str
    asset_name: str = ""
    download_url: str = ""
    size: int = 0

    @property
    def can_self_install(self) -> bool:
        return bool(self.download_url) and bool(install_kind())


def install_kind() -> str:
    """Which installer produced the running copy, or "" when it cannot self update."""
    if not getattr(sys, "frozen", False):
        return ""
    if sys.platform == "win32":
        return NSIS
    if sys.platform == "darwin":
        return APP_BUNDLE if app_bundle_path() is not None else ""
    if sys.platform.startswith("linux"):
        return APPIMAGE if os.environ.get("APPIMAGE") else ""
    return ""


def app_bundle_path() -> Optional[Path]:
    """The `.app` the running executable lives in, or None outside a bundle."""
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.suffix == ".app":
            return parent
    return None


def _macos_arch() -> str:
    machine = platform.machine()
    return "arm64" if machine in ("arm64", "aarch64") else "x86_64"


def select_asset(assets: list[dict[str, Any]], kind: str) -> Optional[dict[str, Any]]:
    """The release asset matching `kind`, or None when the release has no such build.

    macOS ships one DMG per architecture, so the arch suffix decides; a release with
    a single DMG is taken as-is, which keeps a one-arch release usable.
    """
    named = [a for a in assets if isinstance(a, dict) and a.get("name") and a.get("browser_download_url")]

    if kind == APPIMAGE:
        return next((a for a in named if a["name"].endswith(".AppImage")), None)
    if kind == NSIS:
        return next((a for a in named if a["name"].endswith("-Setup.exe")), None)
    if kind == APP_BUNDLE:
        dmgs = [a for a in named if a["name"].endswith(".dmg")]
        exact = [a for a in dmgs if a["name"].endswith(f"-{_macos_arch()}.dmg")]
        if exact:
            return exact[0]
        return dmgs[0] if len(dmgs) == 1 else None
    return None


def find_update(timeout: float = 5.0) -> Optional[UpdateInfo]:
    """The newest release when it is later than the running one, else None.

    An UpdateInfo without a `download_url` still gets reported: the release page is
    always an answer, even when nothing here can be installed automatically.
    """
    current = get_app_version()
    if current == "unknown":
        return None

    payload = fetch_latest_release(timeout)
    if not payload:
        return None

    latest = str(payload.get("tag_name") or "").lstrip("v")
    if not is_newer(latest, current):
        return None

    assets = payload.get("assets") or []
    asset = select_asset(assets if isinstance(assets, list) else [], install_kind())

    return UpdateInfo(
        version=latest,
        notes=str(payload.get("body") or ""),
        page_url=str(payload.get("html_url") or RELEASES_PAGE),
        asset_name=str(asset["name"]) if asset else "",
        download_url=str(asset["browser_download_url"]) if asset else "",
        size=int(asset.get("size") or 0) if asset else 0,
    )


def staging_dir() -> Path:
    """Where the downloaded asset lands.

    An AppImage is replaced by a rename, so it is staged beside the target to keep
    that rename on one filesystem; everything else goes to the temp directory.
    """
    running = os.environ.get("APPIMAGE")
    if install_kind() == APPIMAGE and running:
        target = Path(running).resolve().parent
        if os.access(target, os.W_OK):
            return target
    return Path(tempfile.mkdtemp(prefix="negpy-update-"))


def download_asset(
    info: UpdateInfo,
    dest_dir: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Stream the asset into `dest_dir` and return the file. Raises UpdateError."""
    if not info.download_url:
        raise UpdateError("This release has no download for your platform.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{info.asset_name}.part"
    request = urllib.request.Request(info.download_url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            total = int(response.headers.get("Content-Length") or info.size or 0)
            done = 0
            with open(target, "wb") as out:
                while True:
                    if is_cancelled is not None and is_cancelled():
                        raise UpdateError("Download cancelled.")
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(done, total)
    except UpdateError:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise UpdateError(f"Download failed: {exc}") from exc

    final = dest_dir / info.asset_name
    target.replace(final)
    return final


def _wait_and_run_sh(pid: int, body: str) -> str:
    """A script that starts once NegPy is gone.

    The read on stdin is the death signal: it returns EOF when our end of the pipe
    closes, which happens when the process dies however it dies. Polling the PID
    alone cannot do this — an exited-but-unreaped process is a zombie, and a zombie
    still answers `kill -0`. The poll that follows is only a short grace period,
    bounded because POSIX lets an open file be replaced under a running process.
    """
    return (
        "#!/bin/sh\n"
        "cat >/dev/null 2>&1\n"
        "waited=0\n"
        f"while kill -0 {pid} 2>/dev/null && [ $waited -lt 20 ]; do\n"
        "    sleep 0.5\n"
        "    waited=$((waited + 1))\n"
        "done\n"
        f"{body}"
    )


def appimage_script(downloaded: Path, target: Path, pid: int) -> str:
    """Replace the running AppImage with the downloaded one and start it."""
    new, old = shlex.quote(downloaded.as_posix()), shlex.quote(target.as_posix())
    return _wait_and_run_sh(
        pid,
        f'set -e\nchmod +x {new}\nmv -f {new} {old}\nrm -f "$0"\nexec {old}\n',
    )


def macos_script(dmg: Path, bundle: Path, pid: int, mountpoint: Path, stage: Path) -> str:
    """Mount the DMG, swap the bundle for the one inside it, and reopen the app.

    The new bundle is staged whole before the old one goes: a copy that dies
    half way must not take the installed app with it.
    """
    q = shlex.quote
    return _wait_and_run_sh(
        pid,
        f"""set -e
mkdir -p {q(mountpoint.as_posix())}
hdiutil attach {q(dmg.as_posix())} -nobrowse -readonly -mountpoint {q(mountpoint.as_posix())}
rm -rf {q(stage.as_posix())}
ditto {q((mountpoint / bundle.name).as_posix())} {q(stage.as_posix())}
hdiutil detach {q(mountpoint.as_posix())} -quiet || true
rm -rf {q(bundle.as_posix())}
mv {q(stage.as_posix())} {q(bundle.as_posix())}
xattr -dr com.apple.quarantine {q(bundle.as_posix())} 2>/dev/null || true
rm -f {q(dmg.as_posix())} "$0"
open {q(bundle.as_posix())}
""",
    )


def nsis_script(setup: Path, install_dir: Path, exe: Path, pid: int) -> str:
    """Run the installer silently over the current install, then restart NegPy.

    NSIS wants `/D` last and unquoted even when the path has spaces, which is why
    this is a batch file rather than an argument list. NegPy is restarted through
    Explorer: this script runs elevated, and a child of it would inherit the admin
    token that the app itself must not have.
    """
    return (
        "@echo off\r\n"
        # cmd parses a batch file in the console OEM codepage, not the UTF-8 it is written
        # in. The staging path carries the profile name, so a non-ASCII user name garbles
        # every path below unless the codepage switches first.
        "chcp 65001 >nul\r\n"
        f'powershell -NoProfile -Command "try {{ Wait-Process -Id {pid} -Timeout 240 }} catch {{ }}"\r\n'
        f'start "" /wait "{setup}" /S /D={install_dir}\r\n'
        f'del /q "{setup}"\r\n'
        f'explorer.exe "{exe}"\r\n'
        'del "%~f0"\r\n'
    )


# Holds each swap script's Popen, and with it the write end of the pipe the script waits
# on, open for the rest of this process's life. Dropping it would close the pipe early
# and start the swap under the running app.
_SPAWNED: list[subprocess.Popen] = []


def _spawn_posix(script: str, workdir: Path) -> None:
    path = workdir / "negpy-update.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    _SPAWNED.append(
        subprocess.Popen(
            ["/bin/sh", str(path)],
            start_new_session=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    )


def _spawn_elevated_bat(script: str, workdir: Path) -> None:
    """Launch the batch file through UAC — Program Files needs an admin token."""
    import ctypes

    path = workdir / "negpy-update.bat"
    path.write_text(script, encoding="utf-8")

    # SW_HIDE: the console window is noise, the installer runs silently anyway.
    result = ctypes.windll.shell32.ShellExecuteW(None, "runas", str(path), None, str(workdir), 0)
    if int(result) <= 32:
        raise UpdateError("Windows refused to start the installer (administrator approval is required).")


def apply_update(downloaded: Path, info: UpdateInfo) -> None:
    """Start the swap and return; the caller must then quit so it can finish.

    Nothing is replaced until this process exits, so a failure here leaves the
    current install untouched.
    """
    kind = install_kind()
    pid = os.getpid()

    if kind == APPIMAGE:
        target = Path(os.environ["APPIMAGE"]).resolve()
        if not os.access(target.parent, os.W_OK):
            raise UpdateError(f"{target.parent} is not writable — move NegPy somewhere you own, or install the new AppImage yourself.")
        _spawn_posix(appimage_script(downloaded, target, pid), downloaded.parent)

    elif kind == APP_BUNDLE:
        bundle = app_bundle_path()
        if bundle is None:
            raise UpdateError("NegPy is not running from an application bundle.")
        if not os.access(bundle.parent, os.W_OK):
            raise UpdateError(f"{bundle.parent} is not writable — install the update from the disk image yourself.")
        work = downloaded.parent
        _spawn_posix(macos_script(downloaded, bundle, pid, work / "mnt", work / bundle.name), work)

    elif kind == NSIS:
        exe = Path(sys.executable).resolve()
        _spawn_elevated_bat(nsis_script(downloaded, exe.parent, exe, pid), downloaded.parent)

    else:
        raise UpdateError("This copy of NegPy cannot update itself.")

    logger.info("Update to %s staged from %s", info.version, downloaded)
