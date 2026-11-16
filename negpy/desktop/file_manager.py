"""The OS file manager: the Trash and showing a file in its folder, alike on every platform."""

from negpy.kernel.system.i18n import tr
import os
import subprocess
import sys

from PyQt6.QtCore import QFile, QStorageInfo, QUrl
from PyQt6.QtGui import QDesktopServices

from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

# GetDriveTypeW: only a fixed drive has a Recycle Bin. On any other drive the shell deletes for good.
_DRIVE_FIXED = 3


_BIN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\BitBucket\Volume"
_POLICY_KEY = r"Software\Microsoft\Windows\CurrentVersion\Policies\Explorer"


def _registry_dword(hive, key: str, name: str) -> int | None:
    import winreg

    try:
        with winreg.OpenKey(hive, key) as k:
            return int(winreg.QueryValueEx(k, name)[0])
    except OSError:
        return None


def _windows_bin_takes(path: str) -> bool:
    """The shell deletes for good, with no prompt, on a drive that is not fixed, a bin the user
    turned off, or a file larger than the bin. Each case keeps the file instead."""
    import ctypes
    import winreg

    root = os.path.splitdrive(os.path.abspath(path))[0]
    if not root or root.startswith("\\\\"):
        return False
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    if kernel32.GetDriveTypeW(root + "\\") != _DRIVE_FIXED:
        return False
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        if _registry_dword(hive, _POLICY_KEY, "NoRecycleFiles") == 1:
            return False
    volume = ctypes.create_unicode_buffer(64)
    if not kernel32.GetVolumeNameForVolumeMountPointW(root + "\\", volume, len(volume)):
        return False
    guid = volume.value.rstrip("\\").rsplit("\\", 1)[-1]
    key = rf"{_BIN_KEY}\{guid}"
    if _registry_dword(winreg.HKEY_CURRENT_USER, key, "NukeOnDelete") == 1:
        return False
    capacity_mb = _registry_dword(winreg.HKEY_CURRENT_USER, key, "MaxCapacity")
    return capacity_mb is None or os.path.getsize(path) <= capacity_mb * 1024 * 1024


def _linux_trash_error(path: str, qt_error: str) -> str:
    """Qt reports a stale errno when the drive's Trash folder belongs to another user, as on an
    NTFS or FAT drive mounted without uid=."""
    uid = os.getuid()
    trash = os.path.join(QStorageInfo(path).rootPath(), f".Trash-{uid}")
    if os.path.isdir(trash) and os.stat(trash).st_uid != uid:
        return tr("{trash} belongs to another user. Mount its drive with uid={uid}", trash=trash, uid=uid)
    return qt_error


def move_to_trash(path: str) -> str:
    """Empty when *path* went to the Trash, else why it stayed. Never deletes for good."""
    if sys.platform == "win32" and not _windows_bin_takes(path):
        return tr("the Recycle Bin would delete it for good")
    f = QFile(path)
    if f.moveToTrash():
        return ""
    return _linux_trash_error(path, f.errorString()) if sys.platform.startswith("linux") else f.errorString()


def _linux_show_items(path: str) -> bool:
    """The freedesktop FileManager1 call; False when no file manager on the session bus answers it."""
    try:
        from PyQt6.QtDBus import QDBusConnection, QDBusInterface, QDBusMessage
    except ImportError:
        return False
    bus = QDBusConnection.sessionBus()
    if not bus.isConnected():
        return False
    iface = QDBusInterface("org.freedesktop.FileManager1", "/org/freedesktop/FileManager1", "org.freedesktop.FileManager1", bus)
    if not iface.isValid():
        return False
    reply = iface.call("ShowItems", [QUrl.fromLocalFile(path).toString()], "")
    return reply.type() != QDBusMessage.MessageType.ErrorMessage


def show_in_folder(path: str) -> bool:
    """Open the file manager on *path*'s folder with the file selected. A Linux file manager
    without FileManager1 opens the folder only. False when the folder is gone."""
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        return False
    if not os.path.exists(path):
        return QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
            return True
        if sys.platform == "win32":
            # Explorer parses its own command line: the path is quoted after the comma, not with it.
            subprocess.Popen(f'explorer /select,"{os.path.normpath(path)}"')
            return True
        if _linux_show_items(path):
            return True
    except OSError as e:
        logger.warning("Could not show %s in its folder: %s", path, e)
    return QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
