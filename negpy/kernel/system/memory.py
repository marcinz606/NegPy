from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path


def _windows_available_memory() -> int | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return int(status.ullAvailPhys)
    except Exception:
        pass
    return None


def _linux_available_memory() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _parse_vm_stat(output: str) -> int | None:
    page_match = re.search(r"page size of (\d+) bytes", output)
    if page_match is None:
        return None
    counts: dict[str, int] = {}
    for line in output.splitlines():
        name, separator, value = line.partition(":")
        if not separator:
            continue
        try:
            counts[name] = int(value.strip().rstrip("."))
        except ValueError:
            continue
    pages = sum(counts.get(name, 0) for name in ("Pages free", "Pages inactive", "Pages speculative"))
    return pages * int(page_match.group(1)) if pages > 0 else None


def _macos_available_memory() -> int | None:
    try:
        result = subprocess.run(
            ["/usr/bin/vm_stat"],
            capture_output=True,
            check=False,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_vm_stat(result.stdout) if result.returncode == 0 else None


def _posix_available_memory() -> int | None:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def available_system_memory_bytes() -> int | None:
    """Physical memory available without paging, or None when the OS cannot report it."""
    if sys.platform == "win32":
        return _windows_available_memory()
    if sys.platform.startswith("linux"):
        available = _linux_available_memory()
        if available is not None:
            return available
    if sys.platform == "darwin":
        available = _macos_available_memory()
        if available is not None:
            return available
    return _posix_available_memory()
