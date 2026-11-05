"""How NegPy shot a single-file frame: a Blend or a Single Exposure capture.

Nothing in the RAW records it, so the capture writes it here, one global setting per folder
(file name -> mode), and the Film Strip reads it back for the frame's mark. A Trichrome frame
needs no entry: its three files make it a triplet.
"""

import os
from typing import Any, Dict, Iterable

BLEND = "blend"
SINGLE = "single"
MODES = (BLEND, SINGLE)

_KEY_PREFIX = "capture_modes:"


def _key(folder: str) -> str:
    return _KEY_PREFIX + os.path.normpath(folder)


def _folder_modes(repo: Any, folder: str) -> Dict[str, str]:
    saved = repo.get_global_setting(_key(folder), default=None)
    return dict(saved) if isinstance(saved, dict) else {}


def record_capture_mode(repo: Any, path: str, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown capture mode: {mode!r}")
    folder, name = os.path.split(path)
    modes = _folder_modes(repo, folder)
    if modes.get(name) != mode:
        modes[name] = mode
        repo.save_global_setting(_key(folder), modes)


def capture_modes(repo: Any, paths: Iterable[str]) -> Dict[str, str]:
    """{path: mode} for each of `paths` that a capture recorded."""
    by_folder: Dict[str, list] = {}
    for path in paths:
        by_folder.setdefault(os.path.dirname(path), []).append(path)
    found: Dict[str, str] = {}
    for folder, members in by_folder.items():
        modes = _folder_modes(repo, folder)
        for path in members:
            mode = modes.get(os.path.basename(path))
            if mode in MODES:
                found[path] = mode
    return found
