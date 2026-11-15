import json
import os
from typing import List, Dict, Any, Optional
from negpy.kernel.system.config import APP_CONFIG

# Preset-level data that is not a config field. Reserved keys are stripped before a preset
# reaches WorkspaceConfig.from_flat_dict, which would otherwise warn about them as unknown.
PRESET_NOTES_KEY = "__notes"
_RESERVED_PRESET_KEYS = frozenset({PRESET_NOTES_KEY})


# A preset name is a filename. These characters, the control range and the traversal
# forms have to go before it reaches a path.
_INVALID_NAME_CHARS = '/\\:*?"<>|'


def sanitize_preset_name(name: str) -> str:
    """The usable part of a name: unusable characters become spaces, and leading or
    trailing dots go, so "../escaped" and ".." cannot address anything."""
    cleaned = "".join(" " if c in _INVALID_NAME_CHARS or ord(c) < 32 else c for c in name)
    return cleaned.strip().strip(".").strip()


def is_valid_preset_name(name: str) -> bool:
    """True when the name can be written as-is, with nothing silently rewritten."""
    return bool(name.strip()) and sanitize_preset_name(name) == name.strip()


def preset_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Just the config fields, without the preset's own bookkeeping."""
    return {k: v for k, v in data.items() if k not in _RESERVED_PRESET_KEYS}


def preset_notes(data: Dict[str, Any]) -> str:
    value = data.get(PRESET_NOTES_KEY)
    return str(value) if value else ""


def with_preset_notes(fields: Dict[str, Any], notes: str) -> Dict[str, Any]:
    """A preset payload ready to save: fields plus notes, which are dropped when empty."""
    out = preset_fields(fields)
    if notes.strip():
        out[PRESET_NOTES_KEY] = notes.strip()
    return out


class Presets:
    """
    JSON I/O for user presets.
    """

    # Subdirectory of presets_dir. Empty = the edit presets themselves.
    _SUBDIR = ""

    @classmethod
    def _dir(cls) -> str:
        return os.path.join(APP_CONFIG.presets_dir, cls._SUBDIR) if cls._SUBDIR else APP_CONFIG.presets_dir

    @classmethod
    def _path(cls, name: str) -> str:
        return os.path.join(cls._dir(), f"{name}.json")

    @classmethod
    def exists(cls, name: str) -> bool:
        """Case-folded: on macOS and Windows a differently-cased name is the same file."""
        return name.strip().casefold() in {n.casefold() for n in cls.list_presets()}

    @classmethod
    def rename_preset(cls, old: str, new: str) -> bool:
        """One os.replace. Writing the new file and deleting the old one would delete the
        file it just wrote whenever the two names differ only in case."""
        if not is_valid_preset_name(new):
            return False
        src = cls._path(old)
        if not os.path.isfile(src):
            return False
        os.replace(src, cls._path(new.strip()))
        return True

    @classmethod
    def save_preset(cls, name: str, settings: Dict[str, Any]) -> None:
        # Written through a temp file: notes save on every keystroke, and a crash
        # mid-write would otherwise leave a truncated preset behind.
        if not is_valid_preset_name(name):
            raise ValueError(f"Unusable preset name: {name!r}")
        os.makedirs(cls._dir(), exist_ok=True)
        filepath = cls._path(name.strip())
        tmp = filepath + ".tmp"
        with open(tmp, "w") as f_out:
            json.dump(settings, f_out, indent=4)
        os.replace(tmp, filepath)

    @classmethod
    def load_preset(cls, name: str) -> Optional[Dict[str, Any]]:
        filepath = cls._path(name)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, "r") as f_in:
                res = json.load(f_in)
        except (OSError, ValueError):
            return None
        return res if isinstance(res, dict) else None

    @classmethod
    def list_presets(cls) -> List[str]:
        directory = cls._dir()
        if not os.path.exists(directory):
            return []
        return [f[:-5] for f in os.listdir(directory) if f.endswith(".json")]

    @classmethod
    def delete_preset(cls, name: str) -> bool:
        filepath = cls._path(name)
        if not os.path.exists(filepath):
            return False
        os.remove(filepath)
        return True


class MetadataPresets(Presets):
    """Named sets of Metadata-panel values, in their own namespace so the edit
    preset list stays a list of looks."""

    _SUBDIR = "metadata"
