import os
import uuid
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from negpy.features.flatfield.logic import compute_gain, gain_token
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_EXT = ".npz"


class FlatFieldProfile(NamedTuple):
    id: str
    name: str
    k1: float
    source: str  # provenance path of the reference the gain was baked from


class FlatFieldProfiles:
    """
    npz I/O for flat-field reference profiles (illumination-falloff gain maps).

    One file per profile in ``APP_CONFIG.flatfield_dir``, named ``<uuid>.npz`` and
    holding the baked per-channel gain map plus rig metadata: the distortion ``k1``,
    a display name and the provenance path. The reference image is decoded once, at
    save time; nothing outside this directory is needed to apply the correction
    afterward, so moving or deleting the original reference is harmless.

    Keyed by an opaque id rather than the name — the per-image edit references a
    profile by id, so a rename (or a name collision across machines) never breaks
    the reference. Disk I/O happens on dropdown build, selection and save, never per
    render (the render path resolves gains through the logic-layer provider cache).
    """

    @staticmethod
    def _path_for_id(profile_id: str) -> str:
        return os.path.join(APP_CONFIG.flatfield_dir, f"{profile_id}{_EXT}")

    @staticmethod
    def _bake_gain(reference_path: str) -> Optional[np.ndarray]:
        """Decode a reference like a negative (no WB, linear) and compute its gain map."""
        if not reference_path or not os.path.exists(reference_path):
            return None
        try:
            from negpy.services.rendering.preview_manager import PreviewManager

            reference, _, _ = PreviewManager().load_linear_preview(reference_path, use_camera_wb=False, full_resolution=False)
            return compute_gain(reference)
        except Exception:
            logger.exception("Flat-field: failed to decode reference %s", reference_path)
            return None

    @staticmethod
    def _write(profile_id: str, gain: np.ndarray, *, name: str, k1: float, source: str) -> None:
        os.makedirs(APP_CONFIG.flatfield_dir, exist_ok=True)
        np.savez_compressed(
            FlatFieldProfiles._path_for_id(profile_id),
            gain=gain.astype(np.float32),
            token=gain_token(gain),
            name=name,
            k1=float(k1),
            source=source,
        )

    @staticmethod
    def _read(profile_id: str, keys: Tuple[str, ...]) -> Optional[Dict[str, np.ndarray]]:
        """Read only the named members of a profile file (a zip — untouched members are
        never decompressed, so metadata reads skip the ~MB gain array), or None if absent."""
        path = FlatFieldProfiles._path_for_id(profile_id)
        if not profile_id or not os.path.exists(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return {k: data[k] for k in keys if k in data.files}
        except Exception:
            logger.exception("Flat-field: failed to read profile %s", profile_id)
            return None

    @staticmethod
    def create(name: str, reference_path: str, k1: float = 0.0) -> Optional[str]:
        """Bake a reference into a new profile; returns its id (None if the decode failed)."""
        gain = FlatFieldProfiles._bake_gain(reference_path)
        if gain is None:
            return None
        return FlatFieldProfiles.import_gain(gain, name=name, k1=k1, source=reference_path)

    @staticmethod
    def import_gain(gain: np.ndarray, *, name: str, k1: float = 0.0, source: str = "") -> str:
        """Write an already-baked gain map as a new profile; returns its id."""
        profile_id = uuid.uuid4().hex
        FlatFieldProfiles._write(profile_id, gain, name=name, k1=k1, source=source)
        return profile_id

    @staticmethod
    def load_gain(profile_id: str) -> Optional[Tuple[np.ndarray, str]]:
        """(gain map, content token) for the render-path provider; None if missing/unreadable."""
        data = FlatFieldProfiles._read(profile_id, ("gain", "token"))
        if data is None or "gain" not in data:
            return None
        gain = np.ascontiguousarray(data["gain"], dtype=np.float32)
        token = str(data["token"]) if "token" in data else gain_token(gain)
        return gain, token

    @staticmethod
    def get(profile_id: str) -> Optional[FlatFieldProfile]:
        """Profile metadata (name, k1, source), or None if the file is gone.

        Metadata-only read — the gain array is not decompressed, so this stays cheap
        on the hot path (the sidebar rebuilds the profile list on every config sync).
        """
        data = FlatFieldProfiles._read(profile_id, ("name", "k1", "source"))
        if data is None:
            return None
        return FlatFieldProfile(
            id=profile_id,
            name=str(data["name"]) if "name" in data else profile_id,
            k1=float(data["k1"]) if "k1" in data else 0.0,
            source=str(data["source"]) if "source" in data else "",
        )

    @staticmethod
    def list_profiles() -> List[Tuple[str, str]]:
        """[(id, display name)] for existing profiles, sorted by name (case-insensitive)."""
        directory = APP_CONFIG.flatfield_dir
        if not os.path.isdir(directory):
            return []
        out: List[Tuple[str, str]] = []
        for fname in os.listdir(directory):
            if not fname.endswith(_EXT):
                continue
            prof = FlatFieldProfiles.get(fname[: -len(_EXT)])
            if prof is not None:
                out.append((prof.id, prof.name))
        return sorted(out, key=lambda t: t[1].lower())

    @staticmethod
    def set_k1(profile_id: str, k1: float) -> None:
        """Rewrite a profile's rig distortion in place, preserving its baked gain."""
        data = FlatFieldProfiles._read(profile_id, ("gain", "name", "source"))
        if data is None or "gain" not in data:
            return
        FlatFieldProfiles._write(
            profile_id,
            np.ascontiguousarray(data["gain"], dtype=np.float32),
            name=str(data["name"]) if "name" in data else profile_id,
            k1=k1,
            source=str(data["source"]) if "source" in data else "",
        )

    @staticmethod
    def delete(profile_id: str) -> None:
        try:
            os.remove(FlatFieldProfiles._path_for_id(profile_id))
        except OSError:
            pass
