import os
import uuid
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from negpy.features.flatfield.logic import Evenness, GainEntry, check_copy, check_reference, compute_gain, gain_token, to_sensor_layout
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.logging import get_logger

logger = get_logger(__name__)

_EXT = ".npz"
# Marks a profile whose gain and reference are in sensor layout; one without it is in its
# reference's EXIF orientation (services/assets/migrations/flatfield_layout.py).
SENSOR_LAYOUT = "sensor"


class FlatFieldProfile(NamedTuple):
    id: str
    name: str
    k1: float
    source: str  # provenance path of the reference the gain was baked from


class StoredCheck(NamedTuple):
    gain: np.ndarray
    reference: Optional[np.ndarray]  # the reference's check copy; None on a profile baked before checks
    check: Optional[Evenness]
    sensor_layout: bool = True


class Baked(NamedTuple):
    gain: np.ndarray
    check: Evenness
    reference: np.ndarray  # check copy, stored so the profile can be checked without its source


class FlatFieldProfiles:
    """npz I/O for flat-field gain-map profiles: one ``<uuid>.npz`` per profile in ``APP_CONFIG.flatfield_dir``, with rig
    metadata and a small reference copy with its self-check, so applying or checking never needs the source. Keyed by id, not
    name, so a rename never breaks an edit's reference. Disk I/O never runs per render (the logic-layer provider caches gains)."""

    @staticmethod
    def _path_for_id(profile_id: str) -> str:
        return os.path.join(APP_CONFIG.flatfield_dir, f"{profile_id}{_EXT}")

    @staticmethod
    def _bake_gain(reference_path: str) -> Optional[Baked]:
        """Decode a reference like a negative (no WB, linear); its gain map, self-check and check copy,
        in sensor layout."""
        if not reference_path or not os.path.exists(reference_path):
            return None
        try:
            from negpy.services.rendering.preview_manager import PreviewManager

            reference, _, meta = PreviewManager().load_linear_preview(reference_path, use_camera_wb=False, full_resolution=False)
            reference = to_sensor_layout(reference, meta.get("orientation", 1))
            gain = compute_gain(reference)
            return Baked(gain, check_reference(reference, gain), check_copy(reference))
        except Exception:
            logger.exception("Flat-field: failed to decode reference %s", reference_path)
            return None

    @staticmethod
    def _write(
        profile_id: str,
        gain: np.ndarray,
        *,
        name: str,
        k1: float,
        source: str,
        reference: Optional[np.ndarray] = None,
        check: Optional[Evenness] = None,
    ) -> None:
        os.makedirs(APP_CONFIG.flatfield_dir, exist_ok=True)
        extra: Dict[str, np.ndarray] = {}
        if reference is not None:
            # float16 is plenty for an evenness view and keeps the profile small.
            extra["reference"] = reference.astype(np.float16)
        if check is not None:
            extra["check"] = np.array([check.low, check.high, check.color, float(check.clipped)], dtype=np.float64)
        np.savez_compressed(
            FlatFieldProfiles._path_for_id(profile_id),
            gain=gain.astype(np.float32),
            token=gain_token(gain),
            name=name,
            k1=float(k1),
            source=source,
            layout=SENSOR_LAYOUT,
            **extra,
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
        baked = FlatFieldProfiles.create_checked(name, reference_path, k1)
        return baked[0] if baked else None

    @staticmethod
    def create_checked(name: str, reference_path: str, k1: float = 0.0) -> Optional[Tuple[str, Evenness]]:
        """`create`, plus how evenly the reference corrects itself and whether it clipped."""
        baked = FlatFieldProfiles._bake_gain(reference_path)
        if baked is None:
            return None
        profile_id = uuid.uuid4().hex
        FlatFieldProfiles._write(
            profile_id, baked.gain, name=name, k1=k1, source=reference_path, reference=baked.reference, check=baked.check
        )
        return profile_id, baked.check

    @staticmethod
    def import_gain(gain: np.ndarray, *, name: str, k1: float = 0.0, source: str = "") -> str:
        """Write an already-baked gain map as a new profile; returns its id."""
        profile_id = uuid.uuid4().hex
        FlatFieldProfiles._write(profile_id, gain, name=name, k1=k1, source=source)
        return profile_id

    @staticmethod
    def load_check(profile_id: str) -> Optional[StoredCheck]:
        """The gain with the stored reference copy and self-check, for Check Flat Field."""
        data = FlatFieldProfiles._read(profile_id, ("gain", "reference", "check", "layout"))
        if data is None or "gain" not in data:
            return None
        reference = data["reference"].astype(np.float32) if "reference" in data else None
        check = None
        if "check" in data:
            low, high, color, clipped = (float(v) for v in data["check"])
            check = Evenness(low, high, color, bool(clipped))
        return StoredCheck(np.ascontiguousarray(data["gain"], dtype=np.float32), reference, check, "layout" in data)

    @staticmethod
    def load_gain(profile_id: str) -> Optional[GainEntry]:
        """The gain map, its content token and layout for the render-path provider; None if missing/unreadable."""
        data = FlatFieldProfiles._read(profile_id, ("gain", "token", "layout"))
        if data is None or "gain" not in data:
            return None
        gain = np.ascontiguousarray(data["gain"], dtype=np.float32)
        token = str(data["token"]) if "token" in data else gain_token(gain)
        return GainEntry(gain, token, "layout" in data)

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
    def delete(profile_id: str) -> None:
        try:
            os.remove(FlatFieldProfiles._path_for_id(profile_id))
        except OSError:
            pass
