import os
import tomllib
from typing import List, Optional

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.paths import get_resource_path
from negpy.services.assets.naming import escape_toml_string, slugify

# Renamed from "Default"
DEFAULT_NAME = "Generic C41"

# A profile's `type` records where its numbers came from. Free-form on disk; anything
# outside this set groups under "Other" rather than disappearing.
TYPE_SPECSHEET = "specsheet-based"  # read off published spectral dye-density curves
TYPE_MEASURED = "measured"  # fitted against real scans of a known reference
TYPE_TUNED = "tuned"  # dialled in by eye on a rig; what the editor saves
TYPE_BUILTIN = "built-in"

#: Dropdown group order and headings; the trailing entry is the catch-all.
GROUP_ORDER: tuple[tuple[str, str], ...] = (
    (TYPE_BUILTIN, "Built-in"),
    (TYPE_MEASURED, "Measured"),
    (TYPE_TUNED, "Tuned on a rig"),
    (TYPE_SPECSHEET, "From spec sheets (approx)"),
    ("", "Other"),
)


class CrosstalkProfiles:
    """
    TOML I/O for user spectral-crosstalk matrices.

    Files live in APP_CONFIG.crosstalk_dir. The built-in hardcoded matrix is
    exposed as the "Generic C41" profile. Disk I/O only happens on dropdown build
    and on selection -- never per render (matrices are baked into ProcessConfig).
    """

    DEFAULT_NAME = DEFAULT_NAME

    @staticmethod
    def _scan_dir(directory: str) -> dict:
        """Maps display-name -> flat 9-float matrix for valid .toml files in a directory."""
        result: dict = {}
        if not os.path.isdir(directory):
            return result
        for fname in os.listdir(directory):
            if not fname.endswith(".toml"):
                continue
            path = os.path.join(directory, fname)
            parsed = CrosstalkProfiles._parse_file(path)
            if parsed is None:
                continue
            name, matrix = parsed
            name = name or fname[:-5]
            if name != DEFAULT_NAME:
                result[name] = matrix
        return result

    @staticmethod
    def scan_bundled() -> dict:
        """Read-only matrices shipped with the app, keyed by display name."""
        return CrosstalkProfiles._scan_dir(get_resource_path("crosstalk"))

    @staticmethod
    def scan_user() -> dict:
        """User-editable matrices in the docs folder, keyed by display name."""
        return CrosstalkProfiles._scan_dir(APP_CONFIG.crosstalk_dir)

    @staticmethod
    def _scan() -> dict:
        """Bundled ∪ user custom matrices, keyed by display name; bundled wins."""
        return {**CrosstalkProfiles.scan_user(), **CrosstalkProfiles.scan_bundled()}

    @staticmethod
    def _parse_file(path: str) -> Optional[tuple]:
        """Parses a .toml file to (name, flat 9-float list), or None if invalid.

        `type` is read separately by `_scan_types`: callers unpack this tuple positionally."""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
            rows = data.get("matrix")
            if not isinstance(rows, list) or len(rows) != 3:
                return None
            flat: List[float] = []
            for row in rows:
                if not isinstance(row, list) or len(row) != 3:
                    return None
                for v in row:
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        return None
                    flat.append(float(v))
            raw_name = data.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
            return name, flat
        except Exception:
            return None

    @staticmethod
    def _parse_type(path: str) -> str:
        """The profile's `type`, lowercased; "" when absent or unreadable."""
        try:
            with open(path, "rb") as f:
                raw = tomllib.load(f).get("type")
        except Exception:
            return ""
        return raw.strip().lower() if isinstance(raw, str) else ""

    @staticmethod
    def _scan_types() -> dict:
        """display-name -> type for every valid profile; bundled wins, like _scan."""
        types: dict = {}
        for directory in (APP_CONFIG.crosstalk_dir, get_resource_path("crosstalk")):
            if not os.path.isdir(directory):
                continue
            for fname in os.listdir(directory):
                if not fname.endswith(".toml"):
                    continue
                path = os.path.join(directory, fname)
                parsed = CrosstalkProfiles._parse_file(path)
                if parsed is None:
                    continue
                name = parsed[0] or fname[:-5]
                if name != DEFAULT_NAME:
                    types[name] = CrosstalkProfiles._parse_type(path)
        return types

    @staticmethod
    def grouped_profiles() -> List[tuple]:
        """[(heading, [profile names])] in GROUP_ORDER, skipping empty groups.

        Every profile lands in exactly one group, so the flattened names are `list_profiles()`
        reordered; an unrecognised type cannot drop one."""
        types = CrosstalkProfiles._scan_types()
        known = {t for t, _ in GROUP_ORDER if t}
        buckets: dict = {t: [] for t, _ in GROUP_ORDER}
        buckets[TYPE_BUILTIN].append(DEFAULT_NAME)
        for name in sorted(types):
            bucket = types[name] if types[name] in known else ""
            buckets[bucket].append(name)
        return [(heading, buckets[t]) for t, heading in GROUP_ORDER if buckets[t]]

    @staticmethod
    def get_type(name: str) -> str:
        """A profile's type, or TYPE_BUILTIN for the built-in / "" when unknown."""
        if name == DEFAULT_NAME:
            return TYPE_BUILTIN
        return CrosstalkProfiles._scan_types().get(name, "")

    @staticmethod
    def list_profiles() -> List[str]:
        """["Generic C41", *sorted custom display-names]."""
        return [DEFAULT_NAME, *sorted(CrosstalkProfiles._scan().keys())]

    @staticmethod
    def get_matrix(name: str) -> Optional[List[float]]:
        """
        Flat 9-float list for a profile, or None for the built-in / missing /
        invalid profiles. None means the render path uses process.models.DEFAULT_CROSSTALK_MATRIX.
        """
        if name == DEFAULT_NAME:
            return None
        return CrosstalkProfiles._scan().get(name)

    @staticmethod
    def is_bundled(name: str) -> bool:
        """True for read-only profiles: the built-in or any bundled matrix."""
        return name == DEFAULT_NAME or name in CrosstalkProfiles.scan_bundled()

    @staticmethod
    def path_for_name(name: str) -> str:
        """Filesystem path a user profile with this display name would use."""
        return os.path.join(APP_CONFIG.crosstalk_dir, f"{slugify(name, 'crosstalk')}.toml")

    @staticmethod
    def save(name: str, matrix: List[float], profile_type: str = TYPE_TUNED) -> str:
        """Write a user profile TOML (row-major 3×3) and return its path.

        Defaults to `tuned` so editor saves are not grouped with the spec-sheet estimates."""
        os.makedirs(APP_CONFIG.crosstalk_dir, exist_ok=True)
        rows = "\n".join("  [{:.6g}, {:.6g}, {:.6g}],".format(*matrix[i * 3 : i * 3 + 3]) for i in range(3))
        content = f'name = "{escape_toml_string(name)}"\ntype = "{escape_toml_string(profile_type)}"\nmatrix = [\n{rows}\n]\n'
        path = CrosstalkProfiles.path_for_name(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    @staticmethod
    def delete(name: str) -> None:
        """Remove the user profile file whose display name matches (no-op if absent)."""
        directory = APP_CONFIG.crosstalk_dir
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if not fname.endswith(".toml"):
                continue
            path = os.path.join(directory, fname)
            parsed = CrosstalkProfiles._parse_file(path)
            if parsed is None:
                continue
            parsed_name = parsed[0] or fname[:-5]
            if parsed_name == name:
                os.remove(path)
                return

    @staticmethod
    def ensure_user_dir() -> None:
        """Make sure the user's crosstalk directory exists; no seeding."""
        os.makedirs(APP_CONFIG.crosstalk_dir, exist_ok=True)
