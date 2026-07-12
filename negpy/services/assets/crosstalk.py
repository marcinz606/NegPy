import os
import re
import tomllib
from typing import List, Optional

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.paths import get_resource_path

DEFAULT_NAME = "Default"


# ponytail: 2-line helpers duplicated from contact_sheet_templates; extract to a
# shared module only if a third consumer appears.
def _slugify(name: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", name.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_")
    return slug or "crosstalk"


def _escape_toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


class CrosstalkProfiles:
    """
    TOML I/O for user spectral-crosstalk matrices.

    Files live in APP_CONFIG.crosstalk_dir. The built-in hardcoded matrix is
    exposed as the "Default" profile. Disk I/O only happens on dropdown build
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
        """Parses a .toml file to (name, flat 9-float list), or None if invalid."""
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
    def list_profiles() -> List[str]:
        """["Default", *sorted custom display-names]."""
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
        """True for read-only profiles: the built-in Default or any bundled matrix."""
        return name == DEFAULT_NAME or name in CrosstalkProfiles.scan_bundled()

    @staticmethod
    def path_for_name(name: str) -> str:
        """Filesystem path a user profile with this display name would use."""
        return os.path.join(APP_CONFIG.crosstalk_dir, f"{_slugify(name)}.toml")

    @staticmethod
    def save(name: str, matrix: List[float]) -> str:
        """Write a user profile TOML (row-major 3×3) and return its path."""
        os.makedirs(APP_CONFIG.crosstalk_dir, exist_ok=True)
        rows = "\n".join(
            "  [{:.6g}, {:.6g}, {:.6g}],".format(*matrix[i * 3 : i * 3 + 3]) for i in range(3)
        )
        content = f'name = "{_escape_toml_string(name)}"\nmatrix = [\n{rows}\n]\n'
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
