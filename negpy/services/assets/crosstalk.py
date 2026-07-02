import os
import shutil
import tomllib
from typing import List, Optional

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.paths import get_resource_path

DEFAULT_NAME = "Default"


class CrosstalkProfiles:
    """
    TOML I/O for user spectral-crosstalk matrices.

    Files live in APP_CONFIG.crosstalk_dir. The built-in hardcoded matrix is
    exposed as the "Default" profile. Disk I/O only happens on dropdown build
    and on selection -- never per render (matrices are baked into ProcessConfig).
    """

    DEFAULT_NAME = DEFAULT_NAME

    @staticmethod
    def _scan() -> dict:
        """Maps display-name -> flat 9-float matrix for valid custom .toml files."""
        result: dict = {}
        crosstalk_dir = APP_CONFIG.crosstalk_dir
        if not os.path.isdir(crosstalk_dir):
            return result
        for fname in os.listdir(crosstalk_dir):
            if not fname.endswith(".toml"):
                continue
            path = os.path.join(crosstalk_dir, fname)
            parsed = CrosstalkProfiles._parse_file(path)
            if parsed is None:
                continue
            name, matrix = parsed
            name = name or fname[:-5]
            if name != DEFAULT_NAME:
                result[name] = matrix
        return result

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
    def seed_example() -> None:
        """Copy bundled gallery matrices into the user folder, per file.

        Skips a file that already exists in the user folder, so user edits
        are never overwritten. New matrices bundled in later releases still
        get copied in on next startup.
        """
        crosstalk_dir = APP_CONFIG.crosstalk_dir
        bundled_dir = get_resource_path("crosstalk")
        try:
            if not os.path.isdir(crosstalk_dir) or not os.path.isdir(bundled_dir):
                return
            for fname in os.listdir(bundled_dir):
                if not fname.endswith(".toml"):
                    continue
                dest = os.path.join(crosstalk_dir, fname)
                if os.path.exists(dest):
                    continue
                shutil.copyfile(os.path.join(bundled_dir, fname), dest)
        except OSError:
            pass
