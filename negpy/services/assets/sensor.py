import os
import tomllib
from typing import List, Optional

from negpy.kernel.system.config import APP_CONFIG
from negpy.services.assets.naming import escape_toml_string, slugify

NONE_NAME = "None"


class SensorProfiles:
    """
    TOML I/O for user sensor-crosstalk matrices (features/process/sensor.py).

    Files live in APP_CONFIG.sensor_dir. "None" means no correction — there is
    no built-in matrix, a sensor matrix only exists once the user calibrates
    their setup. Disk I/O only happens on dropdown build and on selection --
    never per render (matrices are baked into ProcessConfig).
    """

    NONE_NAME = NONE_NAME

    @staticmethod
    def _scan() -> dict:
        """Maps display-name -> flat 9-float matrix for valid .toml files."""
        result: dict = {}
        directory = APP_CONFIG.sensor_dir
        if not os.path.isdir(directory):
            return result
        for fname in os.listdir(directory):
            if not fname.endswith(".toml"):
                continue
            parsed = SensorProfiles._parse_file(os.path.join(directory, fname))
            if parsed is None:
                continue
            name, matrix = parsed
            name = name or fname[:-5]
            if name != NONE_NAME:
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
        """["None", *sorted custom display-names]."""
        return [NONE_NAME, *sorted(SensorProfiles._scan().keys())]

    @staticmethod
    def get_matrix(name: str) -> Optional[List[float]]:
        """Flat 9-float list, or None for "None" / missing / invalid (= no correction)."""
        if name == NONE_NAME:
            return None
        return SensorProfiles._scan().get(name)

    @staticmethod
    def is_bundled(name: str) -> bool:
        """True for the read-only "None" entry."""
        return name == NONE_NAME

    @staticmethod
    def path_for_name(name: str) -> str:
        """Filesystem path a user profile with this display name would use."""
        return os.path.join(APP_CONFIG.sensor_dir, f"{slugify(name, 'sensor')}.toml")

    @staticmethod
    def save(name: str, matrix: List[float]) -> str:
        """Write a user profile TOML (row-major 3×3) and return its path."""
        os.makedirs(APP_CONFIG.sensor_dir, exist_ok=True)
        rows = "\n".join("  [{:.6g}, {:.6g}, {:.6g}],".format(*matrix[i * 3 : i * 3 + 3]) for i in range(3))
        content = f'name = "{escape_toml_string(name)}"\nmatrix = [\n{rows}\n]\n'
        path = SensorProfiles.path_for_name(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    @staticmethod
    def delete(name: str) -> None:
        """Remove the user profile file whose display name matches (no-op if absent)."""
        directory = APP_CONFIG.sensor_dir
        if not os.path.isdir(directory):
            return
        for fname in os.listdir(directory):
            if not fname.endswith(".toml"):
                continue
            path = os.path.join(directory, fname)
            parsed = SensorProfiles._parse_file(path)
            if parsed is None:
                continue
            parsed_name = parsed[0] or fname[:-5]
            if parsed_name == name:
                os.remove(path)
                return
