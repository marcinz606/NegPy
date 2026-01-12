import os
import sys
import json


def get_app_version() -> str:
    """
    Reads VERSION or package.json.
    """
    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    )

    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            version_file = os.path.join(sys._MEIPASS, "VERSION")
        else:
            version_file = os.path.join(root_dir, "VERSION")

        if os.path.exists(version_file):
            with open(version_file, "r") as f:
                return f.read().strip()
    except Exception:
        pass

    try:
        pkg_json_path = os.path.join(root_dir, "package.json")
        with open(pkg_json_path, "r") as f:
            data = json.load(f)
            return str(data.get("version", "unknown"))
    except Exception:
        return "unknown"
