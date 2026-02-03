import os
import sys
import json
import urllib.request
from typing import Optional


def get_app_version() -> str:
    """
    Reads VERSION or package.json.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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


def check_for_updates() -> Optional[str]:
    """
    Checks if there is new version available in github and if it is return it's version, else return none.
    """
    current_version = get_app_version()
    if current_version == "unknown":
        return None

    url = "https://api.github.com/repos/marcinz606/NegPy/releases/latest"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "NegPy-Updater"},
        )
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                latest_tag = data.get("tag_name", "").lstrip("v")

                if not latest_tag:
                    return None

                def parse_version(v_str: str) -> list[int]:
                    try:
                        return [int(x) for x in v_str.split(".") if x.isdigit()]
                    except Exception:
                        return []

                current_parts = parse_version(current_version)
                latest_parts = parse_version(latest_tag)

                if latest_parts > current_parts:
                    return str(latest_tag)

    except Exception:
        pass

    return None
