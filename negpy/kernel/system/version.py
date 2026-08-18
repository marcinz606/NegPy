import os
import sys
import json
import urllib.request
from typing import Any, Optional

GITHUB_REPO = "marcinz606/NegPy"
LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases"
ISSUES_PAGE = f"https://github.com/{GITHUB_REPO}/issues/new/choose"
USER_AGENT = "NegPy-Updater"


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
            with open(version_file, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass

    try:
        pkg_json_path = os.path.join(root_dir, "package.json")
        with open(pkg_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return str(data.get("version", "unknown"))
    except Exception:
        return "unknown"


def parse_version(v_str: str) -> list[int]:
    """The numeric parts of a version string, for ordered comparison."""
    try:
        return [int(x) for x in v_str.split(".") if x.isdigit()]
    except Exception:
        return []


def fetch_latest_release(timeout: float = 5.0) -> Optional[dict[str, Any]]:
    """The GitHub payload for the newest release, or None if it cannot be read."""
    try:
        req = urllib.request.Request(LATEST_RELEASE_API, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode())
    except Exception:
        return None

    return payload if isinstance(payload, dict) else None


def is_newer(candidate: str, current: str) -> bool:
    """True when `candidate` names a release later than `current`."""
    candidate_parts = parse_version(candidate)
    if not candidate_parts or current == "unknown":
        return False
    return candidate_parts > parse_version(current)
