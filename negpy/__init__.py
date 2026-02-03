__version__ = "Unknown-dev"

from pathlib import Path

# Read version from VERSION file if it exists
_version_file = Path(__file__).parent / "VERSION"
if _version_file.exists():
    __version__ = _version_file.read_text().strip()
