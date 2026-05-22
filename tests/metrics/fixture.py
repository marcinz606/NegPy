"""
Resolves real RAW file paths for performance timing tests.

Priority for each fixture:
  1. NEGPY_PERF_RAW_<KEY> env var (e.g. NEGPY_PERF_RAW_CR2)
  2. ~/.cache/negpy-metrics/<filename> (previously cached)
  3. Download from rawsamples.ch and cache
  4. None — caller should pytest.skip()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "negpy-metrics"
_RAF_FILENAME = "RAW_FUJI_X-T10.RAF"
_RAF_URL = "http://www.rawsamples.ch/raws/fuji/RAW_FUJI_X-T10.RAF"


@dataclass(frozen=True)
class RawFixture:
    key: str  # short lowercase id used in env vars and metric names
    filename: str  # basename for the local cache file
    url: str  # public download URL


FIXTURES: list[RawFixture] = [
    RawFixture(
        key="cr2",
        filename="RAW_CANON_EOS_5DMARK3.CR2",
        url="http://www.rawsamples.ch/raws/canon/RAW_CANON_EOS_5DMARK3.CR2",
    ),
    RawFixture(
        key="nef",
        filename="RAW_NIKON_D3X.NEF",
        url="http://www.rawsamples.ch/raws/nikon/d3x/RAW_NIKON_D3X.NEF",
    ),
    RawFixture(
        key="arw",
        filename="RAW_SONY_RX10.ARW",
        url="http://rawsamples.ch/raws/sony/RAW_SONY_RX10.ARW",
    ),
    RawFixture(
        key="raf",
        filename="RAW_FUJI_X-T10.RAF",
        url="http://www.rawsamples.ch/raws/fuji/RAW_FUJI_X-T10.RAF",
    ),
    RawFixture(
        key="dng",
        filename="RAW_LEICA_M240.DNG",
        url="http://www.rawsamples.ch/raws/leica/RAW_LEICA_M240.DNG",
    ),
]


def get_fixture_path(fix: RawFixture) -> str | None:
    """Return a local path for *fix*, downloading if needed. None if unavailable."""
    env = os.environ.get(f"NEGPY_PERF_RAW_{fix.key.upper()}", "").strip()
    if env and os.path.isfile(env):
        return env

    cached = _CACHE_DIR / fix.filename
    if cached.is_file():
        return str(cached)

    if _download(fix.url, cached):
        return str(cached)

    return None


def get_perf_raw_path() -> str | None:
    """Return a local path for the RAF perf fixture, downloading if needed."""
    env = os.environ.get("NEGPY_PERF_RAW", "").strip()
    if env and os.path.isfile(env):
        return env

    cached = _CACHE_DIR / _RAF_FILENAME
    if cached.is_file():
        return str(cached)

    if _download(_RAF_URL, cached):
        return str(cached)

    return None


def _download(url: str, dest: Path) -> bool:
    """Download *url* to *dest*, creating parent dirs. Returns True on success."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as f:
            f.write(resp.read())
        tmp.rename(dest)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False
