import os
import sys
import tomllib
from dataclasses import dataclass

from negpy.domain.types import AppConfig


_DEFAULT_TOML_LINUX_WIN = """\
# NegPy Override Configuration
# Edit this file and restart the app to apply changes.
#
# rendering.backend options:
#   "auto"   - platform default (Vulkan on Linux/Windows, Metal on macOS)
#   "vulkan" - force Vulkan (Linux, Windows)
#   "dx12"   - force Direct3D 12 (Windows only)
#   "metal"  - force Metal (macOS only)
#   "cpu"    - disable GPU acceleration entirely

[rendering]
backend = "vulkan"

[display]
# Qt scene-graph backend. Options: "auto", "vulkan", "d3d12", "metal", "opengl", "software"
qt_rhi_backend = "auto"

# Window system plugin (Linux only). Options: "auto", "xcb", "wayland"
qt_platform = "auto"

[performance]
# Override HQ preview on startup. Uncomment to force a value.
# force_hq_preview = false

# Cap GPU texture dimensions in pixels.
# "auto" lets wgpu/hardware decide the maximum. Set a number (e.g. 4096) to cap it.
max_texture_size = "auto"

# Preview cache size. Larger keeps more recently-viewed photos in memory for instant
# navigation; lower it on machines with little RAM. Uncomment to override defaults.
# preview_cache_max_bytes = 1200000000
# preview_cache_max_entries = 8

[logging]
# Verbosity: "debug", "info", "warning", "error"
level = "info"
"""

_DEFAULT_TOML_MACOS = _DEFAULT_TOML_LINUX_WIN.replace('backend = "vulkan"', 'backend = "metal"')


def _default_toml_content() -> str:
    return _DEFAULT_TOML_MACOS if sys.platform == "darwin" else _DEFAULT_TOML_LINUX_WIN


@dataclass
class OverrideConfig:
    backend: str = "auto"
    qt_rhi_backend: str = "auto"
    qt_platform: str = "auto"
    force_hq_preview: bool | None = None
    max_texture_size: int | None = None
    preview_cache_max_bytes: int | None = None
    preview_cache_max_entries: int | None = None
    log_level: str = "info"

    @property
    def log_level_int(self) -> int:
        import logging

        return {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }.get(self.log_level, logging.INFO)


def load_or_create(path: str) -> "OverrideConfig":
    """Load override.toml, creating it with OS-appropriate defaults if absent."""
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(_default_toml_content())
        except OSError:
            pass
        return _platform_defaults()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return _parse(data)
    except Exception:
        return _platform_defaults()


def _platform_defaults() -> OverrideConfig:
    return OverrideConfig(backend="metal" if sys.platform == "darwin" else "vulkan")


def _parse(data: dict) -> OverrideConfig:
    rendering = data.get("rendering", {})
    display = data.get("display", {})
    performance = data.get("performance", {})
    logging_section = data.get("logging", {})

    backend = str(rendering.get("backend", "auto")).lower()
    if backend not in ("auto", "vulkan", "dx12", "metal", "cpu"):
        backend = "auto"

    qt_rhi = str(display.get("qt_rhi_backend", "auto")).lower()
    if qt_rhi not in ("auto", "vulkan", "d3d12", "metal", "opengl", "software"):
        qt_rhi = "auto"

    qt_platform = str(display.get("qt_platform", "auto")).lower()
    if qt_platform not in ("auto", "xcb", "wayland"):
        qt_platform = "auto"

    raw_hq = performance.get("force_hq_preview")
    force_hq: bool | None = bool(raw_hq) if isinstance(raw_hq, bool) else None

    raw_tex = performance.get("max_texture_size")
    max_tex: int | None = int(raw_tex) if isinstance(raw_tex, int) and raw_tex > 0 else None

    raw_cache_b = performance.get("preview_cache_max_bytes")
    cache_b: int | None = int(raw_cache_b) if isinstance(raw_cache_b, int) and raw_cache_b > 0 else None

    raw_cache_n = performance.get("preview_cache_max_entries")
    cache_n: int | None = int(raw_cache_n) if isinstance(raw_cache_n, int) and raw_cache_n > 0 else None

    log_level = str(logging_section.get("level", "info")).lower()
    if log_level not in ("debug", "info", "warning", "error"):
        log_level = "info"

    return OverrideConfig(
        backend=backend,
        qt_rhi_backend=qt_rhi,
        qt_platform=qt_platform,
        force_hq_preview=force_hq,
        max_texture_size=max_tex,
        preview_cache_max_bytes=cache_b,
        preview_cache_max_entries=cache_n,
        log_level=log_level,
    )


_WGPU_BACKEND: dict[str, str] = {
    "vulkan": "Vulkan",
    "dx12": "D3D12",
    "metal": "Metal",
}

_QT_RHI: dict[str, str] = {
    "vulkan": "vulkan",
    "dx12": "d3d12",  # wgpu name → Qt RHI name
    "d3d12": "d3d12",  # Qt RHI name used directly
    "metal": "metal",
    "opengl": "opengl",
    "software": "software",
}

_QT_PLATFORM: dict[str, str] = {
    "xcb": "xcb",
    "wayland": "wayland",
}


def apply(cfg: OverrideConfig, app_config: AppConfig) -> None:
    """Set env vars and mutate app_config based on override settings."""
    if cfg.backend == "cpu":
        app_config.use_gpu = False
    elif cfg.backend != "auto":
        wgpu_val = _WGPU_BACKEND.get(cfg.backend)
        if wgpu_val:
            os.environ["WGPU_BACKEND_TYPE"] = wgpu_val
        # Derive Qt RHI from backend unless overridden independently
        if cfg.qt_rhi_backend == "auto":
            qt_rhi_val = _QT_RHI.get(cfg.backend)
            if qt_rhi_val:
                os.environ["QSG_RHI_BACKEND"] = qt_rhi_val

    # Independent Qt RHI override takes precedence
    if cfg.qt_rhi_backend != "auto":
        qt_rhi_val = _QT_RHI.get(cfg.qt_rhi_backend)
        if qt_rhi_val:
            os.environ["QSG_RHI_BACKEND"] = qt_rhi_val

    # Qt platform plugin (Linux only)
    if sys.platform == "linux" and cfg.qt_platform != "auto":
        plat_val = _QT_PLATFORM.get(cfg.qt_platform)
        if plat_val:
            os.environ["QT_QPA_PLATFORM"] = plat_val

    if cfg.max_texture_size is not None:
        app_config.max_texture_size = cfg.max_texture_size

    if cfg.force_hq_preview is not None:
        app_config.force_hq_preview = cfg.force_hq_preview

    if cfg.preview_cache_max_bytes is not None:
        app_config.preview_cache_max_bytes = cfg.preview_cache_max_bytes

    if cfg.preview_cache_max_entries is not None:
        app_config.preview_cache_max_entries = cfg.preview_cache_max_entries
