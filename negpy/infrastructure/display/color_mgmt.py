import io
import os
from functools import lru_cache
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageCms
from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.paths import get_resource_path
from negpy.kernel.system.logging import get_logger
from negpy.domain.models import ColorSpace
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def open_profile_from_bytes(data: bytes) -> Any:
    """Open an ICC profile from raw bytes (e.g. a monitor profile from Qt)."""
    return ImageCms.getOpenProfile(io.BytesIO(data))


def icc_bytes_for_space(cs_name: str) -> Optional[bytes]:
    """Raw ICC bytes for a named color space's bundled profile, or None if missing.

    Used to back a manual display-profile override with a common space.
    """
    path = ColorSpaceRegistry.get_icc_path(cs_name)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        logger.warning("Failed to read ICC profile for %s", cs_name, exc_info=e)
        return None


def profile_description(data: Optional[bytes]) -> str:
    """Human-readable description of a monitor ICC profile, for the UI.

    ``None`` (no profile detected → sRGB assumed) returns a labelled fallback.
    """
    if not data:
        return "sRGB fallback"
    try:
        return ImageCms.getProfileDescription(open_profile_from_bytes(data)).strip()
    except Exception as e:
        logger.debug("Could not read profile description: %s", e)
        return "Unknown profile"


@lru_cache(maxsize=16)
def get_display_lut(working_color_space: str, dst_bytes: Optional[bytes] = None) -> Optional[np.ndarray]:
    """3D LUT converting the assumed source space (`WORKING_COLOR_SPACE`) to the
    display profile.

    ``working_color_space`` is the profile the camera-native pipeline numbers are
    *assumed* to be in (see `color_spaces.WORKING_COLOR_SPACE`), not a real working
    space. When ``dst_bytes`` is ``None`` the destination is sRGB (legacy behaviour,
    i.e. the display is assumed to be sRGB); otherwise it is the monitor's ICC
    profile. Returns ``None`` only when the transform is a no-op (source is sRGB and
    the display is sRGB), so callers can skip it. Cached per (source, display
    profile). Used by both the CPU (`ImageConverter.to_qimage`) and GPU display paths.
    """
    if working_color_space == ColorSpace.SRGB.value and dst_bytes is None:
        return None
    try:
        from negpy.infrastructure.display.icc_lut import build_3d_lut

        src = ColorService._get_profile(working_color_space)
        dst: Any = open_profile_from_bytes(dst_bytes) if dst_bytes else ImageCms.createProfile("sRGB")
        return build_3d_lut(
            src,
            dst,
            ImageCms.Intent.RELATIVE_COLORIMETRIC,
            ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
    except Exception as e:
        logger.warning("Failed to build display LUT for %s", working_color_space, exc_info=e)
        return None


def apply_display_transform(buffer: np.ndarray, working_color_space: str, dst_bytes: Optional[bytes] = None) -> np.ndarray:
    """Convert a float32 RGB buffer from the working space to the display profile.

    No-op for non-RGB buffers (greyscale display stays neutral) or when the
    transform is identity (sRGB working space on an sRGB display). ``dst_bytes`` is
    the monitor's ICC profile; ``None`` falls back to sRGB. Used on the CPU display
    path before quantizing to 8-bit.
    """
    if buffer.dtype != np.float32 or buffer.ndim != 3 or buffer.shape[2] != 3:
        return buffer
    lut = get_display_lut(working_color_space, dst_bytes)
    if lut is None:
        return buffer
    from negpy.infrastructure.display.icc_lut import apply_lut_f32

    return apply_lut_f32(buffer, lut)


class ColorService:
    """
    ICC profile application & soft-proofing.
    """

    @staticmethod
    def _get_profile(cs_name: str) -> Any:
        """
        Helper to load profile for a named color space.
        """
        path = ColorSpaceRegistry.get_icc_path(cs_name)
        if path and os.path.exists(path):
            return ImageCms.getOpenProfile(path)

        # Fallback to built-in if possible, else sRGB
        if cs_name == ColorSpace.XYZ.value:
            return ImageCms.createProfile("XYZ")

        return ImageCms.createProfile("sRGB")

    @staticmethod
    def apply_icc_profile(
        pil_img: Image.Image,
        src_color_space: str,
        dst_profile_path: Optional[str],
        inverse: bool = False,
    ) -> Image.Image:
        """
        Applies ICC for proofing or correction.
        """
        if not dst_profile_path or not os.path.exists(dst_profile_path):
            return pil_img

        try:
            profile_working = ColorService._get_profile(src_color_space)
            profile_selected: Any = ImageCms.getOpenProfile(dst_profile_path)

            if inverse:
                profile_src = profile_selected
                profile_dst = profile_working
            else:
                profile_src = profile_working
                profile_dst = profile_selected

            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            result_icc = ImageCms.profileToProfile(
                pil_img,
                profile_src,
                profile_dst,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode="RGB",
                flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
            )
            return result_icc if result_icc is not None else pil_img
        except Exception as e:
            logger.warning("Failed to apply ICC profile", exc_info=e)
            return pil_img

    @staticmethod
    def simulate_on_srgb(pil_img: Image.Image, src_color_space: str) -> Image.Image:
        """
        Transforms working space buffer to sRGB for display.
        """
        if src_color_space == ColorSpace.SRGB.value:
            return pil_img

        try:
            src_prof = ColorService._get_profile(src_color_space)
            srgb_prof: Any = ImageCms.createProfile("sRGB")

            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            result_sim = ImageCms.profileToProfile(
                pil_img,
                src_prof,
                srgb_prof,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode="RGB",
            )
            return result_sim if result_sim is not None else pil_img
        except Exception as e:
            logger.warning("Failed to simulate color space transform to sRGB", exc_info=e)
        return pil_img

    @staticmethod
    def get_available_profiles() -> list[str]:
        """
        Returns list of available ICC profile paths.
        """
        icc_root = get_resource_path("icc")
        built_in_icc = []
        if os.path.exists(icc_root):
            built_in_icc = [os.path.join(icc_root, f) for f in os.listdir(icc_root) if f.lower().endswith((".icc", ".icm"))]

        user_icc = []
        if os.path.exists(APP_CONFIG.user_icc_dir):
            user_icc = [
                os.path.join(APP_CONFIG.user_icc_dir, f)
                for f in os.listdir(APP_CONFIG.user_icc_dir)
                if f.lower().endswith((".icc", ".icm"))
            ]
        return sorted(built_in_icc + user_icc)
