import os
from typing import Any, Optional
from PIL import Image, ImageCms
from src.kernel.system.config import APP_CONFIG
from src.kernel.system.paths import get_resource_path
from src.domain.models import ColorSpace
from src.infrastructure.display.color_spaces import ColorSpaceRegistry


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
        except Exception:
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
        except Exception:
            pass
        return pil_img

    @staticmethod
    def get_available_profiles() -> list[str]:
        """
        Returns list of available ICC profile paths.
        """
        icc_root = get_resource_path("icc")
        built_in_icc = []
        if os.path.exists(icc_root):
            built_in_icc = [
                os.path.join(icc_root, f)
                for f in os.listdir(icc_root)
                if f.lower().endswith((".icc", ".icm"))
            ]

        user_icc = []
        if os.path.exists(APP_CONFIG.user_icc_dir):
            user_icc = [
                os.path.join(APP_CONFIG.user_icc_dir, f)
                for f in os.listdir(APP_CONFIG.user_icc_dir)
                if f.lower().endswith((".icc", ".icm"))
            ]
        return sorted(built_in_icc + user_icc)
