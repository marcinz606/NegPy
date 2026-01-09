import os
from typing import Any, Optional
from PIL import Image, ImageCms
from src.config import APP_CONFIG


class ColorService:
    """
    Service for applying ICC profiles and simulating color spaces.
    """

    @staticmethod
    def apply_icc_profile(
        pil_img: Image.Image, src_color_space: str, dst_profile_path: Optional[str]
    ) -> Image.Image:
        """
        Transforms a PIL image to a target ICC profile for soft-proofing.
        """
        if not dst_profile_path or not os.path.exists(dst_profile_path):
            return pil_img

        try:
            # Determine source profile
            profile_src: Any
            if src_color_space == "Adobe RGB" and os.path.exists(
                APP_CONFIG.adobe_rgb_profile
            ):
                profile_src = ImageCms.getOpenProfile(APP_CONFIG.adobe_rgb_profile)
            else:
                profile_src = ImageCms.createProfile("sRGB")

            dst_profile: Any = ImageCms.getOpenProfile(dst_profile_path)

            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")

            # Perform the transformation
            result_icc = ImageCms.profileToProfile(
                pil_img,
                profile_src,
                dst_profile,
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
        Simulates how an image in a wide color space (Adobe RGB) looks on an sRGB screen.
        """
        if src_color_space != "Adobe RGB":
            return pil_img

        try:
            if os.path.exists(APP_CONFIG.adobe_rgb_profile):
                adobe_prof = ImageCms.getOpenProfile(APP_CONFIG.adobe_rgb_profile)
                srgb_prof: Any = ImageCms.createProfile("sRGB")

                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                result_sim = ImageCms.profileToProfile(
                    pil_img,
                    adobe_prof,
                    srgb_prof,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB",
                )
                return result_sim if result_sim is not None else pil_img
        except Exception:
            pass
        return pil_img
