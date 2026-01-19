import rawpy
import os
from typing import Optional, Dict
from src.domain.models import ColorSpace
from src.kernel.system.config import APP_CONFIG
from src.kernel.system.paths import get_resource_path


class ColorSpaceRegistry:
    """
    Registry for color space mappings and ICC profile locations.
    Centralizes rawpy constants and corresponding ICC profile logic.
    """

    # Mapping of ColorSpace Enum value to rawpy.ColorSpace constant
    _RAWPY_MAP: Dict[str, rawpy.ColorSpace] = {
        ColorSpace.SRGB.value: rawpy.ColorSpace.sRGB,
        ColorSpace.ADOBE_RGB.value: rawpy.ColorSpace.Adobe,
        ColorSpace.PROPHOTO.value: rawpy.ColorSpace.ProPhoto,
        ColorSpace.WIDE.value: rawpy.ColorSpace.Wide,
        ColorSpace.ACES.value: rawpy.ColorSpace.ACES,
        ColorSpace.P3_D65.value: rawpy.ColorSpace.P3D65,
        ColorSpace.REC2020.value: rawpy.ColorSpace.Rec2020,
        ColorSpace.XYZ.value: rawpy.ColorSpace.XYZ,
    }

    # Mapping of ColorSpace Enum value to standard ICC filenames in icc/ directory
    _ICC_MAP: Dict[str, str] = {
        ColorSpace.SRGB.value: "sRGB-v4.icc",
        ColorSpace.ADOBE_RGB.value: "AdobeCompat-v4.icc",
        ColorSpace.PROPHOTO.value: "ProPhoto-v4.icc",
        ColorSpace.WIDE.value: "WideGamut-v4.icc",
        ColorSpace.P3_D65.value: "DisplayP3-v4.icc",
        ColorSpace.REC2020.value: "Rec2020-v4.icc",
    }

    @classmethod
    def get_rawpy_space(cls, cs_name: str) -> rawpy.ColorSpace:
        """
        Resolves UI color space string to rawpy constant.
        Defaults to Adobe RGB for unknown spaces.
        """
        return cls._RAWPY_MAP.get(cs_name, rawpy.ColorSpace.Adobe)

    @classmethod
    def get_icc_path(cls, cs_name: str) -> Optional[str]:
        """
        Locates ICC profile for the given color space.
        Checks application defaults then user overrides.
        """
        # 1. Check mapped defaults
        filename = cls._ICC_MAP.get(cs_name)
        if filename:
            path = get_resource_path(os.path.join("icc", filename))
            if os.path.exists(path):
                return path

        # 2. Check user ICC folder for custom overrides (e.g. "MyCustomSpace.icc")
        custom_path = os.path.join(APP_CONFIG.user_icc_dir, f"{cs_name}.icc")
        if os.path.exists(custom_path):
            return custom_path

        # 3. Special handling for hardcoded app config paths
        if cs_name == ColorSpace.ADOBE_RGB.value:
            return APP_CONFIG.adobe_rgb_profile

        return None
