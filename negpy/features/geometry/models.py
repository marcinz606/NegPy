from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Tuple


# Valid fine_rotation span (degrees): shared by the sidebar slider and the crop
# tool's rotation handles so both controls cover the same range. Anything beyond
# ±45° is an orientation change and belongs to the 90° rotate buttons.
FINE_ROTATION_LIMIT = 45.0


class AspectRatio(StrEnum):
    FREE = "Free"
    ORIGINAL = "Original"
    R_1_1 = "1:1"
    R_3_2 = "3:2"
    R_4_3 = "4:3"
    R_5_4 = "5:4"
    R_6_7 = "6:7"
    R_7_5 = "7:5"
    R_65_24 = "65:24"
    R_16_9 = "16:9"
    R_16_10 = "16:10"
    R_8_5_11 = "8.5:11"
    # Reciprocal (portrait) mirrors of the ratios above. Not offered in the crop
    # tool's ratio picker (see CROP_RATIO_CHOICES) — the crop tool auto-orients
    # a ratio to match the current drag/box (overlay._oriented_target_ratio,
    # geometry.logic._resolve_ratio_dims), so showing both forms there would just
    # duplicate the same shape twice. Kept as real members because (a) the export
    # "Paper ratio" picker does NOT auto-orient — a portrait vs. landscape paper
    # size are genuinely different choices there — and (b) "Detect closest aspect
    # ratio" needs both forms to match a portrait-oriented film frame correctly.
    R_2_3 = "2:3"
    R_3_4 = "3:4"
    R_4_5 = "4:5"
    R_7_6 = "7:6"
    R_5_7 = "5:7"
    R_24_65 = "24:65"
    R_9_16 = "9:16"
    R_10_16 = "10:16"
    R_11_8_5 = "11:8.5"


# Ratios offered in the crop tool's ratio picker: one canonical entry per shape
# (see the AspectRatio docstring above for why the reciprocal forms exist but
# aren't listed here). FREE is included since it's the "no constraint" choice.
CROP_RATIO_CHOICES: list[AspectRatio] = [
    AspectRatio.FREE,
    AspectRatio.R_1_1,
    AspectRatio.R_3_2,
    AspectRatio.R_4_3,
    AspectRatio.R_5_4,
    AspectRatio.R_6_7,
    AspectRatio.R_7_5,
    AspectRatio.R_65_24,
    AspectRatio.R_16_9,
    AspectRatio.R_16_10,
    AspectRatio.R_8_5_11,
]

# Maps each reciprocal (portrait) AspectRatio to the canonical entry shown in
# CROP_RATIO_CHOICES, so a value that only exists in its portrait form (e.g. from
# "Detect closest aspect ratio" matching a portrait-oriented frame, or a ratio
# saved before this consolidation) always resolves to something the ratio picker
# can display.
_PORTRAIT_TO_CANONICAL_CROP_RATIO: dict[str, str] = {
    AspectRatio.R_2_3: AspectRatio.R_3_2,
    AspectRatio.R_3_4: AspectRatio.R_4_3,
    AspectRatio.R_4_5: AspectRatio.R_5_4,
    AspectRatio.R_7_6: AspectRatio.R_6_7,
    AspectRatio.R_5_7: AspectRatio.R_7_5,
    AspectRatio.R_24_65: AspectRatio.R_65_24,
    AspectRatio.R_9_16: AspectRatio.R_16_9,
    AspectRatio.R_10_16: AspectRatio.R_16_10,
    AspectRatio.R_11_8_5: AspectRatio.R_8_5_11,
}


def canonical_crop_ratio(ratio: str) -> str:
    """Maps a ratio to the form shown in the crop tool's ratio picker. Portrait-
    oriented AspectRatio values collapse to their landscape/canonical counterpart
    (see _PORTRAIT_TO_CANONICAL_CROP_RATIO); "Free", "Original", and anything
    already canonical pass through unchanged."""
    return _PORTRAIT_TO_CANONICAL_CROP_RATIO.get(ratio, ratio)


# Candidates for "Detect closest aspect ratio" (geometry.logic._closest_standard_ratio):
# real film/scan formats only, both orientations. 7:5, 16:9, 16:10 and 8.5:11 are
# print/screen *output* sizes, not scannable film formats, and sit close enough to
# 3:2 (1.4, 1.778, 1.6) and 5:4 (0.773) in log-ratio space that ordinary contour-
# detection noise on a real 3:2 or 5:4 frame tips the match onto one of them instead
# — including them here regressed detection to reliably misclassify 35mm scans as
# "7:5". Keep this set to formats a camera/scanner could actually produce.
FILM_FORMAT_RATIOS: list[AspectRatio] = [
    AspectRatio.R_1_1,
    AspectRatio.R_3_2,
    AspectRatio.R_2_3,
    AspectRatio.R_4_3,
    AspectRatio.R_3_4,
    AspectRatio.R_5_4,
    AspectRatio.R_4_5,
    AspectRatio.R_6_7,
    AspectRatio.R_7_6,
    AspectRatio.R_65_24,
    AspectRatio.R_24_65,
]


class AutocropMode(StrEnum):
    IMAGE = "image"  # crop to exposed image area (default)
    FILM = "film"  # crop to film extent, keep rebate/sprockets


@dataclass(frozen=True)
class GeometryConfig:
    rotation: int = 0
    fine_rotation: float = 0.0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    auto_crop_enabled: bool = False

    autocrop_offset: int = 0
    # Free, not 3:2: autocrop reads the film format off the frame it detected (see
    # geometry.logic.get_autocrop_coords), so the default fits 6x6, 645 and 6x7 as well as
    # 35mm. A fixed 3:2 default center-cropped every other format down to a third of the
    # picture. Any explicit ratio still overrides.
    autocrop_ratio: AspectRatio = AspectRatio.FREE
    autocrop_mode: AutocropMode = AutocropMode.IMAGE
    # Fraction of the detected rebate to cut: 0.0 stops at the film edge, 1.0 lands on
    # the image edge, above 1.0 bites into the picture. Image mode only.
    autocrop_rebate_trim: float = 1.0
    manual_crop_rect: Optional[Tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        """Ensure a JSON-loaded list is converted back to a tuple, keeping the
        frozen dataclass hashable for pipeline cache keys. Enum fields coerce so a
        retired or hand-edited saved value degrades to the default, not a load failure."""
        if self.manual_crop_rect is not None:
            object.__setattr__(self, "manual_crop_rect", tuple(self.manual_crop_rect))
        if self.autocrop_mode not in (AutocropMode.IMAGE, AutocropMode.FILM):
            object.__setattr__(self, "autocrop_mode", AutocropMode.IMAGE)
        try:
            object.__setattr__(self, "autocrop_ratio", AspectRatio(self.autocrop_ratio))
        except ValueError:
            object.__setattr__(self, "autocrop_ratio", AspectRatio.R_3_2)
