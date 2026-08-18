from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Tuple


# Valid fine_rotation span (degrees), shared by the sidebar slider and the crop
# tool's handles. Beyond ±45° is an orientation change: use the 90° buttons.
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
    # Reciprocal (portrait) mirrors of the ratios above. The crop tool's picker omits
    # them (CROP_RATIO_CHOICES) because it auto-orients a ratio to the current drag,
    # so both forms would show the same shape twice. They stay real members because
    # the export "Paper ratio" picker does NOT auto-orient, and "Detect closest aspect
    # ratio" needs both forms to match a portrait film frame.
    R_2_3 = "2:3"
    R_3_4 = "3:4"
    R_4_5 = "4:5"
    R_7_6 = "7:6"
    R_5_7 = "5:7"
    R_24_65 = "24:65"
    R_9_16 = "9:16"
    R_10_16 = "10:16"
    R_11_8_5 = "11:8.5"


# Ratios offered in the crop tool's picker: one canonical entry per shape (the
# AspectRatio docstring says why the reciprocals are not here). FREE = no constraint.
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

# Maps each portrait AspectRatio to its canonical entry in CROP_RATIO_CHOICES, so a
# portrait-only value always resolves to something the picker can display.
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
# real film/scan formats only, both orientations. Keep output-only sizes (7:5, 16:9,
# 16:10, 8.5:11) out: they sit too close to 3:2 and 5:4 in log-ratio space, so contour
# noise tips the match onto them and 35mm scans misclassify as "7:5".
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
    # Easel tilt and swing, in per-cent of the frame rather than degrees (see
    # geometry.logic). Positive converge_v stretches the top edge, converge_h the left.
    converge_v: float = 0.0  # [-15.0, 15.0] %
    converge_h: float = 0.0  # [-15.0, 15.0] %
    autocrop_offset: int = 0
    # Free, not 3:2: autocrop reads the film format off the detected frame, so the
    # default fits 6x6, 645 and 6x7 as well as 35mm. A fixed 3:2 center-cropped every
    # other format down to a third of the picture. An explicit ratio still overrides.
    autocrop_ratio: AspectRatio = AspectRatio.FREE
    autocrop_mode: AutocropMode = AutocropMode.IMAGE
    # Fraction of the detected rebate to cut: 0.0 = film edge, 1.0 = image edge, above
    # 1.0 bites into the picture. Image mode only.
    autocrop_rebate_trim: float = 1.0

    # One normalized rect in transformed-image space, auto or manual. `crop_from_auto`
    # doubles as Auto Crop's armed state:
    #   None + False  no crop
    #   None + True   armed: the next render detects a rect, the controller freezes it here
    #   rect + True   resolved auto crop
    #   rect + False  manual crop (or an auto crop since dragged)
    crop_rect: Optional[Tuple[float, float, float, float]] = None
    crop_from_auto: bool = False
    # autocrop_detection_key the auto rect was found under; a mismatch re-detects on the
    # next render. Empty for a manual rect, which nothing invalidates.
    crop_detect_key: str = ""

    def __post_init__(self) -> None:
        """Ensure a JSON-loaded list is converted back to a tuple, keeping the
        frozen dataclass hashable for pipeline cache keys. Enum fields coerce so a
        retired or hand-edited saved value degrades to the default, not a load failure."""
        if self.crop_rect is not None:
            object.__setattr__(self, "crop_rect", tuple(self.crop_rect))
        if self.autocrop_mode not in (AutocropMode.IMAGE, AutocropMode.FILM):
            object.__setattr__(self, "autocrop_mode", AutocropMode.IMAGE)
        try:
            object.__setattr__(self, "autocrop_ratio", AspectRatio(self.autocrop_ratio))
        except ValueError:
            object.__setattr__(self, "autocrop_ratio", AspectRatio.R_3_2)
