import copy
import os
import uuid

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Any, Optional
from enum import Enum, StrEnum
from negpy.features.process.models import ProcessConfig
from negpy.features.exposure.models import ExposureConfig, RenderIntent
from negpy.features.geometry.models import GeometryConfig
from negpy.features.lab.models import LabConfig
from negpy.features.local.models import LocalAdjustmentsConfig, PolygonMask
from negpy.features.retouch.models import RetouchConfig
from negpy.features.toning.models import ToningConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.features.metadata.models import MetadataConfig
from negpy.kernel.system.logging import get_logger
import negpy.kernel.system.paths as paths

logger = get_logger("domain.models")

# Map of old field names → new field names for backward-compatible deserialization.
# Add entries here when fields are renamed so old workspace files keep their data.
MIGRATIONS: Dict[str, str] = {
    "export_border_size": "border_size",
    "export_border_color": "border_color",
    # Shadow-neutral + density-balance consolidated into Cast Removal, then Cast
    # Removal itself became a 0..1 strength (bool True/False coerced to 1.0/0.0 in
    # ExposureConfig.__post_init__). Preserve a user's saved on/off.
    "auto_shadow_neutral": "cast_removal_strength",
    "cast_removal": "cast_removal_strength",
    # D-Range Clip split into independent luma + colour range clips; the old single
    # slider maps to the luma axis (colour defaults to its aggressive baseline).
    "drange_clip": "luma_range_clip",
}


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


class ExportFormat(StrEnum):
    JPEG = "JPEG"
    TIFF = "TIFF"
    PNG = "PNG"
    DNG = "DNG"
    JXL = "JXL"
    WEBP = "WEBP"


class ExportPresetOutputMode(StrEnum):
    SUBFOLDER_OF_SOURCE = "subfolder_of_source"
    SAME_AS_SOURCE = "same_as_source"
    ABSOLUTE = "absolute"


class ExportResolutionMode(StrEnum):
    ORIGINAL = "original"
    PRINT = "print"
    TARGET_PX = "target_px"


class ICCMode(Enum):
    OUTPUT = "Output"
    INPUT = "Input"


class ColorSpace(Enum):
    SAME_AS_SOURCE = "Same as Source"
    SRGB = "sRGB"
    ADOBE_RGB = "Adobe RGB"
    PROPHOTO = "ProPhoto RGB"
    ACES = "ACES"
    P3_D65 = "P3 D65"
    REC2020 = "Rec 2020"
    XYZ = "XYZ"
    GREYSCALE = "Greyscale"


# Colour spaces JPEG XL can tag (mirror _JXL_COLOR in image_processor). Same as
# Source is allowed — resolved at export time and rejected by the encoder if it
# lands on an unsupported space.
JXL_TAGGABLE_SPACES = frozenset(
    {
        ColorSpace.SRGB.value,
        ColorSpace.P3_D65.value,
        ColorSpace.REC2020.value,
        ColorSpace.GREYSCALE.value,
        ColorSpace.SAME_AS_SOURCE.value,
    }
)


def export_blocked(fmt: str, color_space: str) -> bool:
    """True when the format + colour-space pairing can't be encoded (JPEG XL
    only tags a subset of colour spaces)."""
    return fmt == ExportFormat.JXL and color_space not in JXL_TAGGABLE_SPACES


@dataclass(frozen=True)
class ExportConfig:
    """
    Export parameters (path, format, sizing).
    """

    userDir: str = field(default_factory=paths.get_default_user_dir)

    export_path: str = field(default_factory=lambda: os.path.join(paths.get_default_user_dir(), "export"))
    export_fmt: str = ExportFormat.JPEG
    jpeg_quality: int = 90
    jxl_lossless: bool = True
    jxl_distance: float = 1.0  # libjxl distance; only used when jxl_lossless is False
    jxl_effort: int = 7
    webp_quality: int = 90
    webp_lossless: bool = False
    webp_method: int = 4  # PIL encode effort 0-6, higher = slower/smaller
    export_color_space: str = ColorSpace.SRGB.value
    paper_aspect_ratio: str = AspectRatio.ORIGINAL
    export_print_size: float = 30.0
    export_dpi: int = 300
    export_resolution_mode: str = ExportResolutionMode.ORIGINAL.value
    export_target_long_edge_px: int = 2000
    filename_pattern: str = "{{ original_name }}"
    # When True, exports silently overwrite existing files; when False, the export
    # prompts (Overwrite / Rename / Cancel) before clobbering anything.
    overwrite: bool = False
    output_mode: str = ExportPresetOutputMode.ABSOLUTE
    output_subfolder: str = ""
    icc_input_path: Optional[str] = None
    icc_output_path: Optional[str] = None

    contact_sheet_cell_px: int = 600
    contact_sheet_gap: int = 16
    contact_sheet_margin: int = 32
    contact_sheet_max_tiles: int = 38
    contact_sheet_output_path: str = ""  # empty = follow export destination rules
    contact_sheet_template: str = ""  # empty = Default template active
    contact_sheet_default_cell_px: int = 600
    contact_sheet_default_gap: int = 16
    contact_sheet_default_margin: int = 32
    contact_sheet_default_max_tiles: int = 38

    export_sidecars_enabled: bool = False


@dataclass
class ExportPreset:
    """
    A single export preset defining format, sizing, destination, and color settings.
    Field names for sizing/color match ExportConfig so PrintService can accept either.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Preset"
    enabled: bool = True
    render_intent: str = RenderIntent.PRINT  # fixed at creation; not exposed in the preset editor form

    # Format
    export_fmt: str = ExportFormat.JPEG
    jpeg_quality: int = 90
    jxl_lossless: bool = True
    jxl_distance: float = 1.0
    jxl_effort: int = 7
    webp_quality: int = 90
    webp_lossless: bool = False
    webp_method: int = 4

    # Sizing (same field names as ExportConfig for PrintService compatibility)
    export_resolution_mode: str = ExportResolutionMode.ORIGINAL.value
    paper_aspect_ratio: str = AspectRatio.ORIGINAL
    export_print_size: float = 30.0
    export_dpi: int = 300
    export_target_long_edge_px: int = 2000

    # Output destination
    output_mode: str = ExportPresetOutputMode.SAME_AS_SOURCE
    output_subfolder: str = ""
    output_path: str = ""
    overwrite: bool = False
    filename_pattern: str = "{{ original_name }}"

    # Color
    export_color_space: str = ColorSpace.SRGB.value
    icc_input_path: Optional[str] = None
    icc_output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "render_intent": self.render_intent,
            "export_fmt": self.export_fmt,
            "jpeg_quality": self.jpeg_quality,
            "jxl_lossless": self.jxl_lossless,
            "jxl_distance": self.jxl_distance,
            "jxl_effort": self.jxl_effort,
            "webp_quality": self.webp_quality,
            "webp_lossless": self.webp_lossless,
            "webp_method": self.webp_method,
            "export_resolution_mode": self.export_resolution_mode,
            "paper_aspect_ratio": self.paper_aspect_ratio,
            "export_print_size": self.export_print_size,
            "export_dpi": self.export_dpi,
            "export_target_long_edge_px": self.export_target_long_edge_px,
            "output_mode": self.output_mode,
            "output_subfolder": self.output_subfolder,
            "output_path": self.output_path,
            "overwrite": self.overwrite,
            "filename_pattern": self.filename_pattern,
            "export_color_space": self.export_color_space,
            "icc_input_path": self.icc_input_path,
            "icc_output_path": self.icc_output_path,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExportPreset":
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in known})


def preset_display_name(preset: ExportPreset) -> str:
    """Preset list/checkbox label, flagging flat-master presets."""
    if preset.render_intent == RenderIntent.FLAT:
        return f"{preset.name} (flat)"
    return preset.name


def preset_from_export_config(conf: ExportConfig, name: str = "Current settings") -> ExportPreset:
    """Builds an ephemeral preset from the current export settings so the export
    pipeline (which is preset-driven) can run a one-off 'export as currently seen'."""
    return ExportPreset(
        name=name,
        enabled=True,
        export_fmt=conf.export_fmt,
        jpeg_quality=conf.jpeg_quality,
        jxl_lossless=conf.jxl_lossless,
        jxl_distance=conf.jxl_distance,
        jxl_effort=conf.jxl_effort,
        webp_quality=conf.webp_quality,
        webp_lossless=conf.webp_lossless,
        webp_method=conf.webp_method,
        export_resolution_mode=conf.export_resolution_mode,
        paper_aspect_ratio=conf.paper_aspect_ratio,
        export_print_size=conf.export_print_size,
        export_dpi=conf.export_dpi,
        export_target_long_edge_px=conf.export_target_long_edge_px,
        output_mode=conf.output_mode,
        output_subfolder=conf.output_subfolder,
        output_path=conf.export_path,
        overwrite=conf.overwrite,
        filename_pattern=conf.filename_pattern,
        export_color_space=conf.export_color_space,
        icc_input_path=conf.icc_input_path,
        icc_output_path=conf.icc_output_path,
    )


@dataclass(frozen=True)
class WorkspaceConfig:
    """
    Complete state for a single image edit.
    """

    process: ProcessConfig = field(default_factory=ProcessConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    flatfield: FlatFieldConfig = field(default_factory=FlatFieldConfig)
    rgbscan: RgbScanConfig = field(default_factory=RgbScanConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    lab: LabConfig = field(default_factory=LabConfig)
    local: LocalAdjustmentsConfig = field(default_factory=LocalAdjustmentsConfig)
    retouch: RetouchConfig = field(default_factory=RetouchConfig)
    toning: ToningConfig = field(default_factory=ToningConfig)
    finish: FinishConfig = field(default_factory=FinishConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> Dict[str, Any]:
        """
        Flattens for serialization.
        """
        res = {}
        res.update(asdict(self.process))
        res.update(asdict(self.exposure))
        res.update(asdict(self.flatfield))
        res.update(asdict(self.rgbscan))
        res.update(asdict(self.geometry))
        res.update(asdict(self.lab))
        res["local_masks"] = asdict(self.local)
        res.update(asdict(self.retouch))
        res.update(asdict(self.toning))
        res.update(asdict(self.finish))
        res.update(asdict(self.metadata))
        res.update(asdict(self.export))
        return res

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> "WorkspaceConfig":
        """
        from DB/JSON.
        """

        local_data = data.pop("local_masks", {})

        # Apply field renames for backward compatibility.
        for old_key, new_key in MIGRATIONS.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)

        # True Black (BPC on) renamed to Paper Black with inverted polarity.
        if "true_black" in data:
            data["paper_black"] = not bool(data.pop("true_black"))

        # Single roll-average toggle split into independent luma + colour axes.
        if "use_roll_average" in data:
            legacy = bool(data.pop("use_roll_average"))
            data.setdefault("use_luma_average", legacy)
            data.setdefault("use_colour_average", legacy)

        # Lab "Separation" moved to the capture side (ProcessConfig crosstalk):
        # the 1.0–2.0 slider maps to strength 0–1. crosstalk_matrix/crosstalk_profile
        # keep their key names and re-route to ProcessConfig by field membership;
        # the old serialized DEFAULT_MATRIX field is dropped.
        if "color_separation" in data and "crosstalk_strength" not in data:
            data["crosstalk_strength"] = min(max(float(data.pop("color_separation")) - 1.0, 0.0), 1.0)
        data.pop("color_separation", None)
        data.pop("DEFAULT_MATRIX", None)
        # Vignette became an exposure-domain burn: old ±1 strength (neg = darken)
        # maps to stops (pos = burn) with an approximate look-preserving factor.
        if "vignette_strength" in data and "vignette_stops" not in data:
            data["vignette_stops"] = -2.0 * float(data.pop("vignette_strength"))
        data.pop("vignette_strength", None)
        # Flare (veiling-glare floor) was removed; drop the key from old edits silently.
        data.pop("flare", None)
        # IR stroke synthesis was replaced by the score-weighted fill; its pad radius is gone.
        data.pop("ir_inpaint_radius", None)
        # Auto cast removal became always-on; drop the old toggle key.
        data.pop("auto_cast_removal", None)

        # Filed carrier's separate on/off toggle was folded into carrier_width itself
        # (0 = off) in #542. A pre-#542 save always serialized both keys together, so
        # if the toggle was off, force width to 0 too — otherwise the leftover numeric
        # width (2.0 by the old default) reads as "on" under the new width>0 gating,
        # silently re-enabling a carrier the user had switched off.
        if "carrier_enabled" in data and not data.pop("carrier_enabled"):
            data["carrier_width"] = 0.0

        if "use_original_res" in data and "export_resolution_mode" not in data:
            data["export_resolution_mode"] = (
                ExportResolutionMode.ORIGINAL.value if data.pop("use_original_res") else ExportResolutionMode.PRINT.value
            )
        else:
            data.pop("use_original_res", None)

        if "same_as_source" in data and "output_mode" not in data:
            data["output_mode"] = ExportPresetOutputMode.SAME_AS_SOURCE if data.pop("same_as_source") else ExportPresetOutputMode.ABSOLUTE
        else:
            data.pop("same_as_source", None)

        # Metadata: migrate legacy combined override fields to structured gear fields.
        if data.get("camera_override") and not data.get("camera_model"):
            co = str(data.pop("camera_override", "")).strip()
            if co and not data.get("camera_make"):
                parts = co.split(None, 1)
                if len(parts) == 2:
                    data["camera_make"], data["camera_model"] = parts[0], parts[1]
                else:
                    data["camera_model"] = co
            elif co:
                data["camera_model"] = co
        else:
            data.pop("camera_override", None)
        if data.get("lens_override") and not data.get("lens_model"):
            data["lens_model"] = str(data.pop("lens_override", "")).strip()
        else:
            data.pop("lens_override", None)

        config_classes = [
            ProcessConfig,
            ExposureConfig,
            FlatFieldConfig,
            RgbScanConfig,
            GeometryConfig,
            LabConfig,
            RetouchConfig,
            ToningConfig,
            FinishConfig,
            MetadataConfig,
            ExportConfig,
        ]
        valid_keys = set()
        for cc in config_classes:
            valid_keys.update(cc.__dataclass_fields__.keys())

        # Fields deliberately removed from the config over time. Every edit saved
        # before the removal still carries them, so they'd otherwise warn on every
        # file load; drop silently and keep the warning for truly unknown keys.
        #   surround: dim-surround print gamma, removed in #432 (replaced by Snap).
        legacy_keys = {"surround"}

        unknown = set(data) - valid_keys - legacy_keys
        if unknown:
            logger.warning("Dropping unknown config keys: %s", sorted(unknown))

        def filter_keys(config_cls: Any, d: Dict[str, Any]) -> Dict[str, Any]:
            valid = config_cls.__dataclass_fields__.keys()
            return {k: v for k, v in d.items() if k in valid}

        def _build_local(d: Dict[str, Any]) -> LocalAdjustmentsConfig:
            masks = []
            for m in d.get("masks", []):
                verts = tuple(tuple(v) for v in m.get("vertices", []))
                masks.append(
                    PolygonMask(
                        vertices=verts,
                        strength=float(m.get("strength", 0.3)),
                        feather=float(m.get("feather", 0.04)),
                    )
                )
            return LocalAdjustmentsConfig(masks=tuple(masks))

        return cls(
            process=ProcessConfig(**filter_keys(ProcessConfig, data)),
            exposure=ExposureConfig(**filter_keys(ExposureConfig, data)),
            flatfield=FlatFieldConfig(**filter_keys(FlatFieldConfig, data)),
            rgbscan=RgbScanConfig(**filter_keys(RgbScanConfig, data)),
            geometry=GeometryConfig(**filter_keys(GeometryConfig, data)),
            lab=LabConfig(**filter_keys(LabConfig, data)),
            local=_build_local(local_data),
            retouch=RetouchConfig(**filter_keys(RetouchConfig, data)),
            toning=ToningConfig(**filter_keys(ToningConfig, data)),
            finish=FinishConfig(**filter_keys(FinishConfig, data)),
            metadata=MetadataConfig(**filter_keys(MetadataConfig, data)),
            export=ExportConfig(**filter_keys(ExportConfig, data)),
        )


def flat_master_config(config: WorkspaceConfig) -> WorkspaceConfig:
    """
    Derive a flat digital-intermediate ("Flat — for editing elsewhere") render
    config from an edit, without mutating it.

    Keeps the framing (geometry/crop), process mode and normalization bounds, and
    any explicit global white balance, but switches the Print stage to the flat
    render intent and turns off every automatic/creative print decision so the
    result is neutral, low-contrast and consistent across a roll. The creative
    stages (lab, local, toning, finish, retouch) are bypassed by the engine when
    the flat intent is set, so their values here are left untouched.
    """
    flat_exposure = replace(
        config.exposure,
        render_intent=RenderIntent.FLAT,
        auto_exposure=False,
        auto_normalize_contrast=False,
        cast_removal_strength=0.0,
        paper_dmin=False,
        toe=0.0,
        shoulder=0.0,
        toe_trim_red=0.0,
        toe_trim_green=0.0,
        toe_trim_blue=0.0,
        shoulder_trim_red=0.0,
        shoulder_trim_green=0.0,
        shoulder_trim_blue=0.0,
        toe_width_trim_red=0.0,
        toe_width_trim_green=0.0,
        toe_width_trim_blue=0.0,
        shoulder_width_trim_red=0.0,
        shoulder_width_trim_green=0.0,
        shoulder_width_trim_blue=0.0,
        grade_trim_red=0.0,
        grade_trim_green=0.0,
        grade_trim_blue=0.0,
        paper_black=True,
        midtone_gamma=0.0,
        midtone_gamma_trim_red=0.0,
        midtone_gamma_trim_green=0.0,
        midtone_gamma_trim_blue=0.0,
    )
    return replace(config, exposure=flat_exposure)


def flat_export_config(export: ExportConfig | ExportPreset, fmt: str = ExportFormat.TIFF) -> Any:
    """
    Override export settings for a flat master. Works on ``ExportConfig`` or
    ``ExportPreset`` — both share the same sizing/format field names — and
    returns the same type it was given.

    Always sets ``export_fmt`` (TIFF 16-bit or DNG). Resolution defaults to
    full original size; if the user explicitly chose Print or Pixels sizing in
    the export panel, those settings are honoured so flat masters can be
    downscaled when requested.
    """
    overrides: Dict[str, Any] = {"export_fmt": fmt}
    if export.export_resolution_mode not in (
        ExportResolutionMode.PRINT.value,
        ExportResolutionMode.TARGET_PX.value,
    ):
        overrides["export_resolution_mode"] = ExportResolutionMode.ORIGINAL.value
        overrides["paper_aspect_ratio"] = AspectRatio.ORIGINAL
    return replace(export, **overrides)


def flat_export_preset(preset: ExportPreset) -> ExportPreset:
    """Apply flat-master delivery rules to a preset (TIFF/DNG, default full-res)."""
    fmt = preset.export_fmt if preset.export_fmt in (ExportFormat.TIFF, ExportFormat.DNG) else ExportFormat.TIFF
    return flat_export_config(preset, fmt=fmt)


def resolve_preset_export(
    preset: ExportPreset,
    params: WorkspaceConfig,
) -> tuple[WorkspaceConfig, ExportPreset]:
    """Return (pipeline_config, delivery_preset) for one preset-driven export task."""
    if preset.render_intent != RenderIntent.FLAT:
        return params, copy.copy(preset)
    return flat_master_config(params), flat_export_preset(preset)
