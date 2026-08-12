import copy
import os
import uuid

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Any, Optional, TypeVar
from enum import Enum, StrEnum
from negpy.features.process.models import ProcessConfig
from negpy.features.exposure.models import ExposureConfig, RenderIntent
from negpy.features.geometry.models import (  # noqa: F401  (re-exported: the crop ratios were defined here before they moved beside GeometryConfig)
    CROP_RATIO_CHOICES,
    FILM_FORMAT_RATIOS,
    AspectRatio,
    GeometryConfig,
    canonical_crop_ratio,
)
from negpy.features.lab.models import LabConfig
from negpy.features.local.models import LocalAdjustmentsConfig, LocalMask, MaskShape
from negpy.features.retouch.models import RetouchConfig
from negpy.features.altprocess.models import AltProcessConfig
from negpy.features.toning.models import ToningConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.rgbscan.models import RgbScanConfig
from negpy.features.hdr.models import HdrConfig
from negpy.features.stitch.models import StitchConfig
from negpy.features.metadata.models import MetadataConfig
from negpy.domain.migrations import migrate_export_fmt, migrate_flat_config
from negpy.kernel.system.logging import get_logger
import negpy.kernel.system.paths as paths

logger = get_logger("domain.models")


class ExportFormat(StrEnum):
    JPEG = "JPEG"
    TIFF = "TIFF"
    PNG = "PNG"
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


_EnumT = TypeVar("_EnumT", bound=Enum)


def coerce_enum(enum_cls: type[_EnumT], value: Any, default: _EnumT) -> _EnumT:
    """`enum_cls(value)`, falling back to `default` for a value no longer offered.

    Saved presets and edits carry whatever the app wrote at the time, so a retired
    or hand-edited value must degrade to the default instead of failing the load.
    """
    try:
        return enum_cls(value)
    except ValueError:
        return default


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


# Colour spaces offered for export. ACES and XYZ are rawpy *decode* spaces with no
# bundled ICC profile — an export targeting them can be neither converted nor tagged
# (the encoder falls back to the working space), so they're excluded here.
EXPORT_COLOR_SPACES: list[str] = [cs.value for cs in ColorSpace if cs not in (ColorSpace.ACES, ColorSpace.XYZ)]


# Colour spaces JPEG XL can tag (mirror _JXL_COLOR in image_processor). Same as
# Source is deliberately excluded: it resolves per-file at export time (usually to
# the Adobe RGB working space for scans/raws with no embedded profile), and Adobe
# RGB isn't JXL-taggable — so allowing it here would pass this upfront check and
# still hard-fail deep in the encoder. Blocking it here forces an explicit,
# taggable choice instead.
JXL_TAGGABLE_SPACES = frozenset(
    {
        ColorSpace.SRGB.value,
        ColorSpace.P3_D65.value,
        ColorSpace.REC2020.value,
        ColorSpace.GREYSCALE.value,
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
    export_fmt: ExportFormat = ExportFormat.JPEG
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
    export_resolution_mode: ExportResolutionMode = ExportResolutionMode.ORIGINAL
    export_target_long_edge_px: int = 2000
    filename_pattern: str = "{{ original_name }}"
    # When True, exports silently overwrite existing files; when False, the export
    # prompts (Overwrite / Rename / Cancel) before clobbering anything.
    overwrite: bool = False
    output_mode: ExportPresetOutputMode = ExportPresetOutputMode.ABSOLUTE
    output_subfolder: str = ""
    icc_input_path: Optional[str] = None
    icc_output_path: Optional[str] = None

    contact_sheet_cell_px: int = 600
    contact_sheet_gap: int = 16
    contact_sheet_margin: int = 32
    contact_sheet_max_tiles: int = 38
    contact_sheet_show_labels: bool = True
    contact_sheet_background_color: str = "#000000"
    contact_sheet_label_color: str = "#ffffff"
    contact_sheet_output_path: str = ""  # empty = follow export destination rules
    contact_sheet_template: str = ""  # empty = Default template active
    contact_sheet_default_cell_px: int = 600
    contact_sheet_default_gap: int = 16
    contact_sheet_default_margin: int = 32
    contact_sheet_default_max_tiles: int = 38
    contact_sheet_default_show_labels: bool = True
    contact_sheet_default_background_color: str = "#000000"
    contact_sheet_default_label_color: str = "#ffffff"

    export_sidecars_enabled: bool = False

    def __post_init__(self) -> None:
        fmt = coerce_enum(ExportFormat, migrate_export_fmt(self.export_fmt), ExportFormat.JPEG)
        object.__setattr__(self, "export_fmt", fmt)
        object.__setattr__(
            self,
            "export_resolution_mode",
            coerce_enum(ExportResolutionMode, self.export_resolution_mode, ExportResolutionMode.ORIGINAL),
        )
        object.__setattr__(self, "output_mode", coerce_enum(ExportPresetOutputMode, self.output_mode, ExportPresetOutputMode.ABSOLUTE))


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
    export_fmt: ExportFormat = ExportFormat.JPEG
    jpeg_quality: int = 90
    jxl_lossless: bool = True
    jxl_distance: float = 1.0
    jxl_effort: int = 7
    webp_quality: int = 90
    webp_lossless: bool = False
    webp_method: int = 4

    # Sizing (same field names as ExportConfig for PrintService compatibility)
    export_resolution_mode: ExportResolutionMode = ExportResolutionMode.ORIGINAL
    paper_aspect_ratio: str = AspectRatio.ORIGINAL
    export_print_size: float = 30.0
    export_dpi: int = 300
    export_target_long_edge_px: int = 2000

    # Output destination
    output_mode: ExportPresetOutputMode = ExportPresetOutputMode.SAME_AS_SOURCE
    output_subfolder: str = ""
    output_path: str = ""
    overwrite: bool = False
    filename_pattern: str = "{{ original_name }}"

    # Color
    export_color_space: str = ColorSpace.SRGB.value
    icc_input_path: Optional[str] = None
    icc_output_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.export_fmt = coerce_enum(ExportFormat, migrate_export_fmt(self.export_fmt), ExportFormat.JPEG)
        self.export_resolution_mode = coerce_enum(ExportResolutionMode, self.export_resolution_mode, ExportResolutionMode.ORIGINAL)
        self.output_mode = coerce_enum(ExportPresetOutputMode, self.output_mode, ExportPresetOutputMode.SAME_AS_SOURCE)

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
    stitch: StitchConfig = field(default_factory=StitchConfig)
    hdr: HdrConfig = field(default_factory=HdrConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    lab: LabConfig = field(default_factory=LabConfig)
    local: LocalAdjustmentsConfig = field(default_factory=LocalAdjustmentsConfig)
    retouch: RetouchConfig = field(default_factory=RetouchConfig)
    altproc: AltProcessConfig = field(default_factory=AltProcessConfig)
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
        res.update(asdict(self.stitch))
        res.update(asdict(self.hdr))
        res.update(asdict(self.geometry))
        res.update(asdict(self.lab))
        res["local_masks"] = asdict(self.local)
        res.update(asdict(self.retouch))
        res.update(asdict(self.altproc))
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

        data = migrate_flat_config(data)

        config_classes = [
            ProcessConfig,
            ExposureConfig,
            FlatFieldConfig,
            RgbScanConfig,
            StitchConfig,
            HdrConfig,
            GeometryConfig,
            LabConfig,
            RetouchConfig,
            AltProcessConfig,
            ToningConfig,
            FinishConfig,
            MetadataConfig,
            ExportConfig,
        ]
        valid_keys = set()
        for cc in config_classes:
            valid_keys.update(cc.__dataclass_fields__.keys())

        unknown = set(data) - valid_keys
        if unknown:
            logger.warning("Dropping unknown config keys: %s", sorted(unknown))

        def filter_keys(config_cls: Any, d: Dict[str, Any]) -> Dict[str, Any]:
            valid = config_cls.__dataclass_fields__.keys()
            return {k: v for k, v in d.items() if k in valid}

        def _build_local(d: Dict[str, Any]) -> LocalAdjustmentsConfig:
            masks = []
            for m in d.get("masks", []):
                verts = tuple(tuple(v) for v in m.get("vertices", []))
                # A mask's exposure was brightness-signed (`strength`, positive =
                # dodge) before it became exposure-signed stops (positive = burn) —
                # the same flip vignette_strength -> vignette_stops made. Nested in
                # local_masks, so it can't live in MIGRATIONS' flat rewrites; keyed
                # on the legacy name, which only a pre-flip save carries.
                stops = -float(m["strength"]) if "strength" in m else float(m.get("stops", 0.0))
                masks.append(
                    LocalMask(
                        vertices=verts,
                        stops=stops,
                        feather=float(m.get("feather", 0.04)),
                        grade=float(m.get("grade", 0.0)),
                        shape=MaskShape(m.get("shape", MaskShape.POLYGON)),
                        invert=bool(m.get("invert", False)),
                    )
                )
            return LocalAdjustmentsConfig(masks=tuple(masks))

        def _build_stitch(d: Dict[str, Any]) -> StitchConfig:
            # JSON round-trips tuples as lists; coerce back or the frozen config
            # loses hashability/equality (breaks config-diff caching).
            canvas = d.get("stitch_canvas", (0, 0))
            return StitchConfig(
                stitch_enabled=bool(d.get("stitch_enabled", False)),
                stitch_paths=tuple(d.get("stitch_paths", ())),
                stitch_transforms=tuple(tuple(float(v) for v in row) for row in d.get("stitch_transforms", ())),
                stitch_canvas=(int(canvas[0]), int(canvas[1])),
                stitch_sizes=tuple((int(s[0]), int(s[1])) for s in d.get("stitch_sizes", ())),
                stitch_triplets=tuple((str(t[0]), str(t[1])) for t in d.get("stitch_triplets", ())),
                stitch_align=bool(d.get("stitch_align", True)),
            )

        def _build_hdr(d: Dict[str, Any]) -> HdrConfig:
            # Tuples, for the same reason _build_stitch coerces: JSON gives back lists.
            return HdrConfig(
                hdr_enabled=bool(d.get("hdr_enabled", False)),
                hdr_paths=tuple(str(p) for p in d.get("hdr_paths", ())),
                hdr_ratios=tuple(float(r) for r in d.get("hdr_ratios", ())),
                hdr_align=bool(d.get("hdr_align", True)),
                hdr_anchor=str(d.get("hdr_anchor", "") or ""),
            )

        return cls(
            process=ProcessConfig(**filter_keys(ProcessConfig, data)),
            exposure=ExposureConfig(**filter_keys(ExposureConfig, data)),
            flatfield=FlatFieldConfig(**filter_keys(FlatFieldConfig, data)),
            rgbscan=RgbScanConfig(**filter_keys(RgbScanConfig, data)),
            stitch=_build_stitch(filter_keys(StitchConfig, data)),
            hdr=_build_hdr(filter_keys(HdrConfig, data)),
            geometry=GeometryConfig(**filter_keys(GeometryConfig, data)),
            lab=LabConfig(**filter_keys(LabConfig, data)),
            local=_build_local(local_data),
            retouch=RetouchConfig(**filter_keys(RetouchConfig, data)),
            altproc=AltProcessConfig(**filter_keys(AltProcessConfig, data)),
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


def flat_export_config(export: ExportConfig | ExportPreset) -> Any:
    """
    Override export settings for a flat master. Works on ``ExportConfig`` or
    ``ExportPreset`` — both share the same sizing/format field names — and
    returns the same type it was given.

    Delivers 16-bit lossless in TIFF (default) or JXL (if the user chose it
    and the colour space is JXL-taggable). Resolution defaults to full
    original size; if the user explicitly chose Print or Pixels sizing in the
    export panel, those settings are honoured so flat masters can be
    downscaled when requested.
    """
    fmt = export.export_fmt
    if fmt == ExportFormat.JXL and not export_blocked(fmt, export.export_color_space):
        overrides: Dict[str, Any] = {"jxl_lossless": True}
    else:
        overrides = {"export_fmt": ExportFormat.TIFF}
    if export.export_resolution_mode not in (
        ExportResolutionMode.PRINT.value,
        ExportResolutionMode.TARGET_PX.value,
    ):
        overrides["export_resolution_mode"] = ExportResolutionMode.ORIGINAL.value
        overrides["paper_aspect_ratio"] = AspectRatio.ORIGINAL
    return replace(export, **overrides)


def resolve_preset_export(
    preset: ExportPreset,
    params: WorkspaceConfig,
) -> tuple[WorkspaceConfig, ExportPreset]:
    """Return (pipeline_config, delivery_preset) for one preset-driven export task."""
    if preset.render_intent != RenderIntent.FLAT:
        return params, copy.copy(preset)
    return flat_master_config(params), flat_export_config(preset)
