import os

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from enum import Enum, StrEnum
from negpy.features.process.models import ProcessConfig
from negpy.features.exposure.models import ExposureConfig
from negpy.features.geometry.models import GeometryConfig
from negpy.features.lab.models import LabConfig
from negpy.features.retouch.models import RetouchConfig
from negpy.features.toning.models import ToningConfig
from negpy.features.finish.models import FinishConfig
from negpy.features.flatfield.models import FlatFieldConfig
from negpy.features.metadata.models import MetadataConfig
from negpy.kernel.system.logging import get_logger
import negpy.kernel.system.paths as paths

logger = get_logger("domain.models")

# Map of old field names → new field names for backward-compatible deserialization.
# Add entries here when fields are renamed so old workspace files keep their data.
MIGRATIONS: Dict[str, str] = {
    "export_border_size": "border_size",
    "export_border_color": "border_color",
    # Shadow-neutral + density-balance consolidated into Cast Removal. Preserve a
    # user's saved on/off; the unpublished "crossover"/"density_balance" keys are
    # just dropped as unknown (default cast_removal=True).
    "auto_shadow_neutral": "cast_removal",
}


class AspectRatio(StrEnum):
    FREE = "Free"
    ORIGINAL = "Original"
    R_3_2 = "3:2"
    R_4_3 = "4:3"
    R_5_4 = "5:4"
    R_6_7 = "6:7"
    R_1_1 = "1:1"
    R_65_24 = "65:24"
    # Verticals
    R_2_3 = "2:3"
    R_3_4 = "3:4"
    R_4_5 = "4:5"
    R_7_6 = "7:6"
    R_24_65 = "24:65"


class ExportFormat(StrEnum):
    JPEG = "JPEG"
    TIFF = "TIFF"


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


@dataclass(frozen=True)
class ExportConfig:
    """
    Export parameters (path, format, sizing).
    """

    userDir: str = field(default_factory=paths.get_default_user_dir)

    export_path: str = field(default_factory=lambda: os.path.join(paths.get_default_user_dir(), "export"))
    export_fmt: str = ExportFormat.JPEG
    export_color_space: str = ColorSpace.ADOBE_RGB.value
    paper_aspect_ratio: str = AspectRatio.ORIGINAL
    export_print_size: float = 30.0
    export_dpi: int = 300
    export_resolution_mode: str = ExportResolutionMode.PRINT.value
    export_target_long_edge_px: int = 2000
    filename_pattern: str = "{{ original_name }}"
    overwrite: bool = True
    same_as_source: bool = False
    icc_input_path: Optional[str] = None
    icc_output_path: Optional[str] = None


@dataclass(frozen=True)
class WorkspaceConfig:
    """
    Complete state for a single image edit.
    """

    process: ProcessConfig = field(default_factory=ProcessConfig)
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    flatfield: FlatFieldConfig = field(default_factory=FlatFieldConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    lab: LabConfig = field(default_factory=LabConfig)
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
        res.update(asdict(self.geometry))
        res.update(asdict(self.lab))
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

        # Apply field renames for backward compatibility.
        for old_key, new_key in MIGRATIONS.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)

        if "use_original_res" in data and "export_resolution_mode" not in data:
            data["export_resolution_mode"] = (
                ExportResolutionMode.ORIGINAL.value if data.pop("use_original_res") else ExportResolutionMode.PRINT.value
            )
        else:
            data.pop("use_original_res", None)

        config_classes = [
            ProcessConfig,
            ExposureConfig,
            FlatFieldConfig,
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

        unknown = set(data) - valid_keys
        if unknown:
            logger.warning("Dropping unknown config keys: %s", sorted(unknown))

        def filter_keys(config_cls: Any, d: Dict[str, Any]) -> Dict[str, Any]:
            valid = config_cls.__dataclass_fields__.keys()
            return {k: v for k, v in d.items() if k in valid}

        return cls(
            process=ProcessConfig(**filter_keys(ProcessConfig, data)),
            exposure=ExposureConfig(**filter_keys(ExposureConfig, data)),
            flatfield=FlatFieldConfig(**filter_keys(FlatFieldConfig, data)),
            geometry=GeometryConfig(**filter_keys(GeometryConfig, data)),
            lab=LabConfig(**filter_keys(LabConfig, data)),
            retouch=RetouchConfig(**filter_keys(RetouchConfig, data)),
            toning=ToningConfig(**filter_keys(ToningConfig, data)),
            finish=FinishConfig(**filter_keys(FinishConfig, data)),
            metadata=MetadataConfig(**filter_keys(MetadataConfig, data)),
            export=ExportConfig(**filter_keys(ExportConfig, data)),
        )
