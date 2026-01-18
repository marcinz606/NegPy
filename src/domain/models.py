from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from enum import Enum
from src.features.exposure.models import ExposureConfig
from src.features.geometry.models import GeometryConfig
from src.features.lab.models import LabConfig
from src.features.retouch.models import RetouchConfig
from src.features.toning.models import ToningConfig
from src.features.metadata.models import FilmMetadataConfig


class ICCMode(Enum):
    OUTPUT = "Output"
    INPUT = "Input"


class ColorSpace(Enum):
    SRGB = "sRGB"
    ADOBE_RGB = "Adobe RGB"
    PROPHOTO = "ProPhoto RGB"
    WIDE = "Wide Gamut RGB"
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

    export_path: str = "export"
    export_fmt: str = "JPEG"
    export_color_space: str = ColorSpace.ADOBE_RGB.value
    paper_aspect_ratio: str = "Original"
    export_print_size: float = 27.0
    export_dpi: int = 300
    export_add_border: bool = False
    export_border_size: float = 0.0
    export_border_color: str = "#ffffff"
    use_original_res: bool = False
    filename_pattern: str = "positive_{{ original_name }}"
    apply_icc: bool = False
    icc_profile_path: Optional[str] = None
    icc_invert: bool = False


@dataclass(frozen=True)
class WorkspaceConfig:
    """
    Complete state for a single image edit.
    """

    process_mode: str = "C41"
    exposure: ExposureConfig = field(default_factory=ExposureConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    lab: LabConfig = field(default_factory=LabConfig)
    retouch: RetouchConfig = field(default_factory=RetouchConfig)
    toning: ToningConfig = field(default_factory=ToningConfig)
    metadata: FilmMetadataConfig = field(default_factory=FilmMetadataConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> Dict[str, Any]:
        """
        Flattens for serialization.
        """
        res = {"process_mode": self.process_mode}
        res.update(asdict(self.exposure))
        res.update(asdict(self.geometry))
        res.update(asdict(self.lab))
        res.update(asdict(self.retouch))
        res.update(asdict(self.toning))
        res.update(asdict(self.metadata))
        res.update(asdict(self.export))
        return res

    @classmethod
    def from_flat_dict(cls, data: Dict[str, Any]) -> "WorkspaceConfig":
        """
        from DB/JSON.
        """

        def filter_keys(config_cls: Any, d: Dict[str, Any]) -> Dict[str, Any]:
            valid_keys = config_cls.__dataclass_fields__.keys()
            return {k: v for k, v in d.items() if k in valid_keys and v is not None}

        return cls(
            process_mode=str(data.get("process_mode", "C41")),
            exposure=ExposureConfig(**filter_keys(ExposureConfig, data)),
            geometry=GeometryConfig(**filter_keys(GeometryConfig, data)),
            lab=LabConfig(**filter_keys(LabConfig, data)),
            retouch=RetouchConfig(**filter_keys(RetouchConfig, data)),
            toning=ToningConfig(**filter_keys(ToningConfig, data)),
            metadata=FilmMetadataConfig(**filter_keys(FilmMetadataConfig, data)),
            export=ExportConfig(**filter_keys(ExportConfig, data)),
        )
