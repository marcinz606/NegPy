from typing import Protocol, Any, Optional, Dict
import streamlit as st
from dataclasses import dataclass
from src.features.exposure.models import ExposureConfig
from src.features.geometry.models import GeometryConfig
from src.features.toning.models import ToningConfig
from src.features.lab.models import LabConfig
from src.features.retouch.models import RetouchConfig
from src.kernel.image.validation import validate_float, validate_int, validate_bool


@dataclass
class SidebarState:
    """
    Transient state for sidebar & export buttons.
    """

    out_fmt: str = "JPEG"
    color_space: str = "sRGB"
    print_width: float = 27.0
    print_dpi: int = 300
    export_path: str = "export"
    add_border: bool = True
    border_size: float = 0.25
    border_color: str = "#ffffff"
    filename_pattern: str = "positive_{{ original_name }}"
    apply_icc: bool = False
    process_btn: bool = False
    export_btn: bool = False


class IViewModel(Protocol):
    """
    Interface for UI-to-Session syncing.
    """

    def get_key(self, field: str) -> str: ...
    def load_from_state(self) -> Any: ...


class BaseViewModel:
    def __init__(self, data_source: Optional[Dict[str, Any]] = None):
        # If no data_source is provided, we use st.session_state
        self._data = data_source if data_source is not None else st.session_state

    def _get_raw(self, key: str, default: Any) -> Any:
        try:
            return self._data.get(key, default)
        except AttributeError:
            return (
                getattr(self._data, key, default)
                if hasattr(self._data, key)
                else default
            )

    def _get_float(self, key: str, default: float = 0.0) -> float:
        return validate_float(self._get_raw(key, default), default)

    def _get_int(self, key: str, default: int = 0) -> int:
        return validate_int(self._get_raw(key, default), default)

    def _get_bool(self, key: str, default: bool = False) -> bool:
        return validate_bool(self._get_raw(key, default), default)

    def _get_str(self, key: str, default: str = "") -> str:
        return str(self._get_raw(key, default))


class ExposureViewModel(BaseViewModel):
    def __init__(self, data_source: Optional[Dict[str, Any]] = None):
        super().__init__(data_source)
        # Internal mapping of domain field -> session state key
        self._keys = {
            "density": "density",
            "grade": "grade",
            "wb_cyan": "wb_cyan",
            "wb_magenta": "wb_magenta",
            "wb_yellow": "wb_yellow",
            "toe": "toe",
            "toe_width": "toe_width",
            "toe_hardness": "toe_hardness",
            "shoulder": "shoulder",
            "shoulder_width": "shoulder_width",
            "shoulder_hardness": "shoulder_hardness",
        }

    def get_key(self, field_name: str) -> str:
        return self._keys.get(field_name, field_name)

    @property
    def density(self) -> float:
        return self._get_float(self.get_key("density"), 1.0)

    @property
    def grade(self) -> float:
        return self._get_float(self.get_key("grade"), 2.5)

    @property
    def is_bw(self) -> bool:
        return self._get_str("process_mode", "C41") == "B&W"

    def to_config(self) -> ExposureConfig:
        return ExposureConfig(
            density=self.density,
            grade=self.grade,
            wb_cyan=self._get_float(self.get_key("wb_cyan")),
            wb_magenta=self._get_float(self.get_key("wb_magenta")),
            wb_yellow=self._get_float(self.get_key("wb_yellow")),
            toe=self._get_float(self.get_key("toe")),
            toe_width=self._get_float(self.get_key("toe_width"), 3.0),
            toe_hardness=self._get_float(self.get_key("toe_hardness"), 1.0),
            shoulder=self._get_float(self.get_key("shoulder")),
            shoulder_width=self._get_float(self.get_key("shoulder_width"), 3.0),
            shoulder_hardness=self._get_float(self.get_key("shoulder_hardness"), 1.0),
        )


class GeometryViewModel(BaseViewModel):
    def __init__(self, data_source: Optional[Dict[str, Any]] = None):
        super().__init__(data_source)
        self._keys = {
            "rotation": "rotation",
            "fine_rotation": "fine_rotation",
            "autocrop": "autocrop",
            "autocrop_offset": "autocrop_offset",
            "autocrop_ratio": "autocrop_ratio",
            "autocrop_assist_point": "autocrop_assist_point",
            "autocrop_assist_luma": "autocrop_assist_luma",
            "pick_assist": "pick_assist",
            "keep_full_frame": "keep_full_frame",
        }

    def get_key(self, field_name: str) -> str:
        return self._keys.get(field_name, field_name)

    def to_config(self) -> GeometryConfig:
        return GeometryConfig(
            rotation=self._get_int(self.get_key("rotation")),
            fine_rotation=self._get_float(self.get_key("fine_rotation")),
            autocrop=self._get_bool(self.get_key("autocrop"), True),
            autocrop_offset=self._get_int(self.get_key("autocrop_offset"), 2),
            autocrop_ratio=self._get_str(self.get_key("autocrop_ratio"), "3:2"),
            autocrop_assist_point=self._get_raw(
                self.get_key("autocrop_assist_point"), None
            ),
            autocrop_assist_luma=self._get_raw(
                self.get_key("autocrop_assist_luma"), None
            ),
            keep_full_frame=self._get_bool(self.get_key("keep_full_frame"), False),
        )


class ToningViewModel(BaseViewModel):
    def __init__(self, data_source: Optional[Dict[str, Any]] = None):
        super().__init__(data_source)
        self._keys = {
            "paper_profile": "paper_profile",
            "selenium_strength": "selenium_strength",
            "sepia_strength": "sepia_strength",
        }

    def get_key(self, field_name: str) -> str:
        return self._keys.get(field_name, field_name)

    def to_config(self) -> ToningConfig:
        return ToningConfig(
            paper_profile=self._get_str(self.get_key("paper_profile"), "None"),
            selenium_strength=self._get_float(self.get_key("selenium_strength")),
            sepia_strength=self._get_float(self.get_key("sepia_strength")),
        )


class LabViewModel(BaseViewModel):
    def __init__(self, data_source: Optional[Dict[str, Any]] = None):
        super().__init__(data_source)
        self._keys = {
            "color_separation": "color_separation",
            "clahe_strength": "clahe_strength",
            "sharpen": "sharpen",
            "crosstalk_matrix": "crosstalk_matrix",
        }

    def get_key(self, field_name: str) -> str:
        return self._keys.get(field_name, field_name)

    def to_config(self) -> LabConfig:
        crosstalk = self._get_raw(self.get_key("crosstalk_matrix"), None)
        if crosstalk is not None and not isinstance(crosstalk, list):
            crosstalk = None

        return LabConfig(
            color_separation=self._get_float(self.get_key("color_separation"), 1.0),
            clahe_strength=self._get_float(self.get_key("clahe_strength")),
            sharpen=self._get_float(self.get_key("sharpen"), 0.25),
            crosstalk_matrix=crosstalk,
        )


class RetouchViewModel(BaseViewModel):
    def __init__(self, data_source: Optional[Dict[str, Any]] = None):
        super().__init__(data_source)
        self._keys = {
            "dust_remove": "dust_remove",
            "dust_threshold": "dust_threshold",
            "dust_size": "dust_size",
            "manual_dust_spots": "manual_dust_spots",
            "manual_dust_size": "manual_dust_size",
            "pick_dust": "pick_dust",
            "dust_scratch_mode": "dust_scratch_mode",
            "show_dust_patches": "show_dust_patches",
            "local_adjustments": "local_adjustments",
        }

    def get_key(self, field_name: str) -> str:
        return self._keys.get(field_name, field_name)

    def to_config(self) -> RetouchConfig:
        return RetouchConfig(
            dust_remove=self._get_bool(self.get_key("dust_remove"), True),
            dust_threshold=self._get_float(self.get_key("dust_threshold"), 0.75),
            dust_size=self._get_int(self.get_key("dust_size"), 2),
            manual_dust_spots=list(
                self._get_raw(self.get_key("manual_dust_spots"), [])
            ),
            manual_dust_size=self._get_int(self.get_key("manual_dust_size"), 5),
            local_adjustments=list(
                self._get_raw(self.get_key("local_adjustments"), [])
            ),
        )
