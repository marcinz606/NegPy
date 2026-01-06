from src.features.exposure.logic import density_to_cmy


import numpy as np
import streamlit as st
from typing import Optional, Dict, Any


def st_init(key: str, default_val: Any) -> Any:
    """
    Ensures a key is initialized in Streamlit session state.
    Returns the current value.
    """
    if key not in st.session_state:
        st.session_state[key] = default_val
    return st.session_state[key]


def render_control_slider(
    label: str,
    min_val: float,
    max_val: float,
    default_val: float,
    step: float,
    key: str,
    help_text: Optional[str] = None,
    format: str = "%.2f",
    disabled: bool = False,
) -> float:
    """
    Standardized slider renderer for the sidebar.
    """
    if key not in st.session_state:
        st.session_state[key] = default_val

    return float(
        st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            format=format,
            key=key,
            help=help_text,
            disabled=disabled,
        )
    )


def apply_wb_gains_to_sliders(r: float, g: float, b: float) -> Dict[str, Any]:
    """
    Translates raw RGB gains (from Auto-WB) into CMY filtration (-1.0 to 1.0).
    """
    c = density_to_cmy(np.log10(max(r, 1e-6)))
    m = density_to_cmy(np.log10(max(g, 1e-6)))
    y = density_to_cmy(np.log10(max(b, 1e-6)))

    return {
        "wb_cyan": float(np.clip(c, -1.0, 1.0)),
        "wb_magenta": float(np.clip(m, -1.0, 1.0)),
        "wb_yellow": float(np.clip(y, -1.0, 1.0)),
        "cr_balance": 1.0,
        "mg_balance": 1.0,
        "yb_balance": 1.0,
    }
