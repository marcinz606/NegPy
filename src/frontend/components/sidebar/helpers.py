import numpy as np
import streamlit as st
from typing import Optional, Dict, Any


def render_control_slider(
    label: str,
    min_val: float,
    max_val: float,
    default_val: float,
    step: float,
    key: str,
    help_text: Optional[str] = None,
    format: str = "%.2f",
) -> float:
    """
    Standardized slider renderer for the sidebar.
    Ensures session state is initialized and clamped.
    """
    if key not in st.session_state:
        st.session_state[key] = default_val
    else:
        # Safety Clamp
        st.session_state[key] = max(
            float(min_val), min(float(st.session_state[key]), float(max_val))
        )

    return float(
        st.slider(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(st.session_state[key]),
            step=float(step),
            format=format,
            key=key,
            help=help_text,
        )
    )


def apply_wb_gains_to_sliders(r: float, g: float, b: float) -> Dict[str, Any]:
    """
    Translates raw RGB gains (from Auto-WB) into CMY filtration (0-170).
    Cyan is anchored at 0. Magenta and Yellow are logarithmic (100 * log10(gain)).
    """
    # Auto-WB already returns gains relative to Red (r=1.0).
    # m = 100 * log10(g_gain), y = 100 * log10(b_gain)
    m = int(np.log10(max(g, 1.0)) * 100.0)
    y = int(np.log10(max(b, 1.0)) * 100.0)

    return {
        "wb_cyan": 0,
        "wb_magenta": int(np.clip(m, 0, 170)),
        "wb_yellow": int(np.clip(y, 0, 170)),
        "cr_balance": 1.0,
        "mg_balance": 1.0,
        "yb_balance": 1.0,
    }
