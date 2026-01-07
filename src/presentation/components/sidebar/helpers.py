from typing import Optional, Dict, Any, Literal
from src.features.exposure.logic import density_to_cmy
import streamlit as st
import numpy as np
from src.presentation.state.state_manager import save_settings


def reset_wb_settings() -> None:
    """
    Resets Cyan, Magenta, and Yellow sliders to 0.
    """
    st.session_state.wb_cyan = 0.0
    st.session_state.wb_magenta = 0.0
    st.session_state.wb_yellow = 0.0
    save_settings()


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
    Uses a shadow-key sync pattern to avoid 'double-set' warnings while ensuring
    backend updates (Reset, Auto-WB) are reflected in the UI.
    """
    # 1. Ensure canonical state exists
    if key not in st.session_state:
        st.session_state[key] = default_val

    # 2. Clamping/Sanity check on canonical state
    current_val = float(st.session_state[key])
    if current_val < float(min_val):
        current_val = float(min_val)
        st.session_state[key] = current_val
    elif current_val > float(max_val):
        current_val = float(max_val)
        st.session_state[key] = current_val

    # 3. Synchronize to Shadow Key (Widget State)
    # We use a shadow key for the widget to avoid Session State API conflicts.
    w_key = f"w_{key}"
    last_key = f"last_{key}"

    # If canonical value changed from outside (e.g. Reset or Auto-WB), force sync to widget
    if st.session_state.get(last_key) != current_val:
        st.session_state[w_key] = current_val
        st.session_state[last_key] = current_val

    # 4. Render Slider (NO explicit value argument to avoid warnings)
    # Streamlit automatically uses st.session_state[w_key]
    res = st.slider(
        label,
        min_value=float(min_val),
        max_value=float(max_val),
        step=float(step),
        format=format,
        key=w_key,
        help=help_text,
        disabled=disabled,
    )

    # 5. Sync back to canonical state
    if float(res) != current_val:
        st.session_state[key] = float(res)
        st.session_state[last_key] = float(res)

    return float(st.session_state[key])


def render_control_checkbox(
    label: str,
    default_val: bool,
    key: str,
    help_text: Optional[str] = None,
    disabled: bool = False,
    is_toggle: bool = False,
    label_visibility: Literal["visible", "hidden", "collapsed"] = "visible",
) -> bool:
    """
    Standardized checkbox renderer for the sidebar.
    Uses a shadow-key sync pattern.
    """
    if key not in st.session_state:
        st.session_state[key] = default_val

    current_val = bool(st.session_state[key])
    w_key = f"w_{key}"
    last_key = f"last_{key}"

    if st.session_state.get(last_key) != current_val:
        st.session_state[w_key] = current_val
        st.session_state[last_key] = current_val

    if is_toggle:
        res = st.toggle(
            label,
            key=w_key,
            help=help_text,
            disabled=disabled,
            label_visibility=label_visibility,
        )
    else:
        res = st.checkbox(
            label,
            key=w_key,
            help=help_text,
            disabled=disabled,
            label_visibility=label_visibility,
        )

    if res != current_val:
        st.session_state[key] = res
        st.session_state[last_key] = res

    return bool(st.session_state[key])


def render_control_selectbox(
    label: str,
    options: list,
    default_val: Any,
    key: str,
    help_text: Optional[str] = None,
    disabled: bool = False,
    format_func: Any = str,
    on_change: Optional[Any] = None,
    args: Optional[tuple] = None,
    kwargs: Optional[dict] = None,
    label_visibility: Literal["visible", "hidden", "collapsed"] = "visible",
) -> Any:
    """
    Standardized selectbox renderer for the sidebar.
    Uses a shadow-key sync pattern.
    """
    if key not in st.session_state:
        st.session_state[key] = default_val

    current_val = st.session_state[key]
    w_key = f"w_{key}"
    last_key = f"last_{key}"

    if st.session_state.get(last_key) != current_val:
        st.session_state[w_key] = current_val
        st.session_state[last_key] = current_val

    res = st.selectbox(
        label,
        options=options,
        key=w_key,
        help=help_text,
        disabled=disabled,
        format_func=format_func,
        on_change=on_change,
        args=args,
        kwargs=kwargs,
        label_visibility=label_visibility,
    )

    if res != current_val:
        st.session_state[key] = res
        st.session_state[last_key] = res

    return st.session_state[key]


def render_control_text_input(
    label: str,
    default_val: str,
    key: str,
    help_text: Optional[str] = None,
    disabled: bool = False,
    placeholder: str = "",
    type: Literal["default", "password"] = "default",
    label_visibility: Literal["visible", "hidden", "collapsed"] = "visible",
) -> str:
    """
    Standardized text_input renderer for the sidebar.
    Uses a shadow-key sync pattern.
    """
    if key not in st.session_state:
        st.session_state[key] = default_val

    current_val = str(st.session_state[key])
    w_key = f"w_{key}"
    last_key = f"last_{key}"

    if st.session_state.get(last_key) != current_val:
        st.session_state[w_key] = current_val
        st.session_state[last_key] = current_val

    res = st.text_input(
        label,
        key=w_key,
        help=help_text,
        disabled=disabled,
        placeholder=placeholder,
        type=type,
        label_visibility=label_visibility,
    )

    if res != current_val:
        st.session_state[key] = res
        st.session_state[last_key] = res

    return str(st.session_state[key])


def render_control_color_picker(
    label: str,
    default_val: str,
    key: str,
    help_text: Optional[str] = None,
    disabled: bool = False,
) -> str:
    """
    Standardized color_picker renderer for the sidebar.
    Uses a shadow-key sync pattern.
    """
    if key not in st.session_state:
        st.session_state[key] = default_val

    current_val = str(st.session_state[key])
    w_key = f"w_{key}"
    last_key = f"last_{key}"

    if st.session_state.get(last_key) != current_val:
        st.session_state[w_key] = current_val
        st.session_state[last_key] = current_val

    res = st.color_picker(
        label,
        key=w_key,
        help=help_text,
        disabled=disabled,
    )

    if res != current_val:
        st.session_state[key] = res
        st.session_state[last_key] = res

    return str(st.session_state[key])


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
