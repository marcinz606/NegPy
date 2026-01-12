from typing import Optional, Dict, Any, Literal, Callable
from src.features.exposure.logic import density_to_cmy
import streamlit as st
import numpy as np
from src.ui.state.state_manager import save_settings


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


def _ensure_and_get_state(
    key: str, default_val: Any, cast_func: Callable[[Any], Any]
) -> Any:
    """
    Internal helper:
    1. Recovers state from Domain Session if missing (partial session loss).
    2. Initializes default if still missing.
    3. Safely casts and returns the current canonical value.
    """
    if st.session_state.get(key) is None:
        session = st.session_state.get("session")
        if session:
            try:
                active_settings = session.get_active_settings()
                if active_settings:
                    val = active_settings.to_dict().get(key)
                    if val is not None:
                        st.session_state[key] = cast_func(val)
            except Exception:
                pass

    if st.session_state.get(key) is None:
        st.session_state[key] = default_val

    try:
        return cast_func(st.session_state[key])
    except (ValueError, TypeError):
        # Fallback to default if cast fails
        safe_val = cast_func(default_val)
        st.session_state[key] = safe_val
        return safe_val


def _sync_shadow_state(key: str, current_val: Any) -> str:
    """
    Internal helper:
    Synchronizes the canonical value to the shadow key (`w_key`) used by the widget.
    Returns the shadow key name.
    """
    w_key = f"w_{key}"
    last_key = f"last_{key}"

    # Sync only if mismatch or missing, to avoid extraneous updates
    if st.session_state.get(last_key) != current_val or w_key not in st.session_state:
        st.session_state[w_key] = current_val
        st.session_state[last_key] = current_val

    return w_key


def _update_canonical_state(key: str, new_val: Any, old_val: Any) -> Any:
    """
    Internal helper:
    Updates the canonical session state if the widget result differs from the old value.
    Returns the updated (or original) value.
    """
    if new_val is not None and new_val != old_val:
        st.session_state[key] = new_val
        st.session_state[f"last_{key}"] = new_val
        return new_val
    return old_val


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
    Standard sidebar slider. Handles state sync.
    """
    current_val = _ensure_and_get_state(key, default_val, float)

    current_val = float(np.clip(current_val, min_val, max_val))

    w_key = _sync_shadow_state(key, current_val)

    res = st.slider(
        label,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(st.session_state[w_key]),
        step=float(step),
        format=format,
        key=w_key,
        help=help_text,
        disabled=disabled,
    )

    return float(
        _update_canonical_state(
            key, float(res) if res is not None else None, current_val
        )
    )


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
    """
    current_val = _ensure_and_get_state(key, default_val, bool)
    w_key = _sync_shadow_state(key, current_val)

    if is_toggle:
        res = st.toggle(
            label,
            value=bool(st.session_state[w_key]),
            key=w_key,
            help=help_text,
            disabled=disabled,
            label_visibility=label_visibility,
        )
    else:
        res = st.checkbox(
            label,
            value=bool(st.session_state[w_key]),
            key=w_key,
            help=help_text,
            disabled=disabled,
            label_visibility=label_visibility,
        )

    return bool(_update_canonical_state(key, res, current_val))


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
    """
    current_val = _ensure_and_get_state(key, default_val, lambda x: x)  # No-op cast
    w_key = _sync_shadow_state(key, current_val)

    try:
        idx = options.index(current_val)
    except ValueError:
        idx = 0

    res = st.selectbox(
        label,
        options=options,
        index=idx,
        key=w_key,
        help=help_text,
        disabled=disabled,
        format_func=format_func,
        on_change=on_change,
        args=args,
        kwargs=kwargs,
        label_visibility=label_visibility,
    )

    return _update_canonical_state(key, res, current_val)


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
    """
    current_val = _ensure_and_get_state(key, default_val, str)
    w_key = _sync_shadow_state(key, current_val)

    res = st.text_input(
        label,
        value=str(st.session_state[w_key]),
        key=w_key,
        help=help_text,
        disabled=disabled,
        placeholder=placeholder,
        type=type,
        label_visibility=label_visibility,
    )

    return str(_update_canonical_state(key, res, current_val))


def render_control_color_picker(
    label: str,
    default_val: str,
    key: str,
    help_text: Optional[str] = None,
    disabled: bool = False,
) -> str:
    """
    Standardized color_picker renderer for the sidebar.
    """
    current_val = _ensure_and_get_state(key, default_val, str)

    if not (current_val.startswith("#") and len(current_val) == 7):
        current_val = str(default_val)
        st.session_state[key] = current_val

    w_key = _sync_shadow_state(key, current_val)

    res = st.color_picker(
        label,
        value=str(st.session_state[w_key]),
        key=w_key,
        help=help_text,
        disabled=disabled,
    )

    return str(_update_canonical_state(key, res, current_val))


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
