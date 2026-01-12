from typing import Optional
import streamlit as st
from src.domain.types import ImageBuffer, Dimensions
from src.kernel.image.validation import (
    validate_int,
    validate_bool,
    ensure_image,
)
from src.domain.session import WorkspaceSession


class SessionContext:
    """
    Typed wrapper for st.session_state access.
    """

    @property
    def session(self) -> WorkspaceSession:
        session = st.session_state.session
        if not isinstance(session, WorkspaceSession):
            raise TypeError("Session is not a WorkspaceSession")
        return session

    @property
    def working_copy_size(self) -> int:
        return validate_int(st.session_state.get("working_copy_size", 1200), 1200)

    @working_copy_size.setter
    def working_copy_size(self, val: int) -> None:
        st.session_state["working_copy_size"] = val

    @property
    def preview_raw(self) -> Optional[ImageBuffer]:
        raw = st.session_state.get("preview_raw")
        if raw is None:
            return None
        return ensure_image(raw)

    @preview_raw.setter
    def preview_raw(self, val: ImageBuffer) -> None:
        st.session_state["preview_raw"] = val

    @property
    def original_res(self) -> Dimensions:
        res = st.session_state.get("original_res", (0, 0))
        return (validate_int(res[0]), validate_int(res[1]))

    @original_res.setter
    def original_res(self, val: Dimensions) -> None:
        st.session_state["original_res"] = val

    @property
    def last_file(self) -> Optional[str]:
        return st.session_state.get("last_file")

    @last_file.setter
    def last_file(self, val: str) -> None:
        st.session_state["last_file"] = val

    @property
    def last_preview_color_space(self) -> str:
        return str(st.session_state.get("last_preview_color_space", "sRGB"))

    @last_preview_color_space.setter
    def last_preview_color_space(self, val: str) -> None:
        st.session_state["last_preview_color_space"] = val

    @property
    def pick_dust(self) -> bool:
        return validate_bool(st.session_state.get("pick_dust", False))

    @property
    def pick_local(self) -> bool:
        return validate_bool(st.session_state.get("pick_local", False))

    @property
    def active_adjustment_idx(self) -> int:
        return validate_int(st.session_state.get("active_adjustment_idx", -1), -1)
