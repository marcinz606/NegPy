import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates
from typing import Tuple
from src.domain_objects import SidebarData
from src.logging_config import get_logger
from src.backend.image_logic.local_adjustments import (
    generate_local_mask,
    calculate_luma_mask,
)
from src.helpers import ensure_array
from src.backend.image_logic.geometry import get_autocrop_coords
from src.frontend.state import save_settings
from src.backend.session import DarkroomSession
from src.backend.image_logic.exposure import (
    prepare_exposure_analysis,
    analyze_sensitometry,
)
from src.config import PIPELINE_CONSTANTS

logger = get_logger(__name__)


def render_image_view(
    pil_prev: Image.Image, border_config: SidebarData | None = None
) -> None:
    """
    Renders the main image viewport and handles coordinate-based interaction.
    """
    session: DarkroomSession = st.session_state.session
    border_px = 0
    orig_w, orig_h = pil_prev.size

    # --- Border Preview ---
    if border_config and border_config.add_border:
        try:
            print_w = border_config.print_width
            border_w = border_config.border_size
            color = border_config.border_color

            # Calculate border width in pixels relative to current preview size
            longer_side_px = float(max(pil_prev.size))
            ratio = border_w / print_w if print_w > 0 else 0.01
            border_px = int(longer_side_px * ratio)

            if border_px > 0:
                pil_prev = ImageOps.expand(pil_prev, border=border_px, fill=color)
        except Exception as e:
            logger.error(f"Border preview error: {e}")

    # 1. State & Config
    is_local_mode = st.session_state.get("pick_local", False)
    active_idx = st.session_state.get("active_adjustment_idx", -1)
    img_raw = st.session_state.preview_raw
    rh_orig, rw_orig = img_raw.shape[:2]
    working_copy_size = st.session_state.get("working_copy_size", 1800)
    rotation = st.session_state.get("rotation", 0) % 4
    fine_rot = st.session_state.get("fine_rotation", 0.0)
    autocrop = st.session_state.get("autocrop", False)
    autocrop_offset = st.session_state.get("autocrop_offset", 0)
    autocrop_ratio = st.session_state.get("autocrop_ratio", "3:2")

    # --- Geometry Mapping ---
    u_raw, v_raw = np.meshgrid(np.linspace(0, 1, rw_orig), np.linspace(0, 1, rh_orig))
    uv_grid = np.stack([u_raw, v_raw], axis=-1).astype(np.float32)

    if rotation != 0:
        uv_grid = ensure_array(np.rot90(uv_grid, k=rotation))

    if fine_rot != 0.0:
        h_r, w_r = uv_grid.shape[:2]
        m_mat = cv2.getRotationMatrix2D((w_r / 2.0, h_r / 2.0), fine_rot, 1.0)
        uv_grid = ensure_array(
            cv2.warpAffine(uv_grid, m_mat, (w_r, h_r), flags=cv2.INTER_LINEAR)
        )

    if autocrop:
        img_geom = ensure_array(np.rot90(img_raw, k=rotation))
        if fine_rot != 0.0:
            h_f, w_f = img_geom.shape[:2]
            m_f = cv2.getRotationMatrix2D((w_f / 2, h_f / 2), fine_rot, 1.0)
            img_geom = ensure_array(cv2.warpAffine(img_geom, m_f, (w_f, h_f)))
        y1, y2, x1, x2 = get_autocrop_coords(
            img_geom, autocrop_offset, 1.0, autocrop_ratio, detect_res=working_copy_size
        )
        uv_grid = uv_grid[y1:y2, x1:x2]

    def map_click_to_raw(nx: float, ny: float) -> Tuple[float, float]:
        h_uv, w_uv = uv_grid.shape[:2]
        px = int(np.clip(nx * (w_uv - 1), 0, w_uv - 1))
        py = int(np.clip(ny * (h_uv - 1), 0, h_uv - 1))
        raw_uv = uv_grid[py, px]
        return float(raw_uv[0]), float(raw_uv[1])

    # --- Mask Overlay ---
    if (
        is_local_mode
        and active_idx >= 0
        and st.session_state.get("show_active_mask", True)
    ):
        adj = st.session_state.local_adjustments[active_idx]
        mask = generate_local_mask(
            rh_orig, rw_orig, adj.points, adj.radius, adj.feather, 1.0
        )
        img_pos_lin = 1.0 - np.clip(img_raw, 0, 1)
        luma_mask = calculate_luma_mask(
            img_pos_lin,
            adj.luma_range,
            adj.luma_softness,
        )
        final_vis_mask = mask * luma_mask

        if rotation != 0:
            final_vis_mask = np.rot90(final_vis_mask, k=rotation)
        if fine_rot != 0.0:
            h_m, w_m = final_vis_mask.shape[:2]
            M_m = cv2.getRotationMatrix2D((w_m / 2, h_m / 2), fine_rot, 1.0)
            final_vis_mask = cv2.warpAffine(
                final_vis_mask, M_m, (w_m, h_m), flags=cv2.INTER_LINEAR
            )
        if autocrop:
            y1, y2, x1, x2 = get_autocrop_coords(
                img_geom,
                autocrop_offset,
                1.0,
                autocrop_ratio,
                detect_res=working_copy_size,
            )
            final_vis_mask = final_vis_mask[y1:y2, x1:x2]

        if final_vis_mask.shape[:2] != (orig_h, orig_w):
            final_vis_mask = cv2.resize(
                final_vis_mask, (orig_w, orig_h), interpolation=cv2.INTER_AREA
            )

        mask_u8 = (final_vis_mask * 180).astype(np.uint8)

        if border_px > 0:
            mask_u8 = cv2.copyMakeBorder(
                mask_u8,
                border_px,
                border_px,
                border_px,
                border_px,
                cv2.BORDER_CONSTANT,
                value=0,
            )

        mask_pil = Image.fromarray(mask_u8, mode="L")
        overlay = Image.new("RGBA", pil_prev.size, (255, 0, 0, 0))
        red_fill = Image.new("RGBA", pil_prev.size, (255, 75, 75, 255))
        overlay = Image.composite(red_fill, overlay, mask_pil)
        if pil_prev.mode != "RGBA":
            pil_prev = pil_prev.convert("RGBA")
        pil_prev = Image.alpha_composite(pil_prev, overlay).convert("RGB")

    current_file = session.current_file
    if current_file:
        # Header: Name + Resolution
        h1, h2 = st.columns([3, 1])
        with h1:
            st.subheader(current_file["name"])
        with h2:
            if "original_res" in st.session_state:
                w, h = st.session_state.original_res
                st.markdown(
                    f"<div style='text-align: right; padding-top: 1rem; color: gray;'>{w} x {h} px</div>",
                    unsafe_allow_html=True,
                )

        is_dust_mode = st.session_state.get("pick_dust", False)

        img_display = pil_prev.copy()

        # Center the image using native Streamlit columns
        # We use a small side-padding to push the content to the center
        _, center_col, _ = st.columns([0.1, 0.8, 0.1])

        with center_col:
            if is_dust_mode:
                value = streamlit_image_coordinates(
                    img_display,
                    key=f"dust_picker_{st.session_state.get('working_copy_size', 1800)}",
                    width=img_display.width,
                )

                st.info("Click to remove dust spot.")

            elif is_local_mode:
                value = streamlit_image_coordinates(
                    img_display,
                    key=f"local_picker_{st.session_state.get('working_copy_size', 1800)}",
                    width=img_display.width,
                )

            else:
                st.image(img_display, width=img_display.width)

                value = None

        # Footer: Sensitometry Data

        try:
            # Lightweight analysis on preview
            norm_log, _ = prepare_exposure_analysis(st.session_state.preview_raw)
            dr, mid = analyze_sensitometry(norm_log)

            # Calculate Zone V Shift (Midtone Anchor)
            density_val = st.session_state.get("density", 1.0)
            shift = 0.1 + (density_val * PIPELINE_CONSTANTS["density_multiplier"])
            pivot = 1.0 - shift
            zone_diff = mid - pivot

            f1, f2 = st.columns(2)
            f1.markdown(f"**DR:** {dr:.2f}")
            f2.markdown(
                f"<div style='text-align: right;'>**Zone V:** {zone_diff:+.2f}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            logger.warning(f"Failed to render sensitometry footer: {e}")

    if value:
        scale_x = pil_prev.width / img_display.width
        scale_y = pil_prev.height / img_display.height
        abs_x = value["x"] * scale_x
        abs_y = value["y"] * scale_y
        content_x = abs_x - border_px
        content_y = abs_y - border_px

        if 0 <= content_x < orig_w and 0 <= content_y < orig_h:
            nx = content_x / orig_w
            ny = content_y / orig_h
            rx, ry = map_click_to_raw(nx, ny)

            if is_dust_mode and value != st.session_state.last_dust_click:
                st.session_state.last_dust_click = value
                if "manual_dust_spots" not in st.session_state:
                    st.session_state.manual_dust_spots = []
                if st.session_state.get("dust_scratch_mode"):
                    if st.session_state.dust_start_point is None:
                        st.session_state.dust_start_point = (rx, ry)
                        st.toast("Start point set. Click end point.")
                        st.rerun()
                    else:
                        sx, sy = st.session_state.dust_start_point
                        current_size = st.session_state.get("manual_dust_size", 10)
                        norm_radius = current_size / float(working_copy_size)
                        step_size = max(0.0005, norm_radius * 0.5)
                        dist = np.hypot(rx - sx, ry - sy)
                        num_steps = int(dist / step_size)
                        for i in range(num_steps + 1):
                            t = i / max(1, num_steps)
                            st.session_state.manual_dust_spots.append(
                                (sx + (rx - sx) * t, sy + (ry - sy) * t, current_size)
                            )
                        st.session_state.dust_start_point = None
                        save_settings()
                        st.toast("Scratch removed.")
                        st.rerun()
                else:
                    current_size = st.session_state.get("manual_dust_size", 10)
                    st.session_state.manual_dust_spots.append((rx, ry, current_size))
                    save_settings()
                    st.rerun()

            elif is_local_mode:
                if active_idx >= 0:
                    adj = st.session_state.local_adjustments[active_idx]
                    points = adj.points
                    if not points or (rx != points[-1][0] or ry != points[-1][1]):
                        points.append((rx, ry))
                        save_settings()
                        st.rerun()
