import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates
from src.logging_config import get_logger
from src.config import APP_CONFIG
from src.features.geometry.logic import get_autocrop_coords
from src.features.retouch.logic import generate_local_mask, calculate_luma_mask
from src.features.exposure.analysis import (
    prepare_exposure_analysis,
    analyze_sensitometry,
)
from src.core.constants import PIPELINE_CONSTANTS
from src.core.validation import validate_int
from src.presentation.state.state_manager import save_settings
from src.domain_objects import SidebarData
from src.presentation.state.session_context import SessionContext
from src.presentation.services.geometry_service import GeometryService
from src.presentation.state.view_models import GeometryViewModel, ExposureViewModel

logger = get_logger(__name__)


def render_image_view(
    pil_prev: Image.Image, border_config: SidebarData | None = None
) -> None:
    """
    Renders the main image viewport and handles coordinate-based interaction.
    """
    ctx = SessionContext()
    session = ctx.session
    border_px = 0
    orig_w, orig_h = pil_prev.size

    # --- Border Preview ---
    if border_config and border_config.add_border:
        try:
            print_w = border_config.print_width
            border_w = border_config.border_size
            color = border_config.border_color

            longer_side_px = float(max(pil_prev.size))
            ratio = border_w / print_w if print_w > 0 else 0.01
            border_px = int(longer_side_px * ratio)

            if border_px > 0:
                pil_prev = ImageOps.expand(pil_prev, border=border_px, fill=color)
        except Exception as e:
            logger.error(f"Border preview error: {e}")

    # 1. State & Geometry
    geo_vm = GeometryViewModel()
    geo_conf = geo_vm.to_config()

    img_raw = ctx.preview_raw
    if img_raw is None:
        return

    rh_orig, rw_orig = img_raw.shape[:2]

    # Calculate ROI for UV grid
    roi = None
    if geo_conf.autocrop:
        img_geom = np.rot90(img_raw, k=geo_conf.rotation % 4)
        if geo_conf.fine_rotation != 0.0:
            h_f, w_f = img_geom.shape[:2]
            m_f = cv2.getRotationMatrix2D(
                (w_f / 2, h_f / 2), geo_conf.fine_rotation, 1.0
            )
            img_geom = cv2.warpAffine(img_geom, m_f, (w_f, h_f)).astype(np.float32)

        roi = get_autocrop_coords(
            img_geom,
            geo_conf.autocrop_offset,
            1.0,
            geo_conf.autocrop_ratio,
            detect_res=APP_CONFIG.preview_render_size,
        )

    uv_grid = GeometryService.create_uv_grid(
        rh_orig,
        rw_orig,
        geo_conf.rotation % 4,
        geo_conf.fine_rotation,
        geo_conf.autocrop,
        {"roi": roi} if roi else None,
    )

    # --- Mask Overlay ---
    is_local_mode = ctx.pick_local
    active_idx = ctx.active_adjustment_idx

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
        luma_mask = calculate_luma_mask(img_pos_lin, adj.luma_range, adj.luma_softness)
        final_vis_mask = mask * luma_mask

        if geo_conf.rotation % 4 != 0:
            final_vis_mask = np.rot90(final_vis_mask, k=geo_conf.rotation % 4)
        if geo_conf.fine_rotation != 0.0:
            h_m, w_m = final_vis_mask.shape[:2]
            M_m = cv2.getRotationMatrix2D(
                (w_m / 2, h_m / 2), geo_conf.fine_rotation, 1.0
            )
            final_vis_mask = cv2.warpAffine(
                final_vis_mask, M_m, (w_m, h_m), flags=cv2.INTER_LINEAR
            )
        if roi:
            y1, y2, x1, x2 = roi
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
        h1, h2 = st.columns([3, 1])
        with h1:
            st.subheader(current_file["name"])
        with h2:
            w, h = ctx.original_res
            st.markdown(
                f"<div style='text-align: right; padding-top: 1rem; color: gray;'>{w} x {h} px</div>",
                unsafe_allow_html=True,
            )

        is_dust_mode = st.session_state.get("pick_dust", False)
        img_display = pil_prev.copy()
        working_size = ctx.working_copy_size

        _, center_col, _ = st.columns([0.1, 0.8, 0.1])
        with center_col:
            if is_dust_mode or is_local_mode:
                value = streamlit_image_coordinates(
                    img_display, key=f"picker_{working_size}", width=working_size
                )
                if is_dust_mode:
                    st.info("Click to remove dust spot.")
            else:
                st.image(img_display, width=working_size)
                value = None

        # Footer Analysis
        try:
            norm_log, _ = prepare_exposure_analysis(img_raw)
            dr, mid = analyze_sensitometry(norm_log)
            exp_vm = ExposureViewModel()
            shift = 0.1 + (exp_vm.density * PIPELINE_CONSTANTS["density_multiplier"])
            pivot = 1.0 - shift
            zone_diff = mid - pivot

            f1, f2 = st.columns(2)
            f1.markdown(f"**DR:** {dr:.2f}")
            f2.markdown(
                f"<div style='text-align: right;'>**Zone V:** {zone_diff:+.2f}</div>",
                unsafe_allow_html=True,
            )
        except Exception as e:
            logger.warning(f"Footer analysis error: {e}")

    if value:
        scale = pil_prev.width / float(working_size)
        content_x = (value["x"] * scale) - border_px
        content_y = (value["y"] * scale) - border_px

        if 0 <= content_x < orig_w and 0 <= content_y < orig_h:
            rx, ry = GeometryService.map_click_to_raw(
                content_x / orig_w, content_y / orig_h, uv_grid
            )

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
                        size = validate_int(
                            st.session_state.get("manual_dust_size", 10), 10
                        )
                        dist = np.hypot(rx - sx, ry - sy)
                        num_steps = int(
                            dist
                            / max(
                                0.0005,
                                (size / float(APP_CONFIG.preview_render_size)) * 0.5,
                            )
                        )
                        for i in range(num_steps + 1):
                            t = i / max(1, num_steps)
                            st.session_state.manual_dust_spots.append(
                                (sx + (rx - sx) * t, sy + (ry - sy) * t, size)
                            )
                        st.session_state.dust_start_point = None
                        save_settings()
                        st.toast("Scratch removed.")
                        st.rerun()
                else:
                    size = validate_int(
                        st.session_state.get("manual_dust_size", 10), 10
                    )
                    st.session_state.manual_dust_spots.append((rx, ry, size))
                    save_settings()
                    st.rerun()

            elif is_local_mode and active_idx >= 0:
                points = st.session_state.local_adjustments[active_idx].points
                if not points or (rx != points[-1][0] or ry != points[-1][1]):
                    points.append((rx, ry))
                    save_settings()
                    st.rerun()
