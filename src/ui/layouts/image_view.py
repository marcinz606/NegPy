import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates
from src.kernel.system.logging import get_logger
from src.kernel.system.config import APP_CONFIG
from src.kernel.image.validation import validate_int
from src.domain.types import LUMA_R, LUMA_G, LUMA_B
from src.ui.state.state_manager import save_settings
from src.ui.state.view_models import SidebarState
from src.ui.state.session_context import SessionContext
from src.services.view.coordinate_mapping import CoordinateMapping
from src.services.view.overlays import Overlays
from src.ui.state.view_models import (
    GeometryViewModel,
    RetouchViewModel,
)

logger = get_logger(__name__)


def render_image_view(
    pil_prev: Image.Image, border_config: SidebarState | None = None
) -> None:
    ctx = SessionContext()
    session = ctx.session
    vm_retouch = RetouchViewModel()
    border_px = 0
    orig_w, orig_h = pil_prev.size

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

    geo_vm = GeometryViewModel()
    geo_conf = geo_vm.to_config()

    img_raw = ctx.preview_raw
    if img_raw is None:
        return

    rh_orig, rw_orig = img_raw.shape[:2]

    # Retrieve ROI from the latest engine pass for perfect alignment
    metrics = st.session_state.get("last_metrics", {})
    roi = metrics.get("active_roi")

    uv_grid = CoordinateMapping.create_uv_grid(
        rh_orig,
        rw_orig,
        geo_conf.rotation % 4,
        geo_conf.fine_rotation,
        geo_conf.autocrop and not geo_conf.keep_full_frame,
        {"roi": roi} if roi else None,
    )

    is_local_mode = ctx.pick_local
    active_idx = ctx.active_adjustment_idx

    if (
        is_local_mode
        and active_idx >= 0
        and st.session_state.get("show_active_mask", True)
    ):
        adj = st.session_state.local_adjustments[active_idx]
        pil_prev = Overlays.apply_adjustment_mask(
            pil_prev,
            img_raw,
            adj.points,
            adj.radius,
            adj.feather,
            adj.luma_range,
            adj.luma_softness,
            geo_conf,
            roi,
            border_px,
        )

    current_file = session.current_file
    if current_file:
        h1, h2 = st.columns([3, 1])
        with h1:
            name = current_file["name"]
            st.markdown(
                f"<div style='text-align: left; padding-top: 1rem; color: gray;'>{name}</div>",
                unsafe_allow_html=True,
            )
        with h2:
            w, h = ctx.original_res
            st.markdown(
                f"<div style='text-align: right; padding-top: 1rem; color: gray;'>{w} x {h} px</div>",
                unsafe_allow_html=True,
            )

        is_dust_mode = st.session_state.get(vm_retouch.get_key("pick_dust"), False)
        is_assist_mode = st.session_state.get(geo_vm.get_key("pick_assist"), False)
        img_display = pil_prev.copy()

        if st.session_state.get(vm_retouch.get_key("show_dust_patches")):
            manual_spots = st.session_state.get(
                vm_retouch.get_key("manual_dust_spots"), []
            )
            img_display = Overlays.apply_dust_patches(
                img_display,
                manual_spots,
                (rh_orig, rw_orig),
                geo_conf,
                roi,
                border_px,
                alpha=100,
            )

        working_size = ctx.working_copy_size

        _, center_col, _ = st.columns([0.1, 0.8, 0.1])
        with center_col:
            if is_dust_mode or is_local_mode or is_assist_mode:
                value = streamlit_image_coordinates(
                    img_display, key=f"picker_{working_size}", width=working_size
                )
                if is_dust_mode:
                    st.info("Click to remove dust spot.")
                elif is_assist_mode:
                    st.info("Click on the film border (unexposed area) to assist crop.")
            else:
                st.image(img_display, width=working_size)
                value = None

    if value:
        scale = pil_prev.width / float(working_size)
        content_x = (value["x"] * scale) - border_px
        content_y = (value["y"] * scale) - border_px

        if 0 <= content_x < orig_w and 0 <= content_y < orig_h:
            rx, ry = CoordinateMapping.map_click_to_raw(
                content_x / orig_w, content_y / orig_h, uv_grid
            )

            if is_dust_mode and value != st.session_state.last_dust_click:
                st.session_state.last_dust_click = value
                manual_spots_key = vm_retouch.get_key("manual_dust_spots")
                if manual_spots_key not in st.session_state:
                    st.session_state[manual_spots_key] = []

                if st.session_state.get(vm_retouch.get_key("dust_scratch_mode")):
                    if st.session_state.dust_start_point is None:
                        st.session_state.dust_start_point = (rx, ry)
                        st.toast("Start point set. Click end point.")
                        st.rerun()
                    else:
                        sx, sy = st.session_state.dust_start_point
                        size = validate_int(
                            st.session_state.get(
                                vm_retouch.get_key("manual_dust_size"), 10
                            ),
                            10,
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
                            st.session_state[manual_spots_key].append(
                                (sx + (rx - sx) * t, sy + (ry - sy) * t, size)
                            )
                        st.session_state.dust_start_point = None
                        save_settings()
                        st.toast("Scratch removed.")
                        st.rerun()
                else:
                    size = validate_int(
                        st.session_state.get(
                            vm_retouch.get_key("manual_dust_size"), 10
                        ),
                        10,
                    )
                    st.session_state[manual_spots_key].append((rx, ry, size))
                    save_settings()
                    st.rerun()

            elif is_assist_mode and value != st.session_state.get("last_assist_click"):
                st.session_state.last_assist_click = value

                # Sample luma at (rx, ry) on img_raw
                raw_h, raw_w = img_raw.shape[:2]
                px = int(np.clip(rx * raw_w, 0, raw_w - 1))
                py = int(np.clip(ry * raw_h, 0, raw_h - 1))

                pixel = img_raw[py, px]
                luma = LUMA_R * pixel[0] + LUMA_G * pixel[1] + LUMA_B * pixel[2]

                st.session_state[geo_vm.get_key("autocrop_assist_point")] = (rx, ry)
                st.session_state[geo_vm.get_key("autocrop_assist_luma")] = float(luma)
                st.session_state[geo_vm.get_key("pick_assist")] = False  # Auto-off

                save_settings()
                st.toast(f"Assisted luma set to {luma:.3f}")
                st.rerun()

            elif is_local_mode and active_idx >= 0:
                points = st.session_state.local_adjustments[active_idx].points
                if not points or (rx != points[-1][0] or ry != points[-1][1]):
                    points.append((rx, ry))
                    save_settings()
                    st.rerun()
