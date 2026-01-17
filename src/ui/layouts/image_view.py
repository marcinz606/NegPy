import streamlit as st
import numpy as np
from PIL import Image
from typing import Any
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
from src.services.export.print import PrintService
from src.ui.state.view_models import (
    GeometryViewModel,
    RetouchViewModel,
    ExposureViewModel,
)

logger = get_logger(__name__)


def render_image_view(
    pil_prev: Image.Image, border_config: SidebarState | None = None
) -> None:
    ctx = SessionContext()
    session = ctx.session
    vm_retouch = RetouchViewModel()

    orig_w, orig_h = pil_prev.size
    content_rect = (0, 0, orig_w, orig_h)

    if border_config and border_config.add_border:
        try:
            pil_prev, content_rect = PrintService.apply_preview_layout_to_pil(
                pil_prev,
                border_config.paper_aspect_ratio,
                border_config.border_size,
                border_config.print_width,
                border_config.border_color,
                float(ctx.working_copy_size),
            )
        except Exception as e:
            logger.error(f"Border preview error: {e}")

    cx, cy, cw, ch = content_rect

    geo_vm = GeometryViewModel()
    geo_conf = geo_vm.to_config()

    img_raw = ctx.preview_raw
    if img_raw is None:
        return

    rh_orig, rw_orig = img_raw.shape[:2]

    metrics = st.session_state.get("last_metrics", {})
    roi = metrics.get("active_roi")

    uv_grid = CoordinateMapping.create_uv_grid(
        rh_orig,
        rw_orig,
        geo_conf.rotation % 4,
        geo_conf.fine_rotation,
        geo_conf.flip_horizontal,
        geo_conf.flip_vertical,
        (geo_conf.autocrop or geo_conf.manual_crop) and not geo_conf.keep_full_frame,
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
            content_rect,
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

        def get_mode_state(vm: Any, field: str) -> bool:
            key = vm.get_key(field)
            return bool(
                st.session_state.get(f"w_{key}", False)
                or st.session_state.get(key, False)
            )

        is_dust_mode = get_mode_state(vm_retouch, "pick_dust")
        is_assist_mode = get_mode_state(geo_vm, "pick_assist")
        is_manual_crop_mode = get_mode_state(geo_vm, "pick_manual_crop")

        vm_exp = ExposureViewModel()
        is_wb_mode = get_mode_state(vm_exp, "pick_wb")

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
                content_rect,
                alpha=100,
            )

        working_size = ctx.working_copy_size

        _, center_col, _ = st.columns([0.1, 0.8, 0.1])
        with center_col:
            if (
                is_dust_mode
                or is_local_mode
                or is_assist_mode
                or is_manual_crop_mode
                or is_wb_mode
            ):
                value = streamlit_image_coordinates(
                    img_display, key=f"picker_{working_size}", width=working_size
                )
                if is_dust_mode:
                    st.info("Click to remove dust spot.")
                elif is_assist_mode:
                    st.info("Click on the film border (unexposed area) to assist crop.")
                elif is_manual_crop_mode:
                    if st.session_state.get("manual_crop_start_point") is None:
                        st.info("Click top-left corner of the crop.")
                    else:
                        st.info("Click bottom-right corner of the crop.")
                elif is_wb_mode:
                    st.info("Click on a neutral grey area to balance colors.")
            else:
                st.image(img_display, width=working_size)
                value = None

    if value:
        scale = pil_prev.width / float(working_size)

        canvas_x = value["x"] * scale
        canvas_y = value["y"] * scale

        if cx <= canvas_x < (cx + cw) and cy <= canvas_y < (cy + ch):
            nx = (canvas_x - cx) / cw
            ny = (canvas_y - cy) / ch

            rx, ry = CoordinateMapping.map_click_to_raw(nx, ny, uv_grid)

            if is_wb_mode and value != st.session_state.get("last_wb_click"):
                st.session_state.last_wb_click = value
                from src.ui.controllers.app_controller import AppController

                ctrl = AppController(ctx)
                ctrl.handle_wb_pick(nx, ny)
                st.rerun()

            elif is_manual_crop_mode and value != st.session_state.get(
                "last_manual_crop_click"
            ):
                st.session_state.last_manual_crop_click = value
                start_point = st.session_state.get("manual_crop_start_point")

                if start_point is None:
                    st.session_state.manual_crop_start_point = (rx, ry)
                    st.toast("Top-left point set. Now click bottom-right.")
                    st.rerun()
                else:
                    x1, y1 = start_point
                    x2, y2 = rx, ry
                    st.session_state[geo_vm.get_key("manual_crop_rect")] = (
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                    st.session_state.manual_crop_start_point = None
                    st.session_state[geo_vm.get_key("pick_manual_crop")] = False
                    save_settings()
                    st.toast("Manual crop set.")
                    st.rerun()

            elif is_dust_mode and value != st.session_state.last_dust_click:
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
