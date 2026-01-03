import streamlit as st
import rawpy
import numpy as np
import os
import asyncio
import concurrent.futures
import cv2
from PIL import Image, ImageCms, ImageDraw
from typing import Any, Tuple
from src.config import DEFAULT_SETTINGS, APP_CONFIG
from src.domain_objects import ImageSettings, ExportSettings
from src.backend.utils import get_thumbnail_worker
from src.helpers import ensure_rgb, imread_raw, ensure_array
from src.backend.db import init_db
from src.backend.image_logic.retouch import get_autocrop_coords
from src.backend.image_logic.post import (
    apply_post_color_grading,
    apply_output_sharpening,
)
from src.backend.processor import (
    load_raw_and_process,
)
from src.frontend.state import init_session_state
from src.frontend.css import apply_custom_css
from src.frontend.components.sidebar.main import (
    render_file_manager,
    render_sidebar_content,
)
from src.frontend.components.main_layout import render_main_layout


def get_processing_params(
    source: Any, overrides: ImageSettings | None = None
) -> ImageSettings:
    """
    Consolidates parameter gathering from either session_state or a file settings dict.
    """
    if overrides:
        pass

    # Extract all relevant fields from source (could be session_state proxy or dict)
    data = {}
    for field_name in ImageSettings.__dataclass_fields__.keys():
        try:
            if field_name in source:
                data[field_name] = source[field_name]
        except Exception:
            # Handle cases where source is not indexable
            if hasattr(source, field_name):
                data[field_name] = getattr(source, field_name)

    return ImageSettings.from_dict(data)


async def main() -> None:
    """
    Initialize frontend app function.
    """
    st.set_page_config(
        page_title="DarkroomPy", layout="wide", page_icon=":material/camera_roll:"
    )
    init_db()
    init_session_state()
    session = st.session_state.session

    if "assets_initialized" not in st.session_state:
        # First time app is loaded in this session:
        session.asset_manager.initialize()
        session.asset_manager.clear_all()
        st.session_state.assets_initialized = True

    apply_custom_css()

    # 1. Global Status Area (Top of page)
    status_area = st.empty()

    # 2. Render File Manager (Uploads)
    render_file_manager()

    if session.uploaded_files:
        current_file = session.current_file
        if not current_file:
            return

        # 3. Load Data
        current_color_space = st.session_state.get("export_color_space", "sRGB")
        if (
            st.session_state.get("last_file") != current_file["name"]
            or st.session_state.get("last_preview_color_space") != current_color_space
        ):
            with status_area.status(
                f"Loading {current_file['name']}...", expanded=False
            ) as status:
                raw_color_space = rawpy.ColorSpace.sRGB
                if current_color_space == "Adobe RGB":
                    raw_color_space = rawpy.ColorSpace.Adobe

                # Read -> numpy
                with imread_raw(current_file["path"]) as raw:
                    rgb = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=False,
                        user_wb=[1, 1, 1, 1],
                        output_bps=16,
                        output_color=raw_color_space,
                    )
                    rgb = ensure_rgb(rgb)

                    full_linear = rgb.astype(np.float32) / 65535.0
                    h_orig, w_orig = full_linear.shape[:2]
                    max_res = APP_CONFIG.preview_max_res
                    if max(h_orig, w_orig) > max_res:
                        scale = max_res / max(h_orig, w_orig)
                        st.session_state.preview_raw = ensure_array(
                            cv2.resize(
                                full_linear,
                                (int(w_orig * scale), int(h_orig * scale)),
                                interpolation=cv2.INTER_AREA,
                            )
                        )
                    else:
                        st.session_state.preview_raw = full_linear.copy()
                    st.session_state.last_file = current_file["name"]
                    st.session_state.last_preview_color_space = current_color_space
                status.update(label=f"Loaded {current_file['name']}", state="complete")

        # 4. Render Sidebar
        sidebar_data = render_sidebar_content()

        # 5. Background Tasks
        # Thumbnails
        missing_thumbs = [
            f for f in session.uploaded_files if f["name"] not in session.thumbnails
        ]
        if missing_thumbs:
            with status_area.status(
                "Generating thumbnails...", expanded=False
            ) as status:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=APP_CONFIG.max_workers
                ) as executor:
                    loop = asyncio.get_running_loop()
                    tasks = [
                        loop.run_in_executor(executor, get_thumbnail_worker, f["path"])
                        for f in missing_thumbs
                    ]
                    results = await asyncio.gather(*tasks)
                    for f_meta, thumb in zip(missing_thumbs, results):
                        if isinstance(thumb, Image.Image):
                            session.thumbnails[f_meta["name"]] = thumb
                status.update(label="Thumbnails ready", state="complete")

        # Build current params
        current_params = get_processing_params(st.session_state)

        # Ensure latest settings are saved before processing/exporting
        session.save_active_settings()

        # Core Processing
        processed_preview = session.engine.process(
            st.session_state.preview_raw.copy(), current_params
        )
        pil_prev = Image.fromarray(
            np.clip(np.nan_to_num(processed_preview * 255), 0, 255).astype(np.uint8)
        )

        # Post-Processing
        pil_prev = apply_post_color_grading(pil_prev, current_params)
        pil_prev = apply_output_sharpening(pil_prev, st.session_state.sharpen)

        is_toned = (
            st.session_state.temperature != 0.0
            or st.session_state.shadow_temp != 0.0
            or st.session_state.highlight_temp != 0.0
        )

        if current_params.is_bw and not is_toned:
            pil_prev = pil_prev.convert("L")

        # ICC Profile Preview
        if session.icc_profile_path:
            try:
                profile_src: Any
                if current_color_space == "Adobe RGB" and os.path.exists(
                    APP_CONFIG.adobe_rgb_profile
                ):
                    profile_src = ImageCms.getOpenProfile(APP_CONFIG.adobe_rgb_profile)
                else:
                    profile_src = ImageCms.createProfile("sRGB")

                dst_profile = ImageCms.getOpenProfile(session.icc_profile_path)
                if pil_prev.mode != "RGB":
                    pil_prev = pil_prev.convert("RGB")

                result_icc = ImageCms.profileToProfile(
                    pil_prev,
                    profile_src,
                    dst_profile,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB",
                    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
                )
                if result_icc is not None:
                    pil_prev = result_icc
            except Exception as e:
                st.sidebar.error(f"ICC Error: {e}")

        # Display Simulation
        if current_color_space == "Adobe RGB" and not session.icc_profile_path:
            try:
                if os.path.exists(APP_CONFIG.adobe_rgb_profile):
                    adobe_prof = ImageCms.getOpenProfile(APP_CONFIG.adobe_rgb_profile)
                    srgb_prof: Any = ImageCms.createProfile("sRGB")
                    if pil_prev.mode != "RGB":
                        pil_prev = pil_prev.convert("RGB")
                    result_sim = ImageCms.profileToProfile(
                        pil_prev,
                        adobe_prof,
                        srgb_prof,
                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                        outputMode="RGB",
                    )
                    if result_sim is not None:
                        pil_prev = result_sim
            except Exception:
                pass

        # Visualization
        if st.session_state.get("pick_dust", False) and st.session_state.get(
            "show_dust_patches", True
        ):
            if st.session_state.get("manual_dust_spots") or st.session_state.get(
                "dust_start_point"
            ):
                if pil_prev.mode != "RGBA":
                    pil_prev = pil_prev.convert("RGBA")

                overlay = Image.new("RGBA", pil_prev.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                pw, ph = pil_prev.size
                m_rad = st.session_state.manual_dust_size

                rh_orig, rw_orig = st.session_state.preview_raw.shape[:2]
                rotation = st.session_state.rotation % 4
                fine_rot = st.session_state.fine_rotation
                autocrop = st.session_state.autocrop
                autocrop_offset = st.session_state.autocrop_offset

                def transform_to_display(rx: float, ry: float) -> Tuple[float, float]:
                    if rotation == 0:
                        tx, ty = rx, ry
                    elif rotation == 1:
                        tx, ty = ry, 1.0 - rx
                    elif rotation == 2:
                        tx, ty = 1.0 - rx, 1.0 - ry
                    elif rotation == 3:
                        tx, ty = 1.0 - ry, rx
                    else:
                        tx, ty = rx, ry

                    rh_r, rw_r = rh_orig, rw_orig
                    if rotation % 2 != 0:
                        rh_r, rw_r = rw_orig, rh_orig

                    if fine_rot != 0.0:
                        cx, cy = rw_r / 2.0, rh_r / 2.0
                        px, py = tx * rw_r, ty * rh_r
                        angle = np.radians(fine_rot)
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        dx, dy = px - cx, py - cy
                        px_f = cx + dx * cos_a + dy * sin_a
                        py_f = cy - dx * sin_a + dy * cos_a
                        tx, ty = px_f / rw_r, py_f / rh_r

                    if autocrop:
                        img_geom = np.rot90(st.session_state.preview_raw, k=rotation)
                        if fine_rot != 0.0:
                            M_f = cv2.getRotationMatrix2D(
                                (rw_r / 2, rh_r / 2), fine_rot, 1.0
                            )
                            img_geom = cv2.warpAffine(img_geom, M_f, (rw_r, rh_r))

                        y1, y2, x1, x2 = get_autocrop_coords(
                            img_geom,
                            autocrop_offset,
                            1.0,
                            st.session_state.autocrop_ratio,
                        )
                        cw, ch = x2 - x1, y2 - y1
                        if cw > 0 and ch > 0:
                            tx = (tx * rw_r - x1) / cw
                            ty = (ty * rh_r - y1) / ch

                    return tx, ty

                for spot in st.session_state.get("manual_dust_spots", []):
                    rx, ry, m_rad = spot

                    tx, ty = transform_to_display(rx, ry)
                    if 0 <= tx <= 1 and 0 <= ty <= 1:
                        px, py = tx * pw, ty * ph
                        # 33% opacity filled red circle
                        draw.ellipse(
                            (px - m_rad, py - m_rad, px + m_rad, py + m_rad),
                            fill=(255, 75, 75, 84),
                            outline=(255, 75, 75, 84),
                            width=1,
                        )

                if (
                    st.session_state.get("dust_scratch_mode")
                    and st.session_state.dust_start_point
                ):
                    sx, sy = transform_to_display(*st.session_state.dust_start_point)
                    if 0 <= sx <= 1 and 0 <= sy <= 1:
                        spx, spy = sx * pw, sy * ph
                        # 33% opacity filled yellow circle for start point
                        draw.ellipse(
                            (spx - m_rad, spy - m_rad, spx + m_rad, spy + m_rad),
                            fill=(241, 196, 15, 84),
                            outline=(241, 196, 15, 84),
                            width=1,
                        )

                pil_prev = Image.alpha_composite(pil_prev, overlay).convert("RGB")

        # 6. Main Content Layout
        export_btn_sidebar = render_main_layout(pil_prev, sidebar_data)

        # Handle Export Logic
        if export_btn_sidebar:
            with status_area.status(
                "Exporting current file...", expanded=True
            ) as status:
                f_hash = current_file["hash"]
                f_settings = session.file_settings.get(f_hash, DEFAULT_SETTINGS)
                f_params = get_processing_params(f_settings)

                export_settings = ExportSettings(
                    output_format=sidebar_data.out_fmt,
                    print_width_cm=sidebar_data.print_width,
                    dpi=sidebar_data.print_dpi,
                    sharpen_amount=f_params.sharpen,
                    filename=current_file["name"],
                    add_border=sidebar_data.add_border,
                    border_size_cm=sidebar_data.border_size,
                    border_color=sidebar_data.border_color,
                    icc_profile_path=session.icc_profile_path
                    if sidebar_data.apply_icc
                    else None,
                    color_space=sidebar_data.color_space,
                )

                img_bytes, ext = load_raw_and_process(
                    current_file["path"], f_params, export_settings
                )
                if img_bytes:
                    os.makedirs(sidebar_data.export_path, exist_ok=True)
                    out_path = os.path.join(
                        sidebar_data.export_path,
                        f"processed_{current_file['name'].rsplit('.', 1)[0]}.{ext}",
                    )
                    with open(out_path, "wb") as out_f:
                        out_f.write(img_bytes)
                    status.update(
                        label=f"Exported to {os.path.basename(out_path)}",
                        state="complete",
                    )
                    st.toast(f"Exported to {os.path.basename(out_path)}")

        if sidebar_data.process_btn:
            os.makedirs(sidebar_data.export_path, exist_ok=True)
            with status_area.status("Batch processing...", expanded=True) as status:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=APP_CONFIG.max_workers
                ) as executor:
                    loop = asyncio.get_running_loop()
                    batch_tasks = []
                    file_names = []
                    for f_meta in session.uploaded_files:
                        f_hash = f_meta["hash"]
                        f_settings = session.file_settings.get(f_hash, DEFAULT_SETTINGS)
                        f_params = get_processing_params(f_settings)
                        file_names.append(f_meta["name"])

                        f_export_settings = ExportSettings(
                            output_format=sidebar_data.out_fmt,
                            print_width_cm=sidebar_data.print_width,
                            dpi=sidebar_data.print_dpi,
                            sharpen_amount=f_params.sharpen,
                            filename=f_meta["name"],
                            add_border=sidebar_data.add_border,
                            border_size_cm=sidebar_data.border_size,
                            border_color=sidebar_data.border_color,
                            icc_profile_path=session.icc_profile_path
                            if sidebar_data.apply_icc
                            else None,
                            color_space=sidebar_data.color_space,
                        )

                        batch_tasks.append(
                            loop.run_in_executor(
                                executor,
                                load_raw_and_process,
                                f_meta["path"],
                                f_params,
                                f_export_settings,
                            )
                        )
                    batch_results = await asyncio.gather(
                        *batch_tasks, return_exceptions=True
                    )

                    for fname, res_any in zip(file_names, batch_results):
                        if isinstance(res_any, tuple):
                            res_tuple: Tuple[Any, Any] = res_any
                            img_bytes = res_tuple[0]
                            ext = res_tuple[1]
                            if img_bytes is not None:
                                out_path = os.path.join(
                                    sidebar_data.export_path,
                                    f"processed_{fname.rsplit('.', 1)[0]}.{ext}",
                                )
                                with open(out_path, "wb") as out_f:
                                    out_f.write(img_bytes)
                        elif isinstance(res_any, Exception):
                            st.error(f"Error processing {fname}: {res_any}")

                status.update(label="Batch Processing Complete", state="complete")
                st.success("Batch Processing Complete")
    else:
        st.info("Upload files to start.")


if __name__ == "__main__":
    asyncio.run(main())
