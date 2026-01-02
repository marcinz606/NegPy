import streamlit as st
import rawpy
import numpy as np
import os
import asyncio
import uuid
import concurrent.futures
import cv2
from PIL import Image, ImageCms, ImageDraw
from typing import Any, cast, Tuple
from src.config import DEFAULT_SETTINGS, APP_CONFIG
from src.domain_objects import ProcessingParams, ExportSettings
from src.backend.utils import get_thumbnail_worker
from src.helpers import ensure_rgb, imread_raw, ensure_array
from src.backend.assets import AssetManager
from src.backend.db import init_db
from src.backend.image_logic.retouch import get_autocrop_coords
from src.backend.image_logic.post import (
    apply_post_color_grading,
    apply_output_sharpening,
)
from src.backend.processor import (
    process_image_core,
    load_raw_and_process,
)
from src.frontend.state import init_session_state, save_settings
from src.frontend.css import apply_custom_css
from src.frontend.components.sidebar.main import (
    render_file_manager,
    render_sidebar_content,
)
from src.frontend.components.main_layout import render_main_layout


def get_processing_params(
    source: Any, overrides: ProcessingParams | None = None
) -> ProcessingParams:
    """
    Consolidates parameter gathering from either session_state or a file settings dict.
    Converts CMY filtration (0-170) to linear gains for the processor.
    Calculates internal Gain/Gamma from the unified Grade slider.
    """
    if overrides:
        # If we have overrides, we would merge them here
        pass

    # 1. Convert CMY Filters to Linear Gains (Corrected Darkroom Model)
    # Adding filtration (slider UP) must REMOVE that color from the print.
    c_val = np.clip(source.get("wb_cyan", 0), 0, 170)
    m_val = np.clip(source.get("wb_magenta", 0), 0, 170)
    y_val = np.clip(source.get("wb_yellow", 0), 0, 170)

    # We use positive exponents to flip the response direction as requested.
    r_gain = 10.0 ** (c_val / 100.0)
    g_gain = 10.0 ** (m_val / 100.0)
    b_gain = 10.0 ** (y_val / 100.0)

    # 2. Density & Grade Mapping (Photometric Model)
    # 'Density' acts as the primary Exposure control (Shift).
    # 'Grade' acts as the paper contrast/slope control.

    # Density 0.0 (Light) -> 3.0 (Dark).
    density = source.get("density", 1.0)
    
    # Grade 0.0 -> 5.0 (Default 2.0).
    grade = source.get("grade", 2.0)

    p: ProcessingParams = {
        "density": float(density),
        "wb_cyan": float(c_val),
        "wb_magenta": float(m_val),
        "wb_yellow": float(y_val),
        "scan_gain": 0.0, # Placeholder, not used in new model
        "gamma": 1.0,    # Placeholder
        "scan_gain_s_toe": source.get("scan_gain_s_toe", 0.0),
        "scan_gain_h_shoulder": source.get("scan_gain_h_shoulder", 0.0),
        "toe": float(source.get("toe", 0.0)),
        "shoulder": float(source.get("shoulder", 0.0)),
        "wb_manual_r": float(r_gain),
        "wb_manual_g": float(g_gain),
        "wb_manual_b": float(b_gain),
        "temperature": source.get("temperature", 0.0),
        "shadow_temp": source.get("shadow_temp", 0.0),
        "highlight_temp": source.get("highlight_temp", 0.0),
        "shadow_desat_strength": source.get("shadow_desat_strength", 1.0),
        "grade": float(grade),
        "color_separation": source.get("color_separation", 1.0),
        "saturation": source.get("saturation", 1.0),
        "exposure": source.get("exposure", 0.0),
        "autocrop": source.get("autocrop", True),
        "autocrop_offset": source.get("autocrop_offset", 5),
        "autocrop_ratio": source.get("autocrop_ratio", "3:2"),
        "dust_remove": source.get("dust_remove", True),
        "dust_threshold": source.get("dust_threshold", 0.6),
        "dust_size": source.get("dust_size", 2),
        "manual_dust_spots": source.get("manual_dust_spots", []),
        "manual_dust_size": source.get("manual_dust_size", 10),
        "c_noise_remove": source.get("c_noise_remove", True),
        "c_noise_strength": source.get("c_noise_strength", 33),
        "local_adjustments": source.get("local_adjustments", []),
        "rotation": source.get("rotation", 0),
        "fine_rotation": source.get("fine_rotation", 0.0),
        "process_mode": source.get("process_mode", "C41"),
        "is_bw": source.get("process_mode") == "B&W",
        "auto_wb": source.get("auto_wb", False),
        "sharpen": source.get("sharpen", 0.75),
    }

    return p


async def main() -> None:
    """
    Initialize frontend app function.
    """
    st.set_page_config(
        page_title="DarkroomPy", layout="wide", page_icon=":material/camera_roll:"
    )
    init_db()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if "assets_initialized" not in st.session_state:
        # First time app is loaded in this session:
        # 1. Ensure base directory exists
        AssetManager.initialize()
        # 2. Wipe old stale sessions from disk
        AssetManager.clear_all()
        st.session_state.assets_initialized = True

    apply_custom_css()
    init_session_state()

    # 1. Render File Manager (Uploads)
    uploaded_files = render_file_manager(st.session_state.session_id)

    if uploaded_files:
        current_file = uploaded_files[st.session_state.selected_file_idx]

        # 3. Load RAW Data (Needed for Auto-Adjustments)
        # We reload if the file changes OR if the output color space changes
        current_color_space = st.session_state.get("export_color_space", "sRGB")
        if (
            st.session_state.get("last_file") != current_file["name"]
            or st.session_state.get("last_preview_color_space") != current_color_space
        ):
            with st.spinner(
                f"Loading preview for {current_file['name']} in {current_color_space}..."
            ):
                # Determine Rawpy Output Color Space
                raw_color_space = rawpy.ColorSpace.sRGB
                if current_color_space == "Adobe RGB":
                    raw_color_space = rawpy.ColorSpace.Adobe

                # Read RAW -> numpy
                with imread_raw(current_file["path"]) as raw:
                    rgb = raw.postprocess(
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=False,
                        user_wb=[1, 1, 1, 1],
                        output_bps=16,
                        output_color=raw_color_space,
                    )
                    # Handle greyscale
                    rgb = ensure_rgb(rgb)

                    full_linear = rgb.astype(np.float32) / 65535.0
                    h_orig, w_orig = full_linear.shape[:2]
                    max_res = APP_CONFIG["preview_max_res"]
                    if max(h_orig, w_orig) > max_res:
                        scale = max_res / max(h_orig, w_orig)
                        # Downscale (for preview)
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

        # 4. Render Sidebar Content (Sliders, Nav, etc.)
        sidebar_data = render_sidebar_content(uploaded_files)

        # 5. Background Tasks & Core Processing
        # Thumbnails
        missing_thumbs = [
            f for f in uploaded_files if f["name"] not in st.session_state.thumbnails
        ]
        if missing_thumbs:
            with st.spinner("Generating thumbnails..."):
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=APP_CONFIG["max_workers"]
                ) as executor:
                    loop = asyncio.get_running_loop()
                    tasks = [
                        loop.run_in_executor(executor, get_thumbnail_worker, f["path"])
                        for f in missing_thumbs
                    ]
                    results = await asyncio.gather(*tasks)
                    for f_meta, thumb in zip(missing_thumbs, results):
                        if isinstance(thumb, Image.Image):
                            st.session_state.thumbnails[f_meta["name"]] = thumb

        # Build current params
        current_params = get_processing_params(st.session_state)

        # Ensure latest settings are saved before processing/exporting
        save_settings(current_file["hash"])

        # Core Processing
        processed_preview = process_image_core(
            st.session_state.preview_raw.copy(), current_params
        )
        pil_prev = Image.fromarray(
            np.clip(np.nan_to_num(processed_preview * 255), 0, 255).astype(np.uint8)
        )

        # Post-Processing (Color Grading & Sharpening)
        pil_prev = apply_post_color_grading(pil_prev, current_params)
        pil_prev = apply_output_sharpening(
            pil_prev, st.session_state.get("sharpen", 0.20)
        )

        # Check if image was toned
        is_toned = (
            st.session_state.get("temperature", 0.0) != 0.0
            or st.session_state.get("shadow_temp", 0.0) != 0.0
            or st.session_state.get("highlight_temp", 0.0) != 0.0
        )

        if current_params["is_bw"] and not is_toned:
            pil_prev = pil_prev.convert("L")

        # ICC Profile Preview (Soft-proofing / Display Simulation)
        if st.session_state.get("icc_profile_path"):
            try:
                # Source is the working color space
                if current_color_space == "Adobe RGB" and os.path.exists(
                    APP_CONFIG.get("adobe_rgb_profile", "")
                ):
                    src_profile = ImageCms.getOpenProfile(
                        APP_CONFIG["adobe_rgb_profile"]
                    )
                else:
                    src_profile = cast(
                        ImageCms.ImageCmsProfile, ImageCms.createProfile("sRGB")
                    )

                # Destination is the selected profile
                dst_profile = ImageCms.getOpenProfile(st.session_state.icc_profile_path)
                # Apply transformation
                if pil_prev.mode != "RGB":
                    pil_prev = pil_prev.convert("RGB")
                # Relative Colorimetric + Blackpoint compensation seems to be standard settings
                # recommended by printing businesses like Saal digital for proofing based on my experience.
                pil_prev = cast(
                    Image.Image,
                    ImageCms.profileToProfile(
                        pil_prev,
                        src_profile,
                        dst_profile,
                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                        outputMode="RGB",
                        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
                    ),
                )
            except Exception as e:
                st.sidebar.error(f"ICC Error: {e}")

        # Display Simulation: Convert back to sRGB for browser if we are in a wider space
        # and not already converted by ICC soft-proofing.
        if current_color_space == "Adobe RGB" and not st.session_state.get(
            "icc_profile_path"
        ):
            try:
                if os.path.exists(APP_CONFIG.get("adobe_rgb_profile", "")):
                    adobe_prof = ImageCms.getOpenProfile(
                        APP_CONFIG["adobe_rgb_profile"]
                    )
                    srgb_prof = cast(
                        ImageCms.ImageCmsProfile, ImageCms.createProfile("sRGB")
                    )
                    if pil_prev.mode != "RGB":
                        pil_prev = pil_prev.convert("RGB")
                    pil_prev = cast(
                        Image.Image,
                        ImageCms.profileToProfile(
                            pil_prev,
                            adobe_prof,
                            srgb_prof,
                            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                            outputMode="RGB",
                        ),
                    )
            except Exception:
                pass

        # Visualization (Red patches for dust spots)
        if st.session_state.get("pick_dust", False) and st.session_state.get(
            "show_dust_patches", True
        ):
            if st.session_state.get("manual_dust_spots"):
                if pil_prev.mode == "L":
                    pil_prev = pil_prev.convert("RGB")
                draw = ImageDraw.Draw(pil_prev)
                pw, ph = pil_prev.size
                m_rad = st.session_state.get("manual_dust_size", 10)

                # We need to transform spots from RAW space to DISPLAY space
                rh_orig, rw_orig = st.session_state.preview_raw.shape[:2]
                rotation = st.session_state.get("rotation", 0) % 4
                fine_rot = st.session_state.get("fine_rotation", 0.0)
                autocrop = st.session_state.get("autocrop", False)
                autocrop_offset = st.session_state.get("autocrop_offset", 0)

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
                        # Match OpenCV getRotationMatrix2D: x' = x*cos + y*sin, y' = -x*sin + y*cos
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
                            st.session_state.get("autocrop_ratio", "3:2"),
                        )
                        cw, ch = x2 - x1, y2 - y1
                        if cw > 0 and ch > 0:
                            tx = (tx * rw_r - x1) / cw
                            ty = (ty * rh_r - y1) / ch

                    return tx, ty

                for rx, ry in st.session_state.manual_dust_spots:
                    tx, ty = transform_to_display(rx, ry)
                    if 0 <= tx <= 1 and 0 <= ty <= 1:
                        px, py = tx * pw, ty * ph
                        draw.ellipse(
                            (px - m_rad, py - m_rad, px + m_rad, py + m_rad),
                            outline="#ff4b4b",
                            width=4,
                        )

                if st.session_state.get("dust_scratch_mode") and st.session_state.get(
                    "dust_start_point"
                ):
                    sx, sy = transform_to_display(*st.session_state.dust_start_point)
                    if 0 <= sx <= 1 and 0 <= sy <= 1:
                        spx, spy = sx * pw, sy * ph
                        draw.ellipse(
                            (spx - m_rad, spy - m_rad, spx + m_rad, spy + m_rad),
                            outline="#f1c40f",
                            width=4,
                        )

        # 6. Main Content Layout
        export_btn_sidebar = render_main_layout(uploaded_files, pil_prev, sidebar_data)

        # Handle Export Logic
        if export_btn_sidebar:
            with st.spinner("Exporting current..."):
                f_current = current_file
                f_settings = st.session_state.file_settings.get(
                    f_current["hash"], DEFAULT_SETTINGS.copy()
                )

                f_params = get_processing_params(f_settings)

                export_settings: ExportSettings = {
                    "output_format": sidebar_data.get("out_fmt", "JPEG"),
                    "print_width_cm": sidebar_data.get("print_width", 27.0),
                    "dpi": sidebar_data.get("print_dpi", 300),
                    "sharpen_amount": f_params.get("sharpen", 0.75),
                    "filename": f_current["name"],
                    "add_border": sidebar_data.get("add_border", False),
                    "border_size_cm": sidebar_data.get("border_size", 1.0),
                    "border_color": sidebar_data.get("border_color", "#000000"),
                    "icc_profile_path": st.session_state.get("icc_profile_path")
                    if sidebar_data.get("apply_icc")
                    else None,
                    "color_space": sidebar_data.get("color_space", "sRGB"),
                }

                img_bytes, ext = load_raw_and_process(
                    f_current["path"], f_params, export_settings
                )
                if img_bytes:
                    os.makedirs(sidebar_data["export_path"], exist_ok=True)
                    out_path = os.path.join(
                        sidebar_data["export_path"],
                        f"processed_{f_current['name'].rsplit('.', 1)[0]}.{ext}",
                    )
                    with open(out_path, "wb") as out_f:
                        out_f.write(img_bytes)
                    st.toast(f"Exported to {os.path.basename(out_path)}")

        if sidebar_data.get("process_btn"):
            os.makedirs(sidebar_data["export_path"], exist_ok=True)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=APP_CONFIG["max_workers"]
            ) as executor:
                loop = asyncio.get_running_loop()
                batch_tasks = []
                file_names = []
                for f_meta in uploaded_files:
                    f_settings = st.session_state.file_settings.get(
                        f_meta["hash"], DEFAULT_SETTINGS.copy()
                    )
                    f_params = get_processing_params(f_settings)
                    file_names.append(f_meta["name"])

                    f_export_settings: ExportSettings = {
                        "output_format": sidebar_data.get("out_fmt", "JPEG"),
                        "print_width_cm": sidebar_data.get("print_width", 27.0),
                        "dpi": sidebar_data.get("print_dpi", 300),
                        "sharpen_amount": f_params.get("sharpen", 0.75),
                        "filename": f_meta["name"],
                        "add_border": sidebar_data.get("add_border", False),
                        "border_size_cm": sidebar_data.get("border_size", 1.0),
                        "border_color": sidebar_data.get("border_color", "#ffffff"),
                        "icc_profile_path": st.session_state.get("icc_profile_path")
                        if sidebar_data.get("apply_icc")
                        else None,
                        "color_space": sidebar_data.get("color_space", "sRGB"),
                    }

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
                        res_tuple = cast(Tuple[Any, Any], res_any)
                        img_bytes = res_tuple[0]
                        ext = res_tuple[1]
                        if img_bytes is not None:
                            out_path = os.path.join(
                                sidebar_data["export_path"],
                                f"processed_{fname.rsplit('.', 1)[0]}.{ext}",
                            )
                            with open(out_path, "wb") as out_f:
                                out_f.write(img_bytes)
                    elif isinstance(res_any, Exception):
                        st.error(f"Error processing {fname}: {res_any}")

                st.success("Batch Processing Complete")
    else:
        st.info("Upload files to start.")


if __name__ == "__main__":
    asyncio.run(main())
