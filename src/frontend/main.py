import streamlit as st
import rawpy
import numpy as np
import io
import os
import asyncio
import concurrent.futures
import cv2
import traceback
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageCms
from typing import Dict, Any, Optional

from src.backend.config import TONE_CURVES_PRESETS, DEFAULT_SETTINGS, APP_CONFIG
from src.backend.utils import create_curve_lut, apply_color_separation, get_thumbnail_worker
from src.backend.db import init_db
from src.backend.image_logic.retouch import get_autocrop_coords
from src.backend.processor import (
    process_image_core, 
    load_raw_and_process, 
    calculate_auto_mask_wb
)
from src.frontend.state import init_session_state, load_settings, save_settings
from src.frontend.components.sidebar.main import render_file_manager, render_sidebar_content
from src.frontend.components.sidebar.adjustments import run_auto_wb, run_auto_density
from src.frontend.components.image_view import render_image_view
from src.frontend.components.contact_sheet import render_contact_sheet

def get_processing_params(source: Dict[str, Any], overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Consolidates parameter gathering from either session_state or a file settings dict.
    Converts CMY filtration (0-170) to linear gains for the processor.
    """
    # 1. Convert CMY Filters to Linear Gains (Corrected Darkroom Model)
    # Adding filtration (slider UP) must REMOVE that color from the print.
    c_val = np.clip(source.get('wb_cyan', 0), 0, 170)
    m_val = np.clip(source.get('wb_magenta', 0), 0, 170)
    y_val = np.clip(source.get('wb_yellow', 0), 0, 170)
    
    # We use positive exponents to flip the response direction as requested.
    r_gain = 10.0 ** (c_val / 100.0)
    g_gain = 10.0 ** (m_val / 100.0)
    b_gain = 10.0 ** (y_val / 100.0)

    p = {
        'scan_gain': source.get('scan_gain', 1.0),
        'scan_gain_s_toe': source.get('scan_gain_s_toe', 0.0),
        'scan_gain_h_shoulder': source.get('scan_gain_h_shoulder', 0.0),
        'wb_manual_r': float(r_gain),
        'wb_manual_g': float(g_gain),
        'wb_manual_b': float(b_gain),
        'temperature': source.get('temperature', 0.0),
        'shadow_temp': source.get('shadow_temp', 0.0),
        'highlight_temp': source.get('highlight_temp', 0.0),
        'gamma': source.get('gamma', 1.0),
        'gamma_mode': source.get('gamma_mode', 'Standard'),
        'shadow_desat_strength': source.get('shadow_desat_strength', 1.0),
        'contrast': source.get('contrast', 1.0),
        'color_separation': source.get('color_separation', 1.0),
        'saturation': source.get('saturation', 1.0),
        'saturation_shadows': source.get('saturation_shadows', 1.0),
        'saturation_highlights': source.get('saturation_highlights', 1.0),
        'exposure': source.get('exposure', 0.0),
        'exposure_shadows': source.get('exposure_shadows', 0.0),
        'exposure_highlights': source.get('exposure_highlights', 0.0),
        'exposure_shadows_range': source.get('exposure_shadows_range', 1.0),
        'exposure_highlights_range': source.get('exposure_highlights_range', 1.0),
        'autocrop': source.get('autocrop', True),
        'autocrop_offset': source.get('autocrop_offset', 5),
        'dust_remove': source.get('dust_remove', True),
        'dust_threshold': source.get('dust_threshold', 0.6),
        'dust_size': source.get('dust_size', 2),
        'manual_dust_spots': source.get('manual_dust_spots', []),
        'manual_dust_size': source.get('manual_dust_size', 10),
        'c_noise_remove': source.get('c_noise_remove', True),
        'c_noise_strength': source.get('c_noise_strength', 33),
        'local_adjustments': source.get('local_adjustments', []),
        'rotation': source.get('rotation', 0),
        'fine_rotation': source.get('fine_rotation', 0.0),
        'monochrome': source.get('monochrome', False),
        'auto_wb': source.get('auto_wb', False),
        'sharpen': source.get('sharpen', 0.75)
    }
    
    # Handle Black/White points
    bw_val = source.get('bw_points', (0.0, 1.0))
    p['black_point'] = bw_val[0]
    p['white_point'] = bw_val[1]
    
    # Add LUT if provided in overrides or source
    if overrides and 'curve_lut_x' in overrides:
        p['curve_lut_x'] = overrides['curve_lut_x']
        p['curve_lut_y'] = overrides['curve_lut_y']
        
    return p

def init_styles():
    st.markdown("""
        <style>
        .stApp { font-size: 18px; }
        h1 { font-size: 36px !important; }
        div.stButton > button {
            font-size: 32px !important;
            height: 1em;
            width: 100%;
        }
        .stDeployButton {
            visibility: hidden;
        }
        </style>
        """, unsafe_allow_html=True)

async def main():
    st.set_page_config(layout="wide", page_title="DarkroomPy")
    init_db()
    init_styles()
    init_session_state()

    # 1. Render File Manager (Uploads) - No Sliders Here
    uploaded_files = render_file_manager()
    
    if uploaded_files:
        current_file = uploaded_files[st.session_state.selected_file_idx]
        
        # 2. Check for File Switch (Only load settings when changing files)
        if st.session_state.get("last_settings_file") != current_file.name:
            is_new_image = load_settings(current_file.name)
            st.session_state.last_settings_file = current_file.name
            
        save_settings(current_file.name)

        # 3. Load RAW Data (Needed for Auto-Adjustments)
        if st.session_state.get("last_file") != current_file.name:
            with st.spinner(f"Loading preview for {current_file.name}..."):
                current_file.seek(0)
                with rawpy.imread(current_file) as raw:
                    rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=False, user_wb=[1, 1, 1, 1], output_bps=16)
                    if rgb.ndim == 2:
                        rgb = np.stack([rgb] * 3, axis=-1)
                    full_linear = rgb.astype(np.float32) / 65535.0
                    h, w = full_linear.shape[:2]
                    max_res = APP_CONFIG['preview_max_res']
                    if max(h, w) > max_res:
                        scale = max_res / max(h, w)
                        st.session_state.preview_raw = cv2.resize(full_linear, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    else:
                        st.session_state.preview_raw = full_linear.copy()
                    st.session_state.last_file = current_file.name

        # 4. Render Sidebar Content (Sliders, Nav, etc.)
        sidebar_data = render_sidebar_content(uploaded_files)

        # Thumbnails
        missing_thumbs = [f for f in uploaded_files if f.name not in st.session_state.thumbnails]
        if missing_thumbs:
            with st.spinner(f"Generating thumbnails..."):
                with concurrent.futures.ProcessPoolExecutor(max_workers=APP_CONFIG['max_workers']) as executor:
                    loop = asyncio.get_event_loop()
                    tasks = [loop.run_in_executor(executor, get_thumbnail_worker, f.getvalue()) for f in missing_thumbs]
                    results = await asyncio.gather(*tasks)
                    for f, thumb in zip(missing_thumbs, results):
                        if thumb: st.session_state.thumbnails[f.name] = thumb

        cx_base, cy_base = TONE_CURVES_PRESETS[st.session_state.get('curve_mode', 'Linear')]
        cy = np.array(cx_base) + (np.array(cy_base) - np.array(cx_base)) * st.session_state.get('curve_strength', 1.0)
        lut_x, lut_y = create_curve_lut(cx_base, cy)

        # Build current params
        current_params = get_processing_params(st.session_state, overrides={
            'curve_lut_x': lut_x, 
            'curve_lut_y': lut_y
        })

        # Core Processing
        processed_preview = process_image_core(st.session_state.preview_raw.copy(), current_params)
        pil_prev = Image.fromarray(np.clip(np.nan_to_num(processed_preview * 255), 0, 255).astype(np.uint8))
        
        # Sharpening
        sharpen_val = st.session_state.get('sharpen', 0.33)
        if sharpen_val > 0:
            img_lab = cv2.cvtColor(np.array(pil_prev), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            l_pil = Image.fromarray(l)
            l_sharpened = l_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharpen_val * 250), threshold=5))
            img_lab_sharpened = cv2.merge([np.array(l_sharpened), a, b])
            pil_prev = Image.fromarray(cv2.cvtColor(img_lab_sharpened, cv2.COLOR_LAB2RGB))

        # Saturation/Mono
        if not current_params.get('monochrome', False):
            # 1. Color Separation
            img_arr = np.array(pil_prev)
            img_sep = apply_color_separation(img_arr, st.session_state.get('color_separation', 1.0))
            pil_prev = Image.fromarray(img_sep)
            
            # 2. Classic Saturation
            sat = st.session_state.get('saturation', 1.0)
            if sat != 1.0:
                enhancer = ImageEnhance.Color(pil_prev)
                pil_prev = enhancer.enhance(sat)
        else:
            pil_prev = pil_prev.convert("L")

        # ICC Profile Preview (Soft-proofing / Display Simulation)
        if st.session_state.get('icc_profile_path'):
            try:
                # Source is sRGB (standard for PIL processed images)
                src_profile = ImageCms.createProfile("sRGB")
                # Destination is the selected profile
                dst_profile = ImageCms.getOpenProfile(st.session_state.icc_profile_path)
                # Apply transformation
                if pil_prev.mode != 'RGB':
                    pil_prev = pil_prev.convert('RGB')
                pil_prev = ImageCms.profileToProfile(
                    pil_prev, 
                    src_profile, 
                    dst_profile, 
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, 
                    outputMode='RGB',
                    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION
                )
            except Exception as e:
                st.sidebar.error(f"ICC Error: {e}")

        # Visualization (Red patches for dust spots)
        if st.session_state.get('pick_dust', False) and st.session_state.get('show_dust_patches', True):
            if st.session_state.get('manual_dust_spots'):
                if pil_prev.mode == 'L': pil_prev = pil_prev.convert("RGB")
                draw = ImageDraw.Draw(pil_prev)
                pw, ph = pil_prev.size
                m_rad = st.session_state.get('manual_dust_size', 10)
                
                # We need to transform spots from RAW space to DISPLAY space
                rh_orig, rw_orig = st.session_state.preview_raw.shape[:2]
                rotation = st.session_state.get('rotation', 0) % 4
                fine_rot = st.session_state.get('fine_rotation', 0.0)
                autocrop = st.session_state.get('autocrop', False)
                autocrop_offset = st.session_state.get('autocrop_offset', 0)
                
                # To transform multiple points efficiently, we can use a helper or manual math
                # Since we already have the logic in map_click_to_raw (inverse), let's implement forward here.
                def transform_to_display(rx, ry):
                    # 1. 90-deg Rotation (Counter-Clockwise to match np.rot90)
                    if rotation == 0: tx, ty = rx, ry
                    elif rotation == 1: tx, ty = ry, 1.0 - rx
                    elif rotation == 2: tx, ty = 1.0 - rx, 1.0 - ry
                    elif rotation == 3: tx, ty = 1.0 - ry, rx
                    else: tx, ty = rx, ry
                    
                    # 2. Fine Rotation
                    # Get rotated dimensions for center
                    rh_r, rw_r = rh_orig, rw_orig
                    if rotation % 2 != 0: rh_r, rw_r = rw_orig, rh_orig
                    
                    if fine_rot != 0.0:
                        cx, cy = rw_r / 2.0, rh_r / 2.0
                        px, py = tx * rw_r, ty * rh_r
                        angle = np.radians(fine_rot)
                        cos_a, sin_a = np.cos(angle), np.sin(angle)
                        dx, dy = px - cx, py - cy
                        px_f = cx + dx * cos_a - dy * sin_a
                        py_f = cy + dx * sin_a + dy * cos_a
                        tx, ty = px_f / rw_r, py_f / rh_r
                    
                    # 3. Autocrop
                    if autocrop:
                        # Find crop box in rotated+fine space
                        img_geom = np.rot90(st.session_state.preview_raw, k=rotation)
                        if fine_rot != 0.0:
                            M_f = cv2.getRotationMatrix2D((rw_r/2, rh_r/2), fine_rot, 1.0)
                            img_geom = cv2.warpAffine(img_geom, M_f, (rw_r, rh_r))
                        
                        y1, y2, x1, x2 = get_autocrop_coords(img_geom, autocrop_offset, 1.0)
                        cw, ch = x2 - x1, y2 - y1
                        if cw > 0 and ch > 0:
                            # Map tx, ty from full image space to [0, 1] of crop box
                            tx = (tx * rw_r - x1) / cw
                            ty = (ty * rh_r - y1) / ch
                    
                    return tx, ty

                for (rx, ry) in st.session_state.manual_dust_spots:
                    tx, ty = transform_to_display(rx, ry)
                    # Draw only if within display bounds
                    if 0 <= tx <= 1 and 0 <= ty <= 1:
                        px, py = tx * pw, ty * ph
                        draw.ellipse((px - m_rad, py - m_rad, px + m_rad, py + m_rad), outline="#ff4b4b", width=4)
                
                if st.session_state.get('dust_scratch_mode') and st.session_state.get('dust_start_point'):
                    sx, sy = transform_to_display(*st.session_state.dust_start_point)
                    if 0 <= sx <= 1 and 0 <= sy <= 1:
                        spx, spy = sx * pw, sy * ph
                        draw.ellipse((spx - m_rad, spy - m_rad, spx + m_rad, spy + m_rad), outline="#f1c40f", width=4)

        # Contact Sheet
        render_contact_sheet(uploaded_files)

        # Main UI Render
        render_image_view(
            pil_prev,
            border_config={
                'add_border': sidebar_data.get('add_border', False),
                'size_cm': sidebar_data.get('border_size', 0.2),
                'color': sidebar_data.get('border_color', '#000000'),
                'print_width_cm': sidebar_data.get('print_width', 27.0)
            }
        )
        if st.sidebar.button("♻️ Reset Manual WB"):
            st.session_state.wb_manual_r, st.session_state.wb_manual_g, st.session_state.wb_manual_b = 1.0, 1.0, 1.0
            save_settings(current_file.name)
            st.rerun()

        # Handle Export Logic
        if sidebar_data['export_btn_sidebar']:
            with st.spinner("Exporting current..."):
                f_current = current_file
                f_settings = st.session_state.file_settings.get(f_current.name, DEFAULT_SETTINGS.copy())
                # ... reuse logic from app.py ...
                cx_b, cy_b = TONE_CURVES_PRESETS[f_settings['curve_mode']]
                cy_f = np.array(cx_b) + (np.array(cy_b) - np.array(cx_b)) * f_settings['curve_strength']
                lx, ly = create_curve_lut(cx_b, cy_f)
                
                f_params = get_processing_params(f_settings, overrides={
                    'curve_lut_x': lx, 'curve_lut_y': ly
                })

                img_bytes, ext = load_raw_and_process(
                    f_current.getvalue(), 
                    f_params, 
                    sidebar_data['out_fmt'], 
                    sidebar_data['print_width'], 
                    sidebar_data['print_dpi'], 
                    f_params['sharpen'],
                    filename=f_current.name,
                    add_border=sidebar_data.get('add_border', False),
                    border_size_cm=sidebar_data.get('border_size', 1.0),
                    border_color=sidebar_data.get('border_color', "#000000"),
                    icc_profile_path=st.session_state.get('icc_profile_path') if sidebar_data.get('apply_icc') else None
                )
                if img_bytes:
                    os.makedirs(sidebar_data['export_path'], exist_ok=True)
                    out_path = os.path.join(sidebar_data['export_path'], f"processed_{f_current.name.rsplit('.', 1)[0]}.{ext}")
                    with open(out_path, "wb") as out_f: out_f.write(img_bytes)
                    st.toast(f"Exported to {os.path.basename(out_path)}")

        if sidebar_data['process_btn']:
            # ... reuse batch logic from app.py ...
            os.makedirs(sidebar_data['export_path'], exist_ok=True)
            with concurrent.futures.ProcessPoolExecutor(max_workers=APP_CONFIG['max_workers']) as executor:
                loop = asyncio.get_event_loop()
                tasks, file_names = [], []
                for f in uploaded_files:
                    f_settings = st.session_state.file_settings.get(f.name, DEFAULT_SETTINGS.copy())
                    cx_b, cy_b = TONE_CURVES_PRESETS[f_settings['curve_mode']]
                    cy_f = np.array(cx_b) + (np.array(cy_b) - np.array(cx_b)) * f_settings['curve_strength']
                    lx, ly = create_curve_lut(cx_b, cy_f)
                    
                    f_params = get_processing_params(f_settings, overrides={
                        'curve_lut_x': lx, 'curve_lut_y': ly
                    })

                    file_names.append(f.name)
                    tasks.append(loop.run_in_executor(
                        executor, 
                        load_raw_and_process, 
                        f.getvalue(), 
                        f_params, 
                        sidebar_data['out_fmt'], 
                        sidebar_data['print_width'], 
                        sidebar_data['print_dpi'], 
                        f_params['sharpen'],
                        f.name,
                        sidebar_data.get('add_border', False),
                        sidebar_data.get('border_size', 1.0),
                        sidebar_data.get('border_color', "#000000"),
                        st.session_state.get('icc_profile_path') if sidebar_data.get('apply_icc') else None
                    ))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for fname, res in zip(file_names, results):
                    if isinstance(res, tuple) and res[0] is not None:
                        img_bytes, ext = res
                        out_path = os.path.join(sidebar_data['export_path'], f"processed_{fname.rsplit('.', 1)[0]}.{ext}")
                        with open(out_path, "wb") as out_f: out_path_final = out_f.write(img_bytes)
                    elif isinstance(res, Exception):
                        st.error(f"Error processing {fname}: {res}")
                
                st.success("Batch Processing Complete")
    else:
        st.info("Upload files to start.")

if __name__ == "__main__":
    asyncio.run(main())
