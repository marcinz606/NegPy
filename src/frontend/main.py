import streamlit as st
import rawpy
import numpy as np
import io
import os
import asyncio
import concurrent.futures
import cv2
import traceback
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

from src.backend.config import TONE_CURVES_PRESETS, DEFAULT_SETTINGS, APP_CONFIG
from src.backend.utils import create_curve_lut, apply_color_separation, get_thumbnail_worker
from src.backend.db import init_db
from src.backend.processor import (
    process_image_core, 
    load_raw_and_process, 
    calculate_auto_mask_wb
)
from src.frontend.state import init_session_state, load_settings, save_settings
from src.frontend.components.sidebar.main import render_sidebar
from src.frontend.components.image_view import render_image_view
from src.frontend.components.contact_sheet import render_contact_sheet

def init_styles():
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
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

    sidebar_data = render_sidebar()
    
    if sidebar_data:
        uploaded_files = st.session_state.uploaded_files
        current_file = uploaded_files[st.session_state.selected_file_idx]
        
        save_settings(current_file.name)

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

        # Preview Loading
        if st.session_state.get("last_file") != current_file.name:
            with st.spinner(f"Loading preview for {current_file.name}..."):
                current_file.seek(0)
                with rawpy.imread(current_file) as raw:
                    rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, output_bps=16)
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

        # Processing Parameters
        m_r, m_g, m_b = 1.0, 1.0, 1.0
        if st.session_state.auto_wb:
            m_r, m_g, m_b = calculate_auto_mask_wb(st.session_state.preview_raw)
            st.sidebar.caption(f"Mask Neutralization: R={m_r:.2f}, G={m_g:.2f}, B={m_b:.2f}")

        cx_base, cy_base = TONE_CURVES_PRESETS[st.session_state.get('curve_mode', 'Linear')]
        cy = np.array(cx_base) + (np.array(cy_base) - np.array(cx_base)) * st.session_state.get('curve_strength', 1.0)
        lut_x, lut_y = create_curve_lut(cx_base, cy)

        bw_val = st.session_state.get('bw_points', (0.0, 1.0))
        current_params = {
            'mask_r': m_r, 'mask_g': m_g, 'mask_b': m_b,
            'wb_manual_r': st.session_state.get('wb_manual_r', 1.0),
            'wb_manual_g': st.session_state.get('wb_manual_g', 1.0),
            'wb_manual_b': st.session_state.get('wb_manual_b', 1.0),
            'cr_balance': st.session_state.get('cr_balance', 1.0), 
            'mg_balance': st.session_state.get('mg_balance', 1.0), 
            'yb_balance': st.session_state.get('yb_balance', 1.0),
            'shadow_cr': st.session_state.get('shadow_cr', 1.0), 'shadow_mg': st.session_state.get('shadow_mg', 1.0), 'shadow_yb': st.session_state.get('shadow_yb', 1.0),
            'highlight_cr': st.session_state.get('highlight_cr', 1.0), 'highlight_mg': st.session_state.get('highlight_mg', 1.0), 'highlight_yb': st.session_state.get('highlight_yb', 1.0),
            'temperature': st.session_state.get('temperature', 0.0),
            'shadow_temp': st.session_state.get('shadow_temp', 0.0),
            'highlight_temp': st.session_state.get('highlight_temp', 0.0),
            'gamma': st.session_state.get('gamma', 2.5),
            'black_point': bw_val[0],
            'white_point': bw_val[1],
            'exposure': st.session_state.exposure, 
            'contrast': st.session_state.get('contrast', 1.0),
            'grade_shadows': st.session_state.grade_shadows,
            'grade_highlights': st.session_state.grade_highlights,
            'saturation': st.session_state.get('saturation', 1.0), 
            'curve_lut_x': lut_x, 'curve_lut_y': lut_y,
            'autocrop': st.session_state.get('autocrop', True), 
            'autocrop_offset': st.session_state.get('autocrop_offset', 5),
            'dust_remove': st.session_state.get('dust_remove', True), 
            'dust_threshold': st.session_state.get('dust_threshold', 0.6), 
            'dust_size': st.session_state.get('dust_size', 2),
            'manual_dust_spots': st.session_state.get('manual_dust_spots', []),
            'manual_dust_size': st.session_state.get('manual_dust_size', 10),
            'c_noise_remove': st.session_state.get('c_noise_remove', True), 
            'c_noise_strength': st.session_state.get('c_noise_strength', 33),
            'local_adjustments': st.session_state.get('local_adjustments', []),
            'rotation': st.session_state.get('rotation', 0),
            'fine_rotation': st.session_state.get('fine_rotation', 0.0),
            'monochrome': st.session_state.get('monochrome', False),
            'auto_wb': st.session_state.get('auto_wb', False)
        }
        
        # Add Selective Color params
        sel_colors = ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']
        for c in sel_colors:
            for attr in ['hue', 'sat', 'lum', 'range']:
                k = f"selective_{c}_{attr}"
                current_params[k] = st.session_state.get(k, 0.0)

        # Core Processing
        processed_preview = process_image_core(st.session_state.preview_raw.copy(), current_params)
        pil_prev = Image.fromarray(np.clip(np.nan_to_num(processed_preview * 255), 0, 255).astype(np.uint8))
        
        # Sharpening
        sharpen_val = st.session_state.get('sharpen', 0.75)
        if sharpen_val > 0:
            img_lab = cv2.cvtColor(np.array(pil_prev), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_lab)
            l_pil = Image.fromarray(l)
            l_sharpened = l_pil.filter(ImageFilter.UnsharpMask(radius=1.0, percent=int(sharpen_val * 250), threshold=5))
            img_lab_sharpened = cv2.merge([np.array(l_sharpened), a, b])
            pil_prev = Image.fromarray(cv2.cvtColor(img_lab_sharpened, cv2.COLOR_LAB2RGB))

        # Saturation/Mono
        if not current_params.get('monochrome', False):
            img_arr = np.array(pil_prev)
            img_sep = apply_color_separation(img_arr, st.session_state.get('saturation', 1.0))
            pil_prev = Image.fromarray(img_sep)
        else:
            pil_prev = pil_prev.convert("L")

        # Visualization
        if st.session_state.get('pick_dust', False) and st.session_state.get('show_dust_patches', True):
            if st.session_state.get('manual_dust_spots'):
                if pil_prev.mode == 'L': pil_prev = pil_prev.convert("RGB")
                draw = ImageDraw.Draw(pil_prev)
                pw, ph = pil_prev.size
                m_rad = st.session_state.get('manual_dust_size', 10)
                for (nx, ny) in st.session_state.manual_dust_spots:
                    px, py = nx * pw, ny * ph
                    draw.ellipse((px - m_rad, py - m_rad, px + m_rad, py + m_rad), outline="#ff4b4b", width=4)
                if st.session_state.get('dust_scratch_mode') and st.session_state.get('dust_start_point'):
                    sx, sy = st.session_state.dust_start_point
                    spx, spy = sx * pw, sy * ph
                    draw.ellipse((spx - m_rad, spy - m_rad, spx + m_rad, spy + m_rad), outline="#f1c40f", width=4)

        # Contact Sheet
        render_contact_sheet(uploaded_files)

        # Main UI Render
        render_image_view(pil_prev, m_r, m_g, m_b)
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
                f_params = {
                    'mask_r': m_r, 'mask_g': m_g, 'mask_b': m_b,
                    'wb_manual_r': f_settings.get('wb_manual_r', 1.0), 'wb_manual_g': f_settings.get('wb_manual_g', 1.0), 'wb_manual_b': f_settings.get('wb_manual_b', 1.0),
                    'cr_balance': f_settings['cr_balance'], 'mg_balance': f_settings['mg_balance'], 'yb_balance': f_settings['yb_balance'],
                    'shadow_cr': f_settings.get('shadow_cr', 1.0), 'shadow_mg': f_settings.get('shadow_mg', 1.0), 'shadow_yb': f_settings.get('shadow_yb', 1.0),
                    'highlight_cr': f_settings.get('highlight_cr', 1.0), 'highlight_mg': f_settings.get('highlight_mg', 1.0), 'highlight_yb': f_settings.get('highlight_yb', 1.0),
                    'temperature': f_settings['temperature'],
                    'shadow_temp': f_settings.get('shadow_temp', 0.0), 'highlight_temp': f_settings.get('highlight_temp', 0.0),
                    'gamma': f_settings.get('gamma', 2.5),
                    'black_point': f_settings.get('black_point', 0.0),
                    'white_point': f_settings.get('white_point', 1.0),
                    'exposure': f_settings['exposure'], 
                    'contrast': f_settings.get('contrast', 1.0),
                    'grade_shadows': f_settings['grade_shadows'], 'grade_highlights': f_settings['grade_highlights'],
                    'saturation': f_settings['saturation'],
                    'curve_lut_x': lx, 'curve_lut_y': ly,
                    'autocrop': f_settings['autocrop'], 'autocrop_offset': f_settings['autocrop_offset'],
                    'dust_remove': f_settings['dust_remove'], 'dust_threshold': f_settings['dust_threshold'], 'dust_size': f_settings['dust_size'],
                    'manual_dust_spots': f_settings.get('manual_dust_spots', []),
                    'manual_dust_size': f_settings.get('manual_dust_size', 10),
                    'c_noise_remove': f_settings['c_noise_remove'], 'c_noise_strength': f_settings['c_noise_strength'],
                    'local_adjustments': f_settings.get('local_adjustments', []),
                    'rotation': f_settings['rotation'], 'fine_rotation': f_settings.get('fine_rotation', 0.0), 'monochrome': f_settings['monochrome'],
                    'sharpen': f_settings.get('sharpen', 0.75)
                }
                
                for c in ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']:
                    for attr in ['hue', 'sat', 'lum', 'range']:
                        k = f"selective_{c}_{attr}"
                        f_params[k] = f_settings.get(k, 0.0)

                img_bytes, ext = load_raw_and_process(
                    f_current.getvalue(), 
                    f_params, 
                    sidebar_data['out_fmt'], 
                    sidebar_data['print_width'], 
                    sidebar_data['print_dpi'], 
                    f_params['sharpen'],
                    save_training_data=st.session_state.get('collect_training_data', False),
                    filename=f_current.name
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
                collect_data = st.session_state.get('collect_training_data', False)
                for f in uploaded_files:
                    f_settings = st.session_state.file_settings.get(f.name, DEFAULT_SETTINGS.copy())
                    cx_b, cy_b = TONE_CURVES_PRESETS[f_settings['curve_mode']]
                    cy_f = np.array(cx_b) + (np.array(cy_b) - np.array(cx_b)) * f_settings['curve_strength']
                    lx, ly = create_curve_lut(cx_b, cy_f)
                    f_params = {
                        'auto_wb': f_settings['auto_wb'],
                        'cr_balance': f_settings['cr_balance'], 'mg_balance': f_settings['mg_balance'], 'yb_balance': f_settings['yb_balance'],
                        'shadow_cr': f_settings.get('shadow_cr', 1.0), 'shadow_mg': f_settings.get('shadow_mg', 1.0), 'shadow_yb': f_settings.get('shadow_yb', 1.0),
                        'highlight_cr': f_settings.get('highlight_cr', 1.0), 'highlight_mg': f_settings.get('highlight_mg', 1.0), 'highlight_yb': f_settings.get('highlight_yb', 1.0),
                        'temperature': f_settings['temperature'],
                        'shadow_temp': f_settings.get('shadow_temp', 0.0), 'highlight_temp': f_settings.get('highlight_temp', 0.0),
                        'gamma': f_settings.get('gamma', 2.5),
                        'black_point': f_settings.get('black_point', 0.0),
                        'white_point': f_settings.get('white_point', 1.0),
                        'exposure': f_settings['exposure'], 
                        'contrast': f_settings.get('contrast', 1.0),
                        'grade_shadows': f_settings['grade_shadows'], 'grade_highlights': f_settings['grade_highlights'],
                        'saturation': f_settings['saturation'],
                        'curve_lut_x': lx, 'curve_lut_y': ly,
                        'autocrop': f_settings['autocrop'], 'autocrop_offset': f_settings['autocrop_offset'],
                        'dust_remove': f_settings['dust_remove'], 'dust_threshold': f_settings['dust_threshold'], 'dust_size': f_settings['dust_size'],
                        'manual_dust_spots': f_settings.get('manual_dust_spots', []),
                        'manual_dust_size': f_settings.get('manual_dust_size', 10),
                        'c_noise_remove': f_settings['c_noise_remove'], 'c_noise_strength': f_settings['c_noise_strength'],
                        'local_adjustments': f_settings.get('local_adjustments', []),
                        'rotation': f_settings['rotation'], 'fine_rotation': f_settings.get('fine_rotation', 0.0), 'monochrome': f_settings['monochrome'],
                        'wb_manual_r': f_settings.get('wb_manual_r', 1.0), 'wb_manual_g': f_settings.get('wb_manual_g', 1.0), 'wb_manual_b': f_settings.get('wb_manual_b', 1.0),
                        'sharpen': f_settings.get('sharpen', 0.75)
                    }
                    
                    for c in ['red', 'orange', 'yellow', 'green', 'aqua', 'blue', 'purple', 'magenta']:
                        for attr in ['hue', 'sat', 'lum', 'range']:
                            k = f"selective_{c}_{attr}"
                            f_params[k] = f_settings.get(k, 0.0)

                    tasks.append(loop.run_in_executor(
                        executor, 
                        load_raw_and_process, 
                        f.getvalue(), 
                        f_params, 
                        sidebar_data['out_fmt'], 
                        sidebar_data['print_width'], 
                        sidebar_data['print_dpi'], 
                        f_params['sharpen'],
                        collect_data,
                        f.name
                    ))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                st.success("Batch Processing Complete")
    else:
        st.info("Upload files to start.")

if __name__ == "__main__":
    asyncio.run(main())
