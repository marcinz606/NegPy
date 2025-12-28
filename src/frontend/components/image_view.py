import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from typing import Optional, Dict, Any
from src.backend.config import APP_CONFIG
from src.backend.utils import plot_histogram
from src.backend.image_logic.local import generate_local_mask, calculate_luma_mask
from src.frontend.state import save_settings

def render_image_view(pil_prev: Image.Image, m_r: float, m_g: float, m_b: float) -> None:
    """
    Renders the main image viewport and handles coordinate-based interaction.
    
    Args:
        pil_prev (Image.Image): The processed PIL image to display.
        m_r (float): Mask neutralization Red gain.
        m_g (float): Mask neutralization Green gain.
        m_b (float): Mask neutralization Blue gain.
    """
    # --- Visualization: Active Mask Overlay (Rubylith) ---
    is_local_mode = st.session_state.get('pick_local', False)
    active_idx = st.session_state.get('active_adjustment_idx', -1)
    
    if is_local_mode and active_idx >= 0 and st.session_state.get('show_active_mask', True):
        adj = st.session_state.local_adjustments[active_idx]
        pw, ph = pil_prev.size
        # Generate spatial mask at preview resolution
        mask = generate_local_mask(ph, pw, adj['points'], adj['radius'], adj['feather'], 1.0)
        
        # Intersect with Tonal Mask for accurate visualization
        luma_range = adj.get('luma_range', (0.0, 1.0))
        luma_softness = adj.get('luma_softness', 0.2)
        
        # Calculate tonal mask using a linear positive version of the raw data
        # This ensures visualization matches user intuition (Highlights=High, Shadows=Low)
        img_lin = st.session_state.preview_raw.copy()
        
        # Apply mask and WB gains to match the processing pipeline
        img_lin[:,:,0] *= (m_r * st.session_state.get('wb_manual_r', 1.0))
        img_lin[:,:,1] *= (m_g * st.session_state.get('wb_manual_g', 1.0))
        img_lin[:,:,2] *= (m_b * st.session_state.get('wb_manual_b', 1.0))
        img_lin = np.clip(img_lin, 0, 1)
        
        # Invert to Positive
        img_pos_lin = 1.0 - img_lin
        
        if img_pos_lin.shape[:2] != (ph, pw):
            img_pos_lin = cv2.resize(img_pos_lin, (pw, ph), interpolation=cv2.INTER_AREA)
            
        luma_mask = calculate_luma_mask(img_pos_lin, luma_range, luma_softness)
        final_vis_mask = mask * luma_mask
        
        # Create red overlay
        mask_u8 = (final_vis_mask * 180).astype(np.uint8) # Max 180/255 opacity
        mask_pil = Image.fromarray(mask_u8, mode='L')
        overlay = Image.new("RGBA", pil_prev.size, (255, 0, 0, 0))
        red_fill = Image.new("RGBA", pil_prev.size, (255, 75, 75, 255))
        overlay = Image.composite(red_fill, overlay, mask_pil)
        
        # Composite over preview
        if pil_prev.mode != 'RGBA':
            pil_prev = pil_prev.convert("RGBA")
        pil_prev = Image.alpha_composite(pil_prev, overlay).convert("RGB")

    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        is_dust_mode = st.session_state.get('pick_dust', False)
        display_width = APP_CONFIG['display_width']
        
        img_display = pil_prev.copy()
        if img_display.width > display_width:
            img_display.thumbnail((display_width, display_width * 4))
        
        if st.session_state.pick_wb:
            value = streamlit_image_coordinates(img_display, key=f"wb_picker_{img_display.width}", use_column_width=False)
            st.info("Click on a neutral grey area.")
        elif is_dust_mode:
            value = streamlit_image_coordinates(img_display, key=f"dust_picker_{img_display.width}", use_column_width=False)
            st.info("Click to remove dust spot.")
        elif is_local_mode:
            value = streamlit_image_coordinates(img_display, key=f"local_picker_{img_display.width}", use_column_width=False)
        else:
            st.image(pil_prev, width="content")
            value = None
            
        _, c_hist, _ = st.columns([1, 3, 1])
        with c_hist:
            st.pyplot(plot_histogram(np.array(pil_prev.convert("RGB")), figsize=(5, 0.8), dpi=200), width="stretch")

    # Click Handling Logic
    if value:
        if st.session_state.pick_wb and value != st.session_state.last_wb_click:
            st.session_state.last_wb_click = value
            st.session_state.pick_wb = False
            px, py = value['x'], value['y']
            pw, ph = img_display.size
            h_p, w_p = st.session_state.preview_raw.shape[:2]
            rx, ry = int((px / pw) * w_p), int((py / ph) * h_p)
            sample_win = 3
            y_s, y_e = max(0, ry-sample_win), min(h_p, ry+sample_win+1)
            x_s, x_e = max(0, rx-sample_win), min(w_p, rx+sample_win+1)
            clicked_rgb = np.mean(st.session_state.preview_raw[y_s:y_e, x_s:x_e], axis=(0,1))
            if not (np.any(np.isnan(clicked_rgb)) or np.all(clicked_rgb == 0)):
                mask_neut = np.array([m_r, m_g, m_b])
                target_neg_val = 0.3
                raw_wb_gains = target_neg_val / (clicked_rgb * mask_neut + 1e-6)
                norm_wb_gains = np.clip(raw_wb_gains / (raw_wb_gains[1] + 1e-6), 0.1, 10.0)
                st.session_state.wb_manual_r, st.session_state.wb_manual_g, st.session_state.wb_manual_b = map(float, norm_wb_gains)
                save_settings(st.session_state.uploaded_files[st.session_state.selected_file_idx].name)
                st.rerun()

        elif is_dust_mode and value != st.session_state.last_dust_click:
            st.session_state.last_dust_click = value
            px, py = value['x'], value['y']
            pw, ph = img_display.size
            nx, ny = px / pw, py / ph
            if 'manual_dust_spots' not in st.session_state:
                st.session_state.manual_dust_spots = []
            
            if st.session_state.get('dust_scratch_mode'):
                if st.session_state.dust_start_point is None:
                    st.session_state.dust_start_point = (nx, ny)
                    st.toast("Start point set. Click end point.")
                    st.rerun()
                else:
                    sx, sy = st.session_state.dust_start_point
                    norm_radius = st.session_state.get('manual_dust_size', 10) / float(APP_CONFIG['preview_max_res'])
                    step_size = norm_radius * 0.5
                    if step_size < 0.0005: step_size = 0.0005
                    dist = np.hypot(nx - sx, ny - sy)
                    num_steps = int(dist / step_size)
                    if num_steps < 1:
                        st.session_state.manual_dust_spots.append((nx, ny))
                    else:
                        for i in range(num_steps + 1):
                            t = i / num_steps
                            ix = sx + (nx - sx) * t
                            iy = sy + (ny - sy) * t
                            st.session_state.manual_dust_spots.append((ix, iy))
                    st.session_state.dust_start_point = None
                    save_settings(st.session_state.uploaded_files[st.session_state.selected_file_idx].name)
                    st.toast("Scratch removed.")
                    st.rerun()
            else:
                st.session_state.manual_dust_spots.append((nx, ny))
                st.session_state.dust_start_point = None
                save_settings(st.session_state.uploaded_files[st.session_state.selected_file_idx].name)
                st.rerun()

        elif is_local_mode:
            px, py = value['x'], value['y']
            pw, ph = img_display.size
            nx, ny = px / pw, py / ph
            if active_idx >= 0:
                points = st.session_state.local_adjustments[active_idx].get('points', [])
                if not points or (nx != points[-1][0] or ny != points[-1][1]):
                    points.append((nx, ny))
                    st.session_state.local_adjustments[active_idx]['points'] = points
                    save_settings(st.session_state.uploaded_files[st.session_state.selected_file_idx].name)
                    st.rerun()