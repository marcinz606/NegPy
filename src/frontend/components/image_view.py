import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps
from streamlit_image_coordinates import streamlit_image_coordinates
from typing import Optional, Dict, Any
from src.backend.config import APP_CONFIG
from src.backend.utils import transform_point
from src.backend.image_logic.local import generate_local_mask, calculate_luma_mask
from src.backend.image_logic.retouch import get_autocrop_coords
from src.frontend.state import save_settings

def render_image_view(pil_prev: Image.Image, border_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Renders the main image viewport and handles coordinate-based interaction.
    """
    border_px = 0
    orig_w, orig_h = pil_prev.size

    # --- Border Preview ---
    if border_config and border_config.get('add_border', False):
        try:
            print_w = float(border_config.get('print_width_cm', 27.0))
            border_w = float(border_config.get('size_cm', 0.2))
            color = border_config.get('color', '#000000')
            
            # Calculate border width in pixels relative to current preview size
            # If print_w represents the longer side (usually), we use that ratio.
            longer_side_px = max(pil_prev.size)
            ratio = border_w / print_w if print_w > 0 else 0.01
            border_px = int(longer_side_px * ratio)
            
            if border_px > 0:
                pil_prev = ImageOps.expand(pil_prev, border=border_px, fill=color)
        except Exception as e:
            print(f"Border preview error: {e}")

    # 1. State & Config
    is_local_mode = st.session_state.get('pick_local', False)
    active_idx = st.session_state.get('active_adjustment_idx', -1)
    img_raw = st.session_state.preview_raw
    rh_orig, rw_orig = img_raw.shape[:2]
    rotation = st.session_state.get('rotation', 0) % 4
    fine_rot = st.session_state.get('fine_rotation', 0.0)
    autocrop = st.session_state.get('autocrop', False)
    autocrop_offset = st.session_state.get('autocrop_offset', 0)
    autocrop_ratio = st.session_state.get('autocrop_ratio', '3:2')

    # --- Foolproof Geometry Mapping via Coordinate Grid ---
    # Create UV grid for RAW (u=x/w, v=y/h)
    u_raw, v_raw = np.meshgrid(np.linspace(0, 1, rw_orig), np.linspace(0, 1, rh_orig))
    uv_grid = np.stack([u_raw, v_raw], axis=-1).astype(np.float32)

    # Apply EXACT SAME Geometry Pipeline as process_image_core
    if rotation != 0:
        uv_grid = np.rot90(uv_grid, k=rotation)
    
    if fine_rot != 0.0:
        h_r, w_r = uv_grid.shape[:2]
        M = cv2.getRotationMatrix2D((w_r / 2.0, h_r / 2.0), fine_rot, 1.0)
        uv_grid = cv2.warpAffine(uv_grid, M, (w_r, h_r), flags=cv2.INTER_LINEAR)
        
    if autocrop:
        img_geom = np.rot90(img_raw, k=rotation)
        if fine_rot != 0.0:
            h_f, w_f = img_geom.shape[:2]
            M_f = cv2.getRotationMatrix2D((w_f/2, h_f/2), fine_rot, 1.0)
            img_geom = cv2.warpAffine(img_geom, M_f, (w_f, h_f))
        y1, y2, x1, x2 = get_autocrop_coords(img_geom, autocrop_offset, 1.0, autocrop_ratio)
        uv_grid = uv_grid[y1:y2, x1:x2]

    def map_click_to_raw(nx, ny):
        h_uv, w_uv = uv_grid.shape[:2]
        px = int(np.clip(nx * (w_uv - 1), 0, w_uv - 1))
        py = int(np.clip(ny * (h_uv - 1), 0, h_uv - 1))
        raw_uv = uv_grid[py, px]
        return float(raw_uv[0]), float(raw_uv[1])

    # --- Visualization: Active Mask Overlay (Rubylith) ---
    if is_local_mode and active_idx >= 0 and st.session_state.get('show_active_mask', True):
        adj = st.session_state.local_adjustments[active_idx]
        # Generate mask in RAW space
        mask = generate_local_mask(rh_orig, rw_orig, adj['points'], adj['radius'], adj['feather'], 1.0)
        # Tonal Mask (RAW)
        img_lin = img_raw.copy()
        img_lin[:,:,0] *= st.session_state.get('wb_manual_r', 1.0)
        img_lin[:,:,1] *= st.session_state.get('wb_manual_g', 1.0)
        img_lin[:,:,2] *= st.session_state.get('wb_manual_b', 1.0)
        img_pos_lin = 1.0 - np.clip(img_lin, 0, 1)
        luma_mask = calculate_luma_mask(img_pos_lin, adj.get('luma_range', (0.0, 1.0)), adj.get('luma_softness', 0.2))
        final_vis_mask = mask * luma_mask
        
        # Apply Geometry to Mask
        if rotation != 0: final_vis_mask = np.rot90(final_vis_mask, k=rotation)
        if fine_rot != 0.0:
            h_m, w_m = final_vis_mask.shape[:2]
            M_m = cv2.getRotationMatrix2D((w_m/2, h_m/2), fine_rot, 1.0)
            final_vis_mask = cv2.warpAffine(final_vis_mask, M_m, (w_m, h_m), flags=cv2.INTER_LINEAR)
        if autocrop:
            # We already have img_geom from uv_grid logic above
            y1, y2, x1, x2 = get_autocrop_coords(img_geom, autocrop_offset, 1.0, autocrop_ratio)
            final_vis_mask = final_vis_mask[y1:y2, x1:x2]

        # Composite
        pw, ph = pil_prev.size
        # IMPORTANT: Mask is for CONTENT only. If border exists, we need to pad the mask to match.
        # But wait, pil_prev NOW has border.
        # final_vis_mask corresponds to orig_w, orig_h (content).
        # We should expand the mask with 0 (transparent) border.
        
        # First resize mask to content dimensions
        if final_vis_mask.shape[:2] != (orig_h, orig_w):
             final_vis_mask = cv2.resize(final_vis_mask, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
             
        mask_u8 = (final_vis_mask * 180).astype(np.uint8)
        
        if border_px > 0:
            # Pad the mask to match bordered image size
            mask_u8 = cv2.copyMakeBorder(mask_u8, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=0)
            
        mask_pil = Image.fromarray(mask_u8, mode='L')
        overlay = Image.new("RGBA", pil_prev.size, (255, 0, 0, 0))
        red_fill = Image.new("RGBA", pil_prev.size, (255, 75, 75, 255))
        overlay = Image.composite(red_fill, overlay, mask_pil)
        if pil_prev.mode != 'RGBA': pil_prev = pil_prev.convert("RGBA")
        pil_prev = Image.alpha_composite(pil_prev, overlay).convert("RGB")

    # Main UI Render
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        if st.session_state.uploaded_files:
            current_file = st.session_state.uploaded_files[st.session_state.selected_file_idx]
            st.subheader(current_file.name)

        is_dust_mode = st.session_state.get('pick_dust', False)
        display_width = APP_CONFIG['display_width']
        img_display = pil_prev.copy()
        
        # Ensure longer side does not exceed display_width
        if max(img_display.size) > display_width:
            img_display.thumbnail((display_width, display_width))
        
        if is_dust_mode:
            value = streamlit_image_coordinates(img_display, key=f"dust_picker_{img_display.width}", use_column_width=False)
            st.info("Click to remove dust spot.")
        elif is_local_mode:
            value = streamlit_image_coordinates(img_display, key=f"local_picker_{img_display.width}", use_column_width=False)
        else:
            # Display at its actual size (which is already capped at display_width)
            st.image(img_display)
            value = None
            
    # Click Handling Logic
    if value:
        # value['x'], value['y'] are coordinates in img_display (which might be a thumbnail of the bordered pil_prev)
        # We need to map them back to original content coordinates.
        
        # 1. Map from Thumbnail to Full Resolution (Bordered)
        scale_x = pil_prev.width / img_display.width
        scale_y = pil_prev.height / img_display.height
        
        abs_x = value['x'] * scale_x
        abs_y = value['y'] * scale_y
        
        # 2. Map from Bordered to Content
        content_x = abs_x - border_px
        content_y = abs_y - border_px
        
        # 3. Check Bounds & Normalize
        if 0 <= content_x < orig_w and 0 <= content_y < orig_h:
            nx = content_x / orig_w
            ny = content_y / orig_h
            
            rx, ry = map_click_to_raw(nx, ny)

            if is_dust_mode and value != st.session_state.last_dust_click:
                st.session_state.last_dust_click = value
                if 'manual_dust_spots' not in st.session_state: st.session_state.manual_dust_spots = []
                if st.session_state.get('dust_scratch_mode'):
                    if st.session_state.dust_start_point is None:
                        st.session_state.dust_start_point = (rx, ry)
                        st.toast("Start point set. Click end point.")
                        st.rerun()
                    else:
                        sx, sy = st.session_state.dust_start_point
                        norm_radius = st.session_state.get('manual_dust_size', 10) / float(APP_CONFIG['preview_max_res'])
                        step_size = max(0.0005, norm_radius * 0.5)
                        dist = np.hypot(rx - sx, ry - sy)
                        num_steps = int(dist / step_size)
                        for i in range(num_steps + 1):
                            t = i / max(1, num_steps)
                            st.session_state.manual_dust_spots.append((sx + (rx - sx) * t, sy + (ry - sy) * t))
                        st.session_state.dust_start_point = None
                        save_settings(st.session_state.uploaded_files[st.session_state.selected_file_idx].name)
                        st.toast("Scratch removed.")
                        st.rerun()
                else:
                    st.session_state.manual_dust_spots.append((rx, ry))
                    save_settings(st.session_state.uploaded_files[st.session_state.selected_file_idx].name)
                    st.rerun()

            elif is_local_mode:
                if active_idx >= 0:
                    points = st.session_state.local_adjustments[active_idx].get('points', [])
                    if not points or (rx != points[-1][0] or ry != points[-1][1]):
                        points.append((rx, ry))
                        st.session_state.local_adjustments[active_idx]['points'] = points
                        save_settings(st.session_state.uploaded_files[st.session_state.selected_file_idx].name)
                        st.rerun()
