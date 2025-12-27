import streamlit as st
from typing import List, Dict, Any
from src.frontend.state import save_settings

def render_local_adjustments():
    """
    Renders the Local Adjustments layer manager and brush controls.
    """
    st.subheader("Adjustment Masks")
    
    if 'local_adjustments' not in st.session_state:
        st.session_state.local_adjustments = []
    
    # Layer Management
    c1, c2 = st.columns([2, 1])
    if c1.button("Add Mask", use_container_width=True):
        new_adj = {
            "name": f"Layer {len(st.session_state.local_adjustments) + 1}",
            "strength": 0.0,
            "radius": 50,
            "feather": 0.5,
            "luma_range": (0.0, 1.0),
            "luma_softness": 0.2,
            "points": []
        }
        st.session_state.local_adjustments.append(new_adj)
        st.session_state.active_adjustment_idx = len(st.session_state.local_adjustments) - 1
        st.rerun()

    if st.session_state.get('local_adjustments'):
        # List layers
        adj_names = [f"{i+1}. {a['name']} ({'Dodge' if a['strength'] > 0 else 'Burn' if a['strength'] < 0 else 'Neutral'})" 
                     for i, a in enumerate(st.session_state.local_adjustments)]
        
        selected_idx = st.selectbox("Active Layer", range(len(adj_names)), 
                                   format_func=lambda x: adj_names[x],
                                   index=max(0, st.session_state.get('active_adjustment_idx', 0)))
        
        st.session_state.active_adjustment_idx = selected_idx
        active_adj = st.session_state.local_adjustments[selected_idx]

        # Layer controls
        c1, c2 = st.columns(2)
        active_adj['name'] = c1.text_input("Layer Name", value=active_adj['name'])
        if c2.button("Delete Layer", use_container_width=True):
            st.session_state.local_adjustments.pop(selected_idx)
            st.session_state.active_adjustment_idx = -1
            st.rerun()

        # Brush controls
        st.markdown("---")
        active_adj['strength'] = st.slider("Exposure (EV)", -1.0, 1.0, active_adj['strength'], 0.01, key=f"adj_str_{selected_idx}")
        active_adj['radius'] = st.slider("Brush Size", 5, 250, active_adj['radius'], 1, key=f"adj_rad_{selected_idx}")
        active_adj['feather'] = st.slider("Feathering", 0.0, 1.0, active_adj['feather'], 0.05, key=f"adj_fth_{selected_idx}")
        
        st.caption("Targeting (Range)")
        active_adj['luma_range'] = st.slider("Luminance Range", 0.0, 1.0, active_adj.get('luma_range', (0.0, 1.0)), 0.01, key=f"adj_lr_{selected_idx}")
        active_adj['luma_softness'] = st.slider("Range Softness", 0.0, 1.0, active_adj.get('luma_softness', 0.2), 0.01, key=f"adj_ls_{selected_idx}")

        c1, c2 = st.columns(2)
        if c1.button("Clear Brush", use_container_width=True):
            active_adj['points'] = []
            st.rerun()
        
        st.checkbox("Show Mask Overlay", value=True, key="show_active_mask")
        
        # Mode toggle
        st.session_state.pick_local = st.toggle("Paint Mode", value=st.session_state.get('pick_local', False))
        
        if st.session_state.pick_local:
            st.info("Click on the image to paint the adjustment.")
    else:
        st.info("No local adjustments added yet.")
        st.session_state.pick_local = False
