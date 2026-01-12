import streamlit as st
from src.features.retouch.models import LocalAdjustmentConfig
from src.ui.components.sidebar.helpers import (
    render_control_checkbox,
    render_control_selectbox,
)


def render_local_adjustments() -> None:
    with st.expander(":material/pen_size_5: Dodge & Burn", expanded=False):
        st.subheader("Adjustment Masks")

        c1, c2 = st.columns([2, 1])
        if c1.button("Add Mask", width="stretch"):
            new_adj = LocalAdjustmentConfig(
                strength=0.0,
                radius=50,
                feather=0.5,
                luma_range=(0.0, 1.0),
                luma_softness=0.2,
                points=[],
            )

            st.session_state.local_adjustments.append(new_adj)
            st.session_state.active_adjustment_idx = (
                len(st.session_state.local_adjustments) - 1
            )
            st.rerun()

        if st.session_state.get("local_adjustments"):
            adj_names = [
                f"{i + 1}. ({'Dodge' if a.strength > 0 else 'Burn' if a.strength < 0 else 'Neutral'})"
                for i, a in enumerate(st.session_state.local_adjustments)
            ]

            current_idx = st.session_state.get("active_adjustment_idx", 0)
            if current_idx < 0 or current_idx >= len(adj_names):
                st.session_state.active_adjustment_idx = 0

            selected_idx = render_control_selectbox(
                "Active Layer",
                options=list(range(len(adj_names))),
                default_val=0,
                key="active_adjustment_idx",
                format_func=lambda x: adj_names[x],
            )

            st.session_state.active_adjustment_idx = selected_idx
            active_adj = st.session_state.local_adjustments[selected_idx]

            c1, c2 = st.columns(2)
            if c2.button("Delete Layer", width="stretch"):
                st.session_state.local_adjustments.pop(selected_idx)
                # Select the previous layer or the first one
                st.session_state.active_adjustment_idx = max(0, selected_idx - 1)
                st.rerun()

            st.markdown("---")

            s_key = f"adj_str_{selected_idx}"
            if s_key not in st.session_state:
                st.session_state[s_key] = float(active_adj.strength)
            active_adj.strength = st.slider(
                "Exposure (EV)",
                -1.0,
                1.0,
                step=0.01,
                key=s_key,
            )

            r_key = f"adj_rad_{selected_idx}"
            if r_key not in st.session_state:
                st.session_state[r_key] = int(active_adj.radius)
            active_adj.radius = int(
                st.slider(
                    "Brush Size",
                    5,
                    250,
                    step=1,
                    key=r_key,
                )
            )

            f_key = f"adj_fth_{selected_idx}"
            if f_key not in st.session_state:
                st.session_state[f_key] = float(active_adj.feather)
            active_adj.feather = st.slider(
                "Feathering",
                0.0,
                1.0,
                step=0.05,
                key=f_key,
            )

            st.caption("Targeting (Range)")
            lr_key = f"adj_lr_{selected_idx}"
            if lr_key not in st.session_state:
                st.session_state[lr_key] = active_adj.luma_range
            active_adj.luma_range = st.slider(
                "Luminance Range",
                0.0,
                1.0,
                key=lr_key,
            )

            ls_key = f"adj_ls_{selected_idx}"
            if ls_key not in st.session_state:
                st.session_state[ls_key] = float(active_adj.luma_softness)
            active_adj.luma_softness = st.slider(
                "Range Softness",
                0.0,
                1.0,
                step=0.01,
                key=ls_key,
            )

            c1, c2 = st.columns(2)
            if c1.button("Clear Brush", width="stretch"):
                active_adj.points = []
                st.rerun()

            render_control_checkbox(
                "Show Mask Overlay", default_val=True, key="show_active_mask"
            )

            render_control_checkbox(
                ":material/brush: Paint Mode",
                default_val=False,
                key="pick_local",
                is_toggle=True,
            )

            if st.session_state.get("pick_local"):
                st.info("Click on the image to paint the adjustment.")
        else:
            st.info("No local adjustments added yet.")
            st.session_state.pick_local = False
