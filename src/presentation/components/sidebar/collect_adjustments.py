from src.presentation.state.view_models import SidebarState
from src.presentation.components.sidebar.local_adjustments_ui import (
    render_local_adjustments,
)
from src.presentation.components.sidebar.exposure_ui import render_exposure_section
from src.presentation.components.sidebar.paper_toning_ui import render_paper_section
from src.presentation.components.sidebar.lab_scanner_ui import (
    render_lab_scanner_section,
)
from src.presentation.components.sidebar.retouch_ui import render_retouch_section
from src.presentation.components.sidebar.export_ui import render_export_section
from src.presentation.components.sidebar.presets_ui import render_presets
from src.presentation.components.sidebar.navigation_ui import render_navigation
from src.presentation.components.sidebar.geometry_ui import render_geometry_section
from src.presentation.components.sidebar.analysis_ui import render_analysis_section
from src.presentation.components.sidebar.helpers import (
    render_control_selectbox,
    reset_wb_settings,
)
from src.config import DEFAULT_WORKSPACE_CONFIG


def render_adjustments() -> SidebarState:
    """
    Renders the various image adjustment expanders by delegating to sub-components.
    """
    # --- Top Controls ---
    render_control_selectbox(
        "Processing Mode",
        ["C41", "B&W"],
        default_val=DEFAULT_WORKSPACE_CONFIG.process_mode,
        key="process_mode",
        on_change=reset_wb_settings,
        help_text="Choose processing mode between Color Negative (C41) and B&W Negative",
    )

    # Navigation buttons (Back, Forward, Rotate, Delete, Copy, Paste, Reset, Export)
    export_btn_sidebar, process_all_btn = render_navigation()

    # --- Geometry & Auto-Crop ---
    render_geometry_section()

    # --- Presets ---
    render_presets()

    # --- Analysis Plots (Histogram, Curve) ---
    render_analysis_section()

    # 1. Exposure & Tonality
    render_exposure_section()

    # 2. Lab Scanner Simulation
    render_lab_scanner_section()

    # 3. Color & Balance (includes Selective Color)
    render_paper_section()

    # 4. Dodge & Burn
    render_local_adjustments()

    # 5. Retouch
    render_retouch_section()

    # 6. Export
    export_data = render_export_section()
    export_data.export_btn = export_btn_sidebar
    export_data.process_btn = process_all_btn

    return export_data
