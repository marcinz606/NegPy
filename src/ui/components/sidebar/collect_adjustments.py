from src.ui.state.view_models import SidebarState
from src.ui.components.sidebar.local_adjustments_ui import (
    render_local_adjustments,
)
from src.ui.components.sidebar.exposure_ui import render_exposure_section
from src.ui.components.sidebar.paper_toning_ui import render_paper_section
from src.ui.components.sidebar.lab_scanner_ui import (
    render_lab_scanner_section,
)
from src.ui.components.sidebar.retouch_ui import render_retouch_section
from src.ui.components.sidebar.export_ui import render_export_section
from src.ui.components.sidebar.presets_ui import render_presets
from src.ui.components.sidebar.navigation_ui import render_navigation
from src.ui.components.sidebar.geometry_ui import render_geometry_section
from src.ui.components.sidebar.analysis_ui import render_analysis_section
from src.ui.components.sidebar.soft_proofing_ui import render_soft_proofing
from src.ui.components.sidebar.helpers import (
    render_control_selectbox,
    reset_wb_settings,
)
from src.kernel.system.config import DEFAULT_WORKSPACE_CONFIG


def render_adjustments() -> SidebarState:
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

    render_geometry_section()
    render_presets()
    render_analysis_section()
    render_exposure_section()
    render_lab_scanner_section()
    render_paper_section()
    render_local_adjustments()
    render_retouch_section()
    render_soft_proofing()

    export_data = render_export_section()
    export_data.export_btn = export_btn_sidebar
    export_data.process_btn = process_all_btn

    return export_data
