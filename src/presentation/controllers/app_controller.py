import numpy as np
from PIL import Image
from src.presentation.state.session_context import SessionContext
from src.presentation.services.preview_service import PreviewService
from src.presentation.services.color_service import ColorService
from src.orchestration.engine import DarkroomEngine


class AppController:
    """
    Main Application Controller (Orchestrator).
    Bridges the UI ViewModels, Background Services, and the Photometric Engine.
    """

    def __init__(self, context: SessionContext):
        self.ctx = context
        self.preview_service = PreviewService()
        self.color_service = ColorService()
        self.engine = DarkroomEngine()

    def handle_file_loading(self, current_file: dict, current_color_space: str) -> bool:
        """
        Triggers RAW loading and downsampling if the file or color space has changed.
        """
        needs_reload = (
            self.ctx.last_file != current_file["name"]
            or self.ctx.last_preview_color_space != current_color_space
        )

        if needs_reload:
            raw, dims = self.preview_service.load_linear_preview(
                current_file["path"], current_color_space
            )
            self.ctx.preview_raw = raw
            self.ctx.original_res = dims
            self.ctx.last_file = current_file["name"]
            self.ctx.last_preview_color_space = current_color_space
            return True
        return False

    def process_frame(self) -> Image.Image:
        """
        Executes the full photometric pipeline and color transformations for the current frame.
        """
        raw = self.ctx.preview_raw
        if raw is None:
            return Image.new("RGB", (100, 100), (0, 0, 0))

        # 1. Compose full settings from session state
        from src.presentation.app import get_processing_params
        import streamlit as st

        params = get_processing_params(st.session_state)

        # 2. Run Engine
        processed = self.engine.process(raw.copy(), params)

        # 3. Convert to PIL
        img_uint8 = np.clip(np.nan_to_num(processed * 255), 0, 255).astype(np.uint8)
        pil_prev = Image.fromarray(img_uint8)

        # 4. Handle Post-Processing (B&W Toning)
        is_toned = (
            params.selenium_strength != 0.0
            or params.sepia_strength != 0.0
            or params.paper_profile != "None"
        )
        if params.is_bw and not is_toned:
            pil_prev = pil_prev.convert("L")

        # 5. Apply ICC/Simulation
        color_space = self.ctx.last_preview_color_space
        if self.ctx.session.icc_profile_path:
            pil_prev = self.color_service.apply_icc_profile(
                pil_prev, color_space, self.ctx.session.icc_profile_path
            )
        else:
            pil_prev = self.color_service.simulate_on_srgb(pil_prev, color_space)

        return pil_prev
