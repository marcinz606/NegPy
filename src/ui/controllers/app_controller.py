import streamlit as st
import os
import numpy as np
from PIL import Image
from src.ui.state.session_context import SessionContext
from src.services.rendering.preview_manager import PreviewManager
from src.services.rendering.image_processor import ImageProcessor
from src.infrastructure.display.color_mgmt import ColorService
from src.infrastructure.filesystem.watcher import FolderWatchService
from src.features.exposure.logic import calculate_wb_shifts
from src.ui.state.view_models import ExposureViewModel


class AppController:
    """
    Glue between UI and Pixel Engine.
    """

    def __init__(self, context: SessionContext):
        self.ctx = context
        self.preview_service = PreviewManager()
        self.color_service = ColorService()
        self.folder_watch_service = FolderWatchService()
        self.image_service = ImageProcessor()

    def sync_hot_folders(self) -> bool:
        """
        Polls watched directories for new files.
        """
        session = self.ctx.session
        if not st.session_state.get("hot_folder_mode"):
            return False

        for f in session.uploaded_files:
            session.watched_folders.add(os.path.dirname(f["path"]))

        if not session.watched_folders:
            return False

        existing_paths = {f["path"] for f in session.uploaded_files}
        new_discovered = []

        for folder in session.watched_folders:
            new_discovered.extend(
                self.folder_watch_service.scan_for_new_files(folder, existing_paths)
            )

        if new_discovered:
            session.add_local_assets(new_discovered)
            return True
        return False

    def handle_file_loading(self, current_file: dict, current_color_space: str) -> bool:
        """
        Reloads linear RAW if file/CS changed.
        """
        needs_reload = (
            self.ctx.last_file != current_file["name"]
            or self.ctx.last_preview_color_space != current_color_space
        )

        if needs_reload:
            raw, dims, metadata = self.preview_service.load_linear_preview(
                current_file["path"], current_color_space
            )
            self.ctx.preview_raw = raw
            self.ctx.original_res = dims
            self.ctx.last_file = current_file["name"]
            self.ctx.last_preview_color_space = current_color_space

            # Apply camera orientation if this is a fresh load (no settings yet)
            f_hash = current_file.get("hash")
            if f_hash and f_hash not in self.ctx.session.file_settings:
                detected_rot = metadata.get("orientation", 0)
                if detected_rot != 0:
                    st.session_state.rotation = detected_rot

            if "last_metrics" in st.session_state:
                del st.session_state.last_metrics
            if "base_positive" in st.session_state:
                del st.session_state.base_positive
            if "retouch_source" in st.session_state:
                del st.session_state.retouch_source

            return True
        return False

    def handle_wb_pick(self, nx: float, ny: float) -> None:
        """
        Calculates and applies WB shifts from a sampled point.
        """
        img = st.session_state.get("base_positive")
        if img is None:
            return

        h, w = img.shape[:2]
        px = int(np.clip(nx * w, 0, w - 1))
        py = int(np.clip(ny * h, 0, h - 1))
        sampled = img[py, px]

        dm, dy = calculate_wb_shifts(sampled)

        vm = ExposureViewModel()
        st.session_state[vm.get_key("wb_cyan")] = 0.0
        st.session_state[vm.get_key("wb_magenta")] = float(np.clip(dm, -1, 1))
        st.session_state[vm.get_key("wb_yellow")] = float(np.clip(dy, -1, 1))
        st.session_state[vm.get_key("pick_wb")] = False
        st.toast(f"WB Picked: M={dm:.2f}, Y={dy:.2f}")

        from src.ui.state.state_manager import save_settings

        save_settings()

    def process_frame(self) -> Image.Image:
        """
        Runs pipeline -> Color Mgmt -> Display PIL.
        """
        raw = self.ctx.preview_raw
        if raw is None:
            return Image.new("RGB", (100, 100), (0, 0, 0))

        from src.ui.app import get_processing_params_composed

        params = get_processing_params_composed(st.session_state)

        f_hash = (
            self.ctx.session.current_file["hash"]
            if self.ctx.session.current_file
            else ""
        )

        buffer, metrics = self.image_service.run_pipeline(
            raw.copy(),
            params,
            f_hash,
            render_size_ref=float(self.image_service.engine.config.preview_render_size),
        )

        pil_prev = self.image_service.buffer_to_pil(buffer, params, bit_depth=8)

        st.session_state.last_metrics = metrics
        if "base_positive" in metrics:
            st.session_state.base_positive = metrics["base_positive"]
        if "retouch_source" in metrics:
            st.session_state.retouch_source = metrics["retouch_source"]

        color_space = self.ctx.last_preview_color_space
        target_icc = self.ctx.session.icc_profile_path
        inverse = self.ctx.session.icc_invert

        if target_icc:
            pil_prev = self.color_service.apply_icc_profile(
                pil_prev, color_space, target_icc, inverse=inverse
            )
        else:
            pil_prev = self.color_service.simulate_on_srgb(pil_prev, color_space)

        return pil_prev
