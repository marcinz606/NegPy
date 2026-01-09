import streamlit as st
import os
import asyncio
from PIL import Image
from typing import Any
from src.core.models import WorkspaceConfig
from src.ui.state.state_manager import init_session_state
from src.ui.styles.theme import apply_custom_css
from src.ui.components.sidebar.files_ui import render_file_manager
from src.ui.components.sidebar.main import render_sidebar_content
from src.ui.layouts.main_layout import (
    render_layout_header,
    render_main_layout,
)
from src.ui.state.session_context import SessionContext
from src.ui.controllers.app_controller import AppController
from src.application.services.export_service import ExportService


def get_processing_params_composed(source: Any) -> WorkspaceConfig:
    """
    Consolidates parameter gathering into the modular WorkspaceConfig object.
    """
    return WorkspaceConfig.from_flat_dict(source)


async def main() -> None:
    """
    Primary Application Entry Point (OO Orchestrator).
    """
    st.set_page_config(
        page_title="DarkroomPy", layout="wide", page_icon="media/icons/icon.png"
    )
    init_session_state()

    ctx = SessionContext()
    controller = AppController(ctx)
    session = ctx.session

    if "assets_initialized" not in st.session_state:
        session.asset_store.initialize()
        st.session_state.assets_initialized = True

    apply_custom_css()

    if controller.sync_hot_folders():
        st.toast("New files discovered in hot folder!")
        st.rerun()

    render_file_manager()

    if session.uploaded_files:
        current_file = session.current_file
        if current_file and current_file["hash"] not in session.file_settings:
            from src.ui.state.state_manager import load_settings

            load_settings()

    main_area, status_area = render_layout_header(ctx)

    if session.uploaded_files:
        current_file = session.current_file
        if current_file is None:
            st.info("Please select a file.")
            return

        current_cs = st.session_state.get("export_color_space", "sRGB")

        if controller.handle_file_loading(current_file, current_cs):
            status_area.success(f"Loaded {current_file['name']}")
            st.rerun()

        sidebar_data = render_sidebar_content()

        missing_thumbs = [
            f for f in session.uploaded_files if f["name"] not in session.thumbnails
        ]
        if missing_thumbs:
            with status_area.status("Generating thumbnails...") as status:
                import src.application.services.thumbnail_service as worker

                def _generate_thumbnails_seq() -> None:
                    for f in missing_thumbs:
                        thumb = worker.get_thumbnail_worker(
                            f["path"], f["hash"], session.asset_store
                        )
                        if isinstance(thumb, Image.Image):
                            session.thumbnails[f["name"]] = thumb

                # Run sequentially in a thread to avoid PID leaks and memory fragmentation
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _generate_thumbnails_seq)
                status.update(label="Thumbnails ready", state="complete")

        from src.ui.state.state_manager import save_settings

        save_settings()
        pil_prev = controller.process_frame()
        st.session_state.last_pil_prev = pil_prev

        render_main_layout(pil_prev, sidebar_data, main_area)

        if sidebar_data.export_btn:
            import time

            with status_area.status("Exporting...") as status:
                start_time = time.perf_counter()
                f_hash = current_file["hash"]
                f_params = session.file_settings.get(f_hash, WorkspaceConfig())

                out_path = ExportService.run_single(
                    current_file,
                    f_params,
                    sidebar_data,
                    session.icc_profile_path,
                )

                if out_path:
                    elapsed = time.perf_counter() - start_time
                    status.update(
                        label=f"Exported to {os.path.basename(out_path)} ({elapsed:.2f}s)",
                        state="complete",
                    )
                    st.toast(
                        f"Exported to {os.path.basename(out_path)} in {elapsed:.2f}s"
                    )

        if sidebar_data.process_btn:
            await ExportService.run_batch(
                session.uploaded_files,
                session.get_settings_for_file,
                sidebar_data,
                status_area,
            )
            st.success("Batch Processing Complete")
    else:
        st.info("Upload files to start.")


if __name__ == "__main__":
    asyncio.run(main())
