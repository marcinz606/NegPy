import streamlit as st
import os
import asyncio
import concurrent.futures
from PIL import Image
from typing import Any
from src.config import APP_CONFIG
from src.core.session.models import WorkspaceConfig, ExportConfig
from src.orchestration.render_service import load_raw_and_process
from src.presentation.state.state_manager import init_session_state
from src.presentation.styles.theme import apply_custom_css
from src.presentation.components.sidebar.files_ui import render_file_manager
from src.presentation.components.sidebar.main import render_sidebar_content
from src.presentation.layouts.main_layout import (
    render_layout_header,
    render_main_layout,
)
from src.presentation.state.session_context import SessionContext
from src.presentation.controllers.app_controller import AppController
from src.presentation.services.export_service import ExportService


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
            from src.presentation.state.state_manager import load_settings

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
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=APP_CONFIG.max_workers
                ) as executor:
                    loop = asyncio.get_running_loop()
                    # Use absolute import path for the worker to avoid pickling issues
                    import src.orchestration.thumbnail_worker as worker

                    tasks = [
                        loop.run_in_executor(
                            executor, worker.get_thumbnail_worker, f["path"], f["hash"]
                        )
                        for f in missing_thumbs
                    ]
                    results = await asyncio.gather(*tasks)
                    for f_meta, thumb in zip(missing_thumbs, results):
                        if isinstance(thumb, Image.Image):
                            session.thumbnails[f_meta["name"]] = thumb
                status.update(label="Thumbnails ready", state="complete")

        from src.presentation.state.state_manager import save_settings

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

                export_settings = ExportConfig(
                    export_fmt=sidebar_data.out_fmt,
                    export_color_space=sidebar_data.color_space,
                    export_print_size=sidebar_data.print_width,
                    export_dpi=sidebar_data.print_dpi,
                    export_add_border=sidebar_data.add_border,
                    export_border_size=sidebar_data.border_size,
                    export_border_color=sidebar_data.border_color,
                    icc_profile_path=session.icc_profile_path
                    if sidebar_data.apply_icc
                    else None,
                    export_path=sidebar_data.export_path,
                )

                img_bytes, ext = load_raw_and_process(
                    current_file["path"], f_params, export_settings
                )
                if img_bytes is not None:
                    os.makedirs(sidebar_data.export_path, exist_ok=True)
                    out_path = os.path.join(
                        sidebar_data.export_path,
                        f"processed_{current_file['name'].rsplit('.', 1)[0]}.{ext}",
                    )
                    with open(out_path, "wb") as out_f:
                        out_f.write(img_bytes)

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
                session.uploaded_files, session.file_settings, sidebar_data, status_area
            )
            st.success("Batch Processing Complete")
    else:
        st.info("Upload files to start.")


if __name__ == "__main__":
    asyncio.run(main())
