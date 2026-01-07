import os
import asyncio
import concurrent.futures
from typing import List, Dict, Any
import streamlit as st
from src.config import APP_CONFIG
from src.core.session.models import WorkspaceConfig, ExportConfig


class ExportService:
    """
    Service responsible for single and batch file exports.
    """

    @staticmethod
    async def run_batch(
        files: List[Dict[str, str]],
        settings_map: Dict[str, Any],
        sidebar_data: Any,
        status_area: Any,
    ) -> None:
        """
        Executes a multi-threaded batch export.
        """
        import time

        os.makedirs(sidebar_data.export_path, exist_ok=True)
        total_files = len(files)
        start_time = time.perf_counter()

        with status_area.status(
            f"Processing {total_files} images...", expanded=True
        ) as status:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=APP_CONFIG.max_workers
            ) as executor:
                loop = asyncio.get_running_loop()
                batch_tasks = []
                file_names = []

                import src.orchestration.render_service as renderer

                for f_meta in files:
                    f_hash = f_meta["hash"]
                    f_settings = settings_map.get(f_hash, WorkspaceConfig())
                    f_params = f_settings
                    file_names.append(f_meta["name"])

                    f_export_settings = ExportConfig(
                        export_fmt=sidebar_data.out_fmt,
                        export_color_space=sidebar_data.color_space,
                        export_print_size=sidebar_data.print_width,
                        export_dpi=sidebar_data.print_dpi,
                        export_add_border=sidebar_data.add_border,
                        export_border_size=sidebar_data.border_size,
                        export_border_color=sidebar_data.border_color,
                        icc_profile_path=st.session_state.session.icc_profile_path
                        if sidebar_data.apply_icc
                        else None,
                        export_path=sidebar_data.export_path,
                    )

                    task = loop.run_in_executor(
                        executor,
                        renderer.load_raw_and_process,
                        f_meta["path"],
                        f_params,
                        f_export_settings,
                    )
                    batch_tasks.append(task)

                results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for fname, res in zip(file_names, results):
                    if isinstance(res, tuple) and res[0] is not None:
                        img_bytes, ext = res
                        out_path = os.path.join(
                            sidebar_data.export_path,
                            f"processed_{fname.rsplit('.', 1)[0]}.{ext}",
                        )
                        with open(out_path, "wb") as out_f:
                            if img_bytes is not None:
                                out_f.write(img_bytes)
                    elif isinstance(res, Exception):
                        st.error(f"Error processing {fname}: {res}")

            elapsed = time.perf_counter() - start_time
            status.update(
                label=f"Batch Processing Complete in {elapsed:.2f}s", state="complete"
            )
