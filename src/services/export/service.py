import os
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Callable
import streamlit as st
from src.domain.models import WorkspaceConfig, ExportConfig
from src.services.export.templating import FilenameTemplater
from src.kernel.system.logging import get_logger
from src.kernel.system.config import APP_CONFIG
from src.services.rendering.image_processor import ImageProcessor

logger = get_logger(__name__)


def _process_and_save_worker(
    file_path: str,
    file_meta: Dict[str, str],
    f_params: WorkspaceConfig,
    export_settings: ExportConfig,
) -> str:
    """
    Worker function for ProcessPoolExecutor.
    Initializes its own processor and templater to ensure thread/process safety.
    """
    # Lazy init inside worker to avoid issues with unpicklable objects or shared state
    image_service = ImageProcessor()
    templater_instance = FilenameTemplater()

    res = image_service.process_export(
        file_path, f_params, export_settings, source_hash=file_meta["hash"]
    )

    img_bytes, ext = res
    if img_bytes is None:
        raise RuntimeError(f"Render failed: {ext}")

    context = {
        "original_name": file_meta["name"].rsplit(".", 1)[0],
        "mode": f_params.process_mode,
        "colorspace": export_settings.export_color_space,
        "border": "border" if (export_settings.export_border_size > 0.0) else "",
    }

    base_name = templater_instance.render(export_settings.filename_pattern, context)
    out_path = os.path.join(export_settings.export_path, f"{base_name}.{ext}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as out_f:
        out_f.write(img_bytes)

    return out_path


class ExportService:
    """
    Application service for managing single and batch exports.
    """

    @staticmethod
    def run_single(
        file_meta: Dict[str, str],
        f_params: WorkspaceConfig,
        sidebar_data: Any,
        icc_profile_path: Any,
    ) -> str:
        """
        Executes a single image export based on sidebar state.
        """
        export_settings = ExportConfig(
            export_fmt=sidebar_data.out_fmt,
            export_color_space=sidebar_data.color_space,
            export_print_size=sidebar_data.print_width,
            export_dpi=sidebar_data.print_dpi,
            export_add_border=sidebar_data.add_border,
            export_border_size=sidebar_data.border_size,
            export_border_color=sidebar_data.border_color,
            icc_profile_path=icc_profile_path if sidebar_data.apply_icc else None,
            export_path=sidebar_data.export_path,
            filename_pattern=sidebar_data.filename_pattern,
        )

        return _process_and_save_worker(
            file_meta["path"], file_meta, f_params, export_settings
        )

    @staticmethod
    async def run_batch(
        files: List[Dict[str, str]],
        get_settings_cb: Callable[[str], WorkspaceConfig],
        sidebar_data: Any,
        status_area: Any,
    ) -> None:
        """
        Executes a parallelized batch export using ProcessPoolExecutor.
        This provides better isolation for heavy C-libraries (rawpy, opencv)
        and avoids threading crashes specifically on macOS/Apple Silicon.
        """
        os.makedirs(sidebar_data.export_path, exist_ok=True)
        total_files = len(files)
        start_time = time.perf_counter()

        # Concurrency limit for batch exports
        # to avoid threadig issues when combined with numba and opencv
        # internal threading
        limit = max(1, APP_CONFIG.max_workers // 4)

        icc_path = (
            st.session_state.session.icc_profile_path
            if sidebar_data.apply_icc
            else None
        )

        tasks_args = []
        for f_meta in files:
            f_settings = get_settings_cb(f_meta["hash"])
            f_export_settings = ExportConfig(
                export_fmt=sidebar_data.out_fmt,
                export_color_space=sidebar_data.color_space,
                export_print_size=sidebar_data.print_width,
                export_dpi=sidebar_data.print_dpi,
                export_add_border=sidebar_data.add_border,
                export_border_size=sidebar_data.border_size,
                export_border_color=sidebar_data.border_color,
                icc_profile_path=icc_path,
                export_path=sidebar_data.export_path,
                filename_pattern=sidebar_data.filename_pattern,
            )
            tasks_args.append((f_meta["path"], f_meta, f_settings, f_export_settings))

        with status_area.status(
            f"Printing {total_files} images...", expanded=True
        ) as status:
            logger.info(f"Starting batch print with {limit} workers...")

            loop = asyncio.get_running_loop()
            results = []

            with concurrent.futures.ProcessPoolExecutor(max_workers=limit) as executor:
                futures = [
                    loop.run_in_executor(executor, _process_and_save_worker, *args)
                    for args in tasks_args
                ]
                results = await asyncio.gather(*futures, return_exceptions=True)

            for f_meta, res in zip(files, results):
                if isinstance(res, Exception):
                    logger.error(f"Error printing {f_meta['name']}: {res}")
                    st.error(f"Error printing {f_meta['name']}: {res}")
                elif not res:
                    st.warning(f"Failed to print {f_meta['name']}")

            elapsed = time.perf_counter() - start_time
            status.update(
                label=f"Batch Printing Complete in {elapsed:.2f}s", state="complete"
            )
