import os
import time
import gc
import asyncio
from typing import List, Dict, Any, Callable
import streamlit as st
from src.domain.models import WorkspaceConfig, ExportConfig
from src.services.export.templating import FilenameTemplater
from src.kernel.system.logging import get_logger
from src.kernel.system.config import APP_CONFIG
from src.services.rendering.image_processor import ImageProcessor

templater = FilenameTemplater()
logger = get_logger(__name__)
image_service = ImageProcessor()


def _process_and_save(
    file_path: str,
    file_meta: Dict[str, str],
    f_params: WorkspaceConfig,
    export_settings: ExportConfig,
    templater_instance: FilenameTemplater,
) -> str:
    """
    Orchestrates the rendering, filename templating, and disk write for a single export.
    """
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

    logger.info(f"Exported {file_meta['name']} to {os.path.basename(out_path)}")
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

        return _process_and_save(
            file_meta["path"], file_meta, f_params, export_settings, templater
        )

    @staticmethod
    async def run_batch(
        files: List[Dict[str, str]],
        get_settings_cb: Callable[[str], WorkspaceConfig],
        sidebar_data: Any,
        status_area: Any,
    ) -> None:
        """
        Executes a parallelized batch export using a semaphore to limit concurrency.
        """
        os.makedirs(sidebar_data.export_path, exist_ok=True)
        total_files = len(files)
        start_time = time.perf_counter()

        # Limit concurrency to 1/3 of available cores to avoid oversubscription
        # and 'fork bombs' as numba compiled functions are already multi-threaded
        limit = APP_CONFIG.max_workers // 3
        semaphore = asyncio.Semaphore(limit)

        async def _worker(f_meta: Dict[str, str]) -> Any:
            async with semaphore:
                f_hash = f_meta["hash"]
                f_settings = get_settings_cb(f_hash)

                icc_path = (
                    st.session_state.session.icc_profile_path
                    if sidebar_data.apply_icc
                    else None
                )

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

                try:
                    result = await asyncio.to_thread(
                        _process_and_save,
                        f_meta["path"],
                        f_meta,
                        f_settings,
                        f_export_settings,
                        templater,
                    )
                    return result
                except Exception as e:
                    return e
                finally:
                    gc.collect()

        with status_area.status(
            f"Processing {total_files} images...", expanded=True
        ) as status:
            logger.info(f"Processing {total_files} images with concurrency {limit}...")

            tasks = [_worker(f) for f in files]
            results = await asyncio.gather(*tasks)

            for f_meta, res in zip(files, results):
                if isinstance(res, Exception):
                    logger.error(f"Error processing {f_meta['name']}: {res}")
                    st.error(f"Error processing {f_meta['name']}: {res}")
                elif not res:
                    st.warning(f"Failed to export {f_meta['name']}")

            elapsed = time.perf_counter() - start_time
            status.update(
                label=f"Batch Processing Complete in {elapsed:.2f}s", state="complete"
            )
