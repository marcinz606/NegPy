import os
import time
import gc
from typing import List, Dict, Any, Callable
import streamlit as st
from src.core.models import WorkspaceConfig, ExportConfig
from src.core.templating import FilenameTemplater
from src.logging_config import get_logger
import src.application.services.render_service as renderer

templater = FilenameTemplater()
logger = get_logger(__name__)


def _process_and_save(
    file_path: str,
    file_meta: Dict[str, str],
    f_params: WorkspaceConfig,
    export_settings: ExportConfig,
    templater_instance: FilenameTemplater,
) -> str:
    """
    Handles rendering, templating, and saving.
    """
    res = renderer.load_raw_and_process(file_path, f_params, export_settings)

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

    # Log completion
    logger.info(f"Exported {file_meta['name']} to {os.path.basename(out_path)}")

    return out_path


class ExportService:
    """
    Service responsible for single and batch file exports.
    """

    @staticmethod
    def run_single(
        file_meta: Dict[str, str],
        f_params: WorkspaceConfig,
        sidebar_data: Any,
        icc_profile_path: Any,
    ) -> str:
        """
        Executes a single file export.
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
    def run_batch(
        files: List[Dict[str, str]],
        get_settings_cb: Callable[[str], WorkspaceConfig],
        sidebar_data: Any,
        status_area: Any,
    ) -> None:
        """
        Executes a synchronous sequential batch export. Single export is already
        multithreaded via numba, adding parallel processing on top of that just
        creates issues (fork bombs + ooms) without significant speedup
        so we resort to just simple for loop
        """
        os.makedirs(sidebar_data.export_path, exist_ok=True)
        total_files = len(files)
        start_time = time.perf_counter()

        with status_area.status(
            f"Processing {total_files} images...", expanded=True
        ) as status:
            logger.info(f"Processing {total_files} images...")
            for f_meta in files:
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
                    _process_and_save(
                        f_meta["path"], f_meta, f_settings, f_export_settings, templater
                    )
                except Exception as e:
                    logger.error(f"Error processing {f_meta['name']}: {e}")
                    st.error(f"Error processing {f_meta['name']}: {e}")

                # Stability measures
                gc.collect()
                time.sleep(0.05)

            elapsed = time.perf_counter() - start_time
            status.update(
                label=f"Batch Processing Complete in {elapsed:.2f}s", state="complete"
            )
