import os
import asyncio
from typing import List, Dict, Any
import streamlit as st
from src.core.session.models import WorkspaceConfig, ExportConfig
from src.core.io.templating import FilenameTemplater

templater = FilenameTemplater()


def _process_and_save(
    file_path: str,
    file_meta: Dict[str, str],
    f_params: WorkspaceConfig,
    export_settings: ExportConfig,
) -> str:
    """
    Handles rendering, templating, and saving.
    """
    # avoid circular imports
    import src.orchestration.render_service as renderer
    from src.core.io.templating import FilenameTemplater

    worker_templater = FilenameTemplater()

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

    base_name = worker_templater.render(export_settings.filename_pattern, context)
    out_path = os.path.join(export_settings.export_path, f"{base_name}.{ext}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as out_f:
        out_f.write(img_bytes)

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
            file_meta["path"], file_meta, f_params, export_settings
        )

    @staticmethod
    async def run_batch(
        files: List[Dict[str, str]],
        settings_map: Dict[str, Any],
        sidebar_data: Any,
        status_area: Any,
    ) -> None:
        """
        Executes a sequential batch export.
        Leverages full multi-threading capability of the engine for each image.
        """
        import time

        os.makedirs(sidebar_data.export_path, exist_ok=True)
        total_files = len(files)
        start_time = time.perf_counter()

        loop = asyncio.get_running_loop()

        with status_area.status(
            f"Processing {total_files} images...", expanded=True
        ) as status:
            # Prepare all tasks data upfront to minimize MainThread interaction inside the loop
            tasks_data = []
            for f_meta in files:
                f_hash = f_meta["hash"]
                f_params = settings_map.get(f_hash, WorkspaceConfig())

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
                    filename_pattern=sidebar_data.filename_pattern,
                )
                tasks_data.append((f_meta, f_params, f_export_settings))

            def _sequence_runner() -> List[Any]:
                results: List[Any] = []
                for i, (meta, params, settings) in enumerate(tasks_data):
                    try:
                        path = _process_and_save(meta["path"], meta, params, settings)
                        results.append(path)
                    except Exception as e:
                        results.append(e)
                return results

            # Run the heavy loop in a separate thread            # This allows the UI to remain responsive while the CPU churns through images
            results = await loop.run_in_executor(None, _sequence_runner)

            for f_meta, res in zip(files, results):
                if isinstance(res, Exception):
                    st.error(f"Error processing {f_meta['name']}: {res}")
                elif not res:
                    st.warning(f"Failed to export {f_meta['name']}")

            elapsed = time.perf_counter() - start_time
            status.update(
                label=f"Batch Processing Complete in {elapsed:.2f}s", state="complete"
            )
