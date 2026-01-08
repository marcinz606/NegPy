import os

# Fix for Numba TBB error: "Attempted to fork from a non-main thread"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import asyncio
import logging
import multiprocessing
import sys
from src.logging_config import setup_logging


def handle_subtask() -> bool:
    """Checks if the app was called to perform a CLI subtask (like picking files)."""
    if len(sys.argv) > 1:
        initial_dir = sys.argv[2] if len(sys.argv) > 2 else None
        if sys.argv[1] == "--pick-files":
            from src.infrastructure.loaders.dialog_worker import pick_files

            pick_files(initial_dir)
            return True
        elif sys.argv[1] == "--pick-folder":
            from src.infrastructure.loaders.dialog_worker import pick_folder

            pick_folder(initial_dir)
            return True
        elif sys.argv[1] == "--pick-export-folder":
            from src.infrastructure.loaders.dialog_worker import pick_export_folder

            pick_export_folder(initial_dir)
            return True
    return False


async def start_app() -> None:
    from src.presentation.app import main

    await main()


if __name__ == "__main__":
    # If this is a subtask (e.g. file picker), run it and exit
    if handle_subtask():
        sys.exit(0)

    multiprocessing.freeze_support()
    setup_logging(level=logging.INFO)

    from src.core.performance import clear_perf_log

    clear_perf_log()
    asyncio.run(start_app())
