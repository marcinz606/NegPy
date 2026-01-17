import os

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import sys
from pathlib import Path
from src.kernel.system.logging import init_streams
import streamlit.web.cli as stcli


init_streams()


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        bundle_dir = sys._MEIPASS
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))

    user_dir = os.environ.get("NEGPY_USER_DIR")
    if not user_dir:
        # Standard fallback for development
        docs = Path.home() / "Documents"
        user_dir = str(docs / "NegPy")
        os.environ["NEGPY_USER_DIR"] = user_dir

    if not os.path.exists(user_dir):
        os.makedirs(user_dir, exist_ok=True)

    sys.stderr.write("Engine starting...\n")
    sys.stderr.write(f"User Directory: {user_dir}\n")
    sys.stderr.write(f"Is Bundled: {getattr(sys, 'frozen', False)}\n")

    # Hack to make native file picker work with streamlit
    os.chdir(bundle_dir)
    if "--pick-files" in sys.argv:
        from src.infrastructure.loaders.dialog_worker import pick_files

        idx = sys.argv.index("--pick-files")
        initial_dir = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        pick_files(initial_dir)
        sys.exit(0)
    elif "--pick-folder" in sys.argv:
        from src.infrastructure.loaders.dialog_worker import pick_folder

        idx = sys.argv.index("--pick-folder")
        initial_dir = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        pick_folder(initial_dir)
        sys.exit(0)
    elif "--pick-export-folder" in sys.argv:
        from src.infrastructure.loaders.dialog_worker import pick_export_folder

        idx = sys.argv.index("--pick-export-folder")
        initial_dir = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
        pick_export_folder(initial_dir)
        sys.exit(0)

    if getattr(sys, "frozen", False):
        app_path = os.path.join(bundle_dir, "app.py")
    else:
        app_path = os.path.join(os.path.dirname(bundle_dir), "app.py")

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--global.developmentMode=false",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    sys.exit(stcli.main())
