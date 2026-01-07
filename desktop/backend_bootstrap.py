import os
import sys
import streamlit.web.cli as stcli


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path


if __name__ == "__main__":
    # Check if running in a bundled PyInstaller environment
    if getattr(sys, "frozen", False):
        # We are running in a bundle
        bundle_dir = sys._MEIPASS
    else:
        # We are running in a normal Python environment
        bundle_dir = os.path.dirname(os.path.abspath(__file__))

    # Handle DARKROOM_USER_DIR
    # Docker usually sets this to /app/user
    # Electron sets this to Documents/DarkroomPy
    # If not set, we default to ./user
    user_dir = os.environ.get("DARKROOM_USER_DIR")
    if not user_dir:
        if os.path.exists("/.dockerenv"):
            user_dir = "/app/user"
        else:
            user_dir = os.path.join(os.getcwd(), "user")
        os.environ["DARKROOM_USER_DIR"] = user_dir

    if not os.path.exists(user_dir):
        os.makedirs(user_dir, exist_ok=True)

    sys.stderr.write("Engine starting...\n")
    sys.stderr.write(f"User Directory: {user_dir}\n")
    sys.stderr.write(f"Is Bundled: {getattr(sys, 'frozen', False)}\n")

    # Change CWD to bundle_dir so relative paths in the app (like 'icc/')
    # resolve correctly to the bundled files.
    os.chdir(bundle_dir)

    # Check for subtasks (native dialogs) before starting Streamlit
    if "--pick-files" in sys.argv:
        from src.infrastructure.loaders.dialog_worker import pick_files

        # The initial_dir is passed as the next argument after the flag
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

    # Streamlit execution
    # If bundled, app.py is at the root of bundle_dir.
    # If dev, app.py is in the parent directory of this script.
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
