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

    print("Engine starting...")
    print(f"User Directory: {user_dir}")
    print(f"Is Bundled: {getattr(sys, 'frozen', False)}")

    # Change CWD to bundle_dir so relative paths in the app (like 'icc/')
    # resolve correctly to the bundled files.
    os.chdir(bundle_dir)

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
