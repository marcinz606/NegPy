import os
import platform
import subprocess
import sys
import argparse
from pathlib import Path


def get_documents_dir() -> Path:
    """
    Find the Documents directory for the current OS.
    """
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        # Windows standard path
        docs = home / "Documents"
    elif system == "Darwin":  # macOS
        docs = home / "Documents"
    else:
        # Linux / XDG
        try:
            result = subprocess.run(
                ["xdg-user-dir", "DOCUMENTS"],
                capture_output=True,
                text=True,
                check=True,
            )
            path = Path(result.stdout.strip())
            if path.exists() and path != home:
                return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Environment variable
        xdg_docs = os.getenv("XDG_DOCUMENTS_DIR")
        if xdg_docs:
            return Path(xdg_docs)

        # Home fallback
        docs = home / "Documents"

    # Verify it exists, else fallback to home
    if docs.exists():
        return docs
    return home


def main() -> None:
    parser = argparse.ArgumentParser(description="Start NegPy via Docker Compose")
    parser.add_argument(
        "--build", action="store_true", help="Rebuild the container before starting"
    )
    args, unknown = parser.parse_known_args()

    documents_dir = get_documents_dir()
    app_data_dir = documents_dir / "NegPy"

    print(f"[{platform.system()}] Located Documents dir: {documents_dir}")
    print(f"Setting up application data at: {app_data_dir}")

    try:
        app_data_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {app_data_dir}: {e}")
        sys.exit(1)

    env = os.environ.copy()
    env["NEGPY_HOST_DIR"] = str(app_data_dir.absolute())

    cmd = ["docker", "compose", "up"]

    if args.build:
        cmd.append("--build")

    # pass unknown args along
    cmd.extend(unknown)

    print(f"Starting Docker Compose with host volume: {app_data_dir} -> /app/user")
    print("Run command:", " ".join(cmd))

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running docker compose: {e}")
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
