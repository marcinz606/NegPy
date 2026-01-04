import os
import platform
import subprocess
import sys
from pathlib import Path

def get_documents_dir():
    """
    Attempt to locate the user's Documents directory in a cross-platform way.
    """
    system = platform.system()
    home = Path.home()
    
    if system == "Windows":
        # Windows standard path
        docs = home / "Documents"
    elif system == "Darwin": # macOS
        docs = home / "Documents"
    else:
        # Linux and others
        # 1. Try xdg-user-dir command (standard way to get localized paths)
        try:
            result = subprocess.run(["xdg-user-dir", "DOCUMENTS"], capture_output=True, text=True, check=True)
            path = Path(result.stdout.strip())
            if path.exists() and path != home:
                return path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # 2. Try XDG standard environment variable
        xdg_docs = os.getenv("XDG_DOCUMENTS_DIR")
        if xdg_docs:
            return Path(xdg_docs)
        
        # 3. Fallback to standard ~/Documents
        docs = home / "Documents"
        
    # Verify it exists, else fallback to home
    if docs.exists():
        return docs
    return home

def main():
    documents_dir = get_documents_dir()
    app_data_dir = documents_dir / "DarkroomPy"
    
    print(f"[{platform.system()}] Located Documents dir: {documents_dir}")
    print(f"Setting up application data at: {app_data_dir}")
    
    # Ensure directory exists
    try:
        app_data_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {app_data_dir}: {e}")
        sys.exit(1)

    # Prepare environment variables
    env = os.environ.copy()
    env["DARKROOM_HOST_DIR"] = str(app_data_dir.absolute())
    
    # Check if docker-compose or docker compose command is available
    cmd = ["docker", "compose", "up"]
    
    # If user passed arguments (e.g. --build, -d), pass them along
    cmd.extend(sys.argv[1:])
    
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
