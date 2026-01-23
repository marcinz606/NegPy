import PyInstaller.__main__
import os
import shutil
import platform
import subprocess

# Define the application name
APP_NAME = "NegPy"

# Read version
VERSION = "dev"
if os.path.exists("VERSION"):
    with open("VERSION", "r") as f:
        VERSION = f.read().strip()

# Define the entry point
ENTRY_POINT = "desktop.py"

# Define platform-specific settings
system = platform.system()
is_windows = system == "Windows"
is_macos = system == "Darwin"
is_linux = system == "Linux"

# Basic PyInstaller arguments
params = [
    ENTRY_POINT,
    f"--name={APP_NAME}",
    "--onedir",
    "--windowed",  # GUI app, no console
    "--clean",
    "--noconfirm",
    # Hidden imports (based on previous build_backend.py and new dependencies)
    "--hidden-import=rawpy",
    "--hidden-import=cv2",
    "--hidden-import=numpy",
    "--hidden-import=numba",
    "--hidden-import=PIL",
    "--hidden-import=PIL.Image",
    "--hidden-import=PIL.ImageEnhance",
    "--hidden-import=PIL.ImageFilter",
    "--hidden-import=PIL.ImageCms",
    "--hidden-import=PIL.ImageDraw",
    "--hidden-import=PIL.ImageOps",
    "--hidden-import=scipy",
    "--hidden-import=scipy.ndimage",
    "--hidden-import=scipy.stats",
    "--hidden-import=scipy.special",
    "--hidden-import=matplotlib",
    "--hidden-import=matplotlib.pyplot",
    "--hidden-import=imageio",
    "--hidden-import=imageio.v3",
    "--hidden-import=tifffile",
    "--hidden-import=imagecodecs",
    "--hidden-import=tkinter",
    "--hidden-import=_tkinter",
    "--hidden-import=PyQt6",
    "--hidden-import=qtawesome",
    # Metadata
    "--copy-metadata=imageio",
    "--copy-metadata=rawpy",
    # Collect all for complex binary packages
    "--collect-all=wgpu",
    "--collect-all=rawpy",
    "--collect-all=imageio",
    "--collect-all=imagecodecs",
    # Data files
    "--add-data=src:src",
    "--add-data=icc:icc",
    "--add-data=media:media",
    "--add-data=VERSION:.",
]

# Add platform-specific icon
if is_windows:
    icon_path = os.path.abspath("media/icons/icon.ico")
    if os.path.exists(icon_path):
        params.append(f"--icon={icon_path}")
elif is_macos:
    if os.path.exists("media/icons/icon.icns"):
        params.append("--icon=media/icons/icon.icns")
    elif os.path.exists("media/icons/icon.png"):
        params.append("--icon=media/icons/icon.png")


def package_linux():
    """Package the built application into an AppImage."""
    print("Packaging for Linux (AppImage)...")
    dist_dir = os.path.join("dist", APP_NAME)
    appdir = os.path.join("dist", f"{APP_NAME}.AppDir")

    if os.path.exists(appdir):
        shutil.rmtree(appdir)

    # 1. Create AppDir structure
    shutil.copytree(dist_dir, appdir)

    # 2. Add Desktop file and Icon
    shutil.copy("negpy.desktop", os.path.join(appdir, "negpy.desktop"))
    shutil.copy("media/icons/icon.png", os.path.join(appdir, "icon.png"))

    # 3. Create Symlink for AppRun if it doesn't exist
    apprun_path = os.path.join(appdir, "AppRun")
    if not os.path.exists(apprun_path):
        with open(apprun_path, "w") as f:
            f.write("#!/bin/sh\n")
            f.write('HERE="$(dirname "$(readlink -f "${0}")")"\n')
            f.write(f'exec "${{HERE}}/{APP_NAME}" "$@"\n')
        os.chmod(apprun_path, 0o755)

    # 4. Run appimagetool
    try:
        tool = "./appimagetool-x86_64.AppImage"
        if not os.path.exists(tool):
            tool = "appimagetool"

        output_filename = os.path.join("dist", f"{APP_NAME}-{VERSION}-x86_64.AppImage")
        subprocess.run([tool, appdir, output_filename], check=True)
        print(f"AppImage created: {output_filename}")
    except Exception as e:
        print(f"Error creating AppImage: {e}")
        raise


def package_windows():
    """Package the built application into an NSIS installer."""
    print(f"Packaging for Windows (NSIS) version {VERSION}...")

    cmd = "makensis"
    # Try to find makensis in common locations if not in PATH
    if not shutil.which(cmd):
        common_paths = [
            r"C:\Program Files (x86)\NSIS\makensis.exe",
            r"C:\Program Files\NSIS\makensis.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                cmd = p
                break

    try:
        setup_name = f"{APP_NAME}-{VERSION}-Win64-Setup.exe"
        subprocess.run(
            [cmd, f"/DVERSION={VERSION}", f"/DOUTFILE={setup_name}", "installer.nsi"],
            check=True,
        )
        print(f"Windows Installer created: dist/{setup_name}")
    except Exception as e:
        print(f"Error creating Windows Installer: {e}")
        raise


def package_macos():
    """Package the built application into a DMG with Applications symlink."""
    print(f"Packaging for macOS (DMG) version {VERSION}...")
    app_path = os.path.join("dist", f"{APP_NAME}.app")
    dmg_name = f"{APP_NAME}-{VERSION}-macOS.dmg"
    dmg_path = os.path.join("dist", dmg_name)
    temp_dmg_dir = os.path.join("dist", "dmg_temp")

    if os.path.exists(dmg_path):
        os.remove(dmg_path)
    if os.path.exists(temp_dmg_dir):
        shutil.rmtree(temp_dmg_dir)

    os.makedirs(temp_dmg_dir)

    try:
        # 1. Copy .app to temp dir (preserve symlinks for macOS bundles)
        shutil.copytree(
            app_path, os.path.join(temp_dmg_dir, f"{APP_NAME}.app"), symlinks=True
        )

        # 2. Create symlink to /Applications
        os.symlink("/Applications", os.path.join(temp_dmg_dir, "Applications"))

        # 3. Create DMG from temp dir
        subprocess.run(
            [
                "hdiutil",
                "create",
                "-volname",
                f"{APP_NAME} {VERSION}",
                "-srcfolder",
                temp_dmg_dir,
                "-ov",
                "-format",
                "UDZO",
                dmg_path,
            ],
            check=True,
        )
        print(f"macOS DMG created: {dmg_path}")
    except Exception as e:
        print(f"Error creating macOS DMG: {e}")
        raise
    finally:
        if os.path.exists(temp_dmg_dir):
            shutil.rmtree(temp_dmg_dir)


def build():
    print(f"Building {APP_NAME} for {system}...")
    print("PyInstaller parameters:", params)

    PyInstaller.__main__.run(params)

    print("Build complete.")
    if os.path.exists("dist"):
        print(f"Contents of dist: {os.listdir('dist')}")
        if os.path.exists(f"dist/{APP_NAME}"):
            print(
                f"Contents of dist/{APP_NAME}: {os.listdir(f'dist/{APP_NAME}')[:10]}... (truncated)"
            )
    else:
        print("ERROR: dist directory not found!")

    if is_linux:
        package_linux()
    elif is_windows:
        package_windows()
    elif is_macos:
        package_macos()


if __name__ == "__main__":
    build()
