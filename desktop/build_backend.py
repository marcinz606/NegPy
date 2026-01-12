# PyInstaller build script for the Python backend.
# Not part of the runtime app.

import PyInstaller.__main__
import os
import shutil
import platform
import streamlit
import streamlit_image_coordinates


os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
streamlit_dir = os.path.dirname(streamlit.__file__)
streamlit_image_coordinates_dir = os.path.dirname(streamlit_image_coordinates.__file__)

# build params
params = [
    "desktop/backend_bootstrap.py",  # electron entry point
    "--name=negpy",
    "--onedir",
    "--clean",
    "--noconfirm",
    "--additional-hooks-dir=desktop",
    "--copy-metadata=streamlit",
    "--copy-metadata=streamlit-image-coordinates",
    "--copy-metadata=imageio",
    "--collect-all=imagecodecs",
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
    "--hidden-import=streamlit_image_coordinates",
    "--hidden-import=jinja2",
    "--hidden-import=tkinter",
    "--hidden-import=_tkinter",
    # Include the main app logic
    "--add-data=app.py:.",
    "--add-data=src:src",
    "--add-data=icc:icc",
    "--add-data=media:media",
    # Streamlit files
    f"--add-data={streamlit_dir}:streamlit",
    f"--add-data={streamlit_image_coordinates_dir}:streamlit_image_coordinates",
    # Config for streamlit
    "--add-data=.streamlit:.streamlit",
    "--add-data=VERSION:.",
]

if platform.system() == "Windows":
    params.append("--windowed")

PyInstaller.__main__.run(params)

dest_dir = os.path.join("desktop", "bin", "negpy")
src_folder_name = "negpy"
src_path = os.path.join("dist", src_folder_name)

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

if os.path.exists(src_path):
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
    shutil.move(src_path, dest_dir)
    print(f"Successfully built and moved to {dest_dir}")

    if os.path.exists("dist") and not os.listdir("dist"):
        os.rmdir("dist")
else:
    print(f"Error: Could not find {src_path}")
