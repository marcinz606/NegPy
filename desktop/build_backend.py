import PyInstaller.__main__
import os
import shutil
import platform
import streamlit
import streamlit_image_coordinates

# This script is used to build python app using pyinstaller
# It is not used in the app itself

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
streamlit_dir = os.path.dirname(streamlit.__file__)
streamlit_image_coordinates_dir = os.path.dirname(streamlit_image_coordinates.__file__)

# build params
params = [
    "desktop/backend_bootstrap.py",  # electron point
    "--name=darkroompy",
    "--onefile",
    "--clean",
    "--noconfirm",
    "--additional-hooks-dir=desktop",
    "--copy-metadata=streamlit",
    "--copy-metadata=streamlit-image-coordinates",
    "--copy-metadata=imageio",
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
]

if platform.system() == "Windows":
    params.append("--windowed")

PyInstaller.__main__.run(params)
os.makedirs("desktop/bin/darkroompy", exist_ok=True)

if platform.system() == "Windows":
    dist_name = "darkroompy.exe"
else:
    dist_name = "darkroompy"

src_path = os.path.join("dist", dist_name)
dst_path = os.path.join("desktop/bin/darkroompy", dist_name)

if os.path.exists(src_path):
    if os.path.exists(dst_path):
        if os.path.isdir(dst_path):
            shutil.rmtree(dst_path)
        else:
            os.remove(dst_path)

    shutil.move(src_path, dst_path)
    print(f"Successfully built and moved to {dst_path}")

    # Cleanup empty dist folder created by PyInstaller
    if os.path.exists("dist") and not os.listdir("dist"):
        os.rmdir("dist")
else:
    print(f"Error: Could not find {src_path}")
