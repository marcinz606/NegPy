import PyInstaller.__main__
import os
import shutil
import streamlit
import streamlit_image_coordinates

# Since this script is now in /desktop, we should change CWD to root 
# so PyInstaller can find app.py, src/, etc. easily.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the directory of the streamlit library
streamlit_dir = os.path.dirname(streamlit.__file__)
streamlit_image_coordinates_dir = os.path.dirname(streamlit_image_coordinates.__file__)

# Define the build parameters
params = [
    'desktop/backend_bootstrap.py', # Entry point
    '--name=backend',         # Output name
    '--onefile',              # Bundle into a single executable
    '--windowed',             # No console window
    '--clean',                # Clean cache
    '--additional-hooks-dir=desktop', 
    '--copy-metadata=streamlit',
    '--copy-metadata=streamlit-image-coordinates',
    '--copy-metadata=imageio',
    '--hidden-import=rawpy',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--hidden-import=PIL',
    '--hidden-import=PIL.Image',
    '--hidden-import=PIL.ImageEnhance',
    '--hidden-import=PIL.ImageFilter',
    '--hidden-import=PIL.ImageCms',
    '--hidden-import=PIL.ImageDraw',
    '--hidden-import=PIL.ImageOps',
    '--hidden-import=scipy',
    '--hidden-import=scipy.ndimage',
    '--hidden-import=scipy.stats',
    '--hidden-import=scipy.special',
    '--hidden-import=matplotlib',
    '--hidden-import=matplotlib.pyplot',
    '--hidden-import=imageio',
    '--hidden-import=imageio.v3',
    '--hidden-import=tifffile',
    '--hidden-import=streamlit_image_coordinates',
    # Include the main app logic
    '--add-data=app.py:.',
    '--add-data=src:src',
    '--add-data=icc:icc',
    # Streamlit files (crucial)
    f'--add-data={streamlit_dir}:streamlit',
    f'--add-data={streamlit_image_coordinates_dir}:streamlit_image_coordinates',
    # Config for streamlit
    '--add-data=.streamlit:.streamlit',
]

# Run PyInstaller
PyInstaller.__main__.run(params)

# Move the result to desktop/bin/backend
os.makedirs('desktop/bin/backend', exist_ok=True)
dist_name = 'backend.exe' if os.name == 'nt' else 'backend'
src_path = os.path.join('dist', dist_name)
dst_path = os.path.join('desktop/bin/backend', dist_name)

if os.path.exists(src_path):
    if os.path.exists(dst_path):
        os.remove(dst_path)
    shutil.move(src_path, dst_path)
    print(f"Successfully built and moved to {dst_path}")
else:
    print(f"Error: Could not find {src_path}")