import tkinter as tk
from tkinter import filedialog
import os
import json
import sys
from typing import Optional


def pick_files(initial_dir: Optional[str] = None) -> None:
    """Opens a multi-file selection dialog and prints JSON to stdout."""
    root = tk.Tk()
    root.withdraw()

    # macOS specific fix to bring dialog to front
    if sys.platform == "darwin":
        os.system(
            """/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' """
        )

    root.attributes("-topmost", True)

    # Use initial_dir if provided and exists
    start_dir = initial_dir if initial_dir and os.path.exists(initial_dir) else None

    file_paths = filedialog.askopenfilenames(
        title="Select RAW Files",
        initialdir=start_dir,
        filetypes=[
            ("RAW files", "*.dng *.tiff *.tif *.nef *.arw *.raw *.raf"),
            ("All files", "*.*"),
        ],
    )
    output = json.dumps([os.path.abspath(p) for p in file_paths])
    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    root.destroy()


def pick_folder(initial_dir: Optional[str] = None) -> None:
    """Opens a folder selection dialog and prints JSON result to stdout."""
    root = tk.Tk()
    root.withdraw()

    # macOS specific fix to bring dialog to front
    if sys.platform == "darwin":
        os.system(
            """/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' """
        )

    root.attributes("-topmost", True)

    start_dir = initial_dir if initial_dir and os.path.exists(initial_dir) else None

    folder_path = filedialog.askdirectory(
        title="Select Folder containing RAWs", initialdir=start_dir
    )
    if not folder_path:
        output = json.dumps(["", []])
    else:
        supported_exts = {".dng", ".tiff", ".tif", ".nef", ".arw", ".raw", ".raf"}
        found_files = []
        for r, _, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in supported_exts:
                    found_files.append(os.path.abspath(os.path.join(r, file)))
        output = json.dumps([os.path.abspath(folder_path), found_files])

    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    root.destroy()


def pick_export_folder(initial_dir: Optional[str] = None) -> None:
    """Opens a folder selection dialog and prints the path as JSON string to stdout."""
    root = tk.Tk()
    root.withdraw()

    # macOS specific fix to bring dialog to front
    if sys.platform == "darwin":
        os.system(
            """/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' """
        )

    root.attributes("-topmost", True)

    start_dir = initial_dir if initial_dir and os.path.exists(initial_dir) else None

    folder_path = filedialog.askdirectory(
        title="Select Export Folder", initialdir=start_dir
    )

    output = json.dumps(os.path.abspath(folder_path) if folder_path else "")

    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    root.destroy()
