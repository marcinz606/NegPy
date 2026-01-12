import tkinter as tk
from tkinter import filedialog
import os
import json
import sys
from typing import Optional


def _bring_to_front() -> None:
    """macOS fix: bring dialog to front."""
    if sys.platform == "darwin":
        proc_name = os.path.basename(sys.executable)
        os.system(
            f"""/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "{proc_name}" to true' """
        )


def pick_files(initial_dir: Optional[str] = None) -> None:
    """Multi-file dialog (tk)."""
    root = tk.Tk()
    root.withdraw()

    _bring_to_front()

    root.attributes("-topmost", True)

    start_dir = initial_dir if initial_dir and os.path.exists(initial_dir) else None

    file_paths = filedialog.askopenfilenames(
        title="Select RAW Files",
        initialdir=start_dir,
        filetypes=[
            (
                "RAW files",
                "*.dng *.DNG *.tiff *.TIFF *.tif *.TIF *.nef *.NEF "
                "*.arw *.ARW *.raw *.RAW *.raf *.RAF",
            ),
            ("All files", "*.*"),
        ],
    )
    output = json.dumps([os.path.abspath(p) for p in file_paths])
    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    root.destroy()


def pick_folder(initial_dir: Optional[str] = None) -> None:
    """Folder dialog (tk)."""
    root = tk.Tk()
    root.withdraw()

    _bring_to_front()

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
    """Export folder dialog (tk)."""
    root = tk.Tk()
    root.withdraw()

    _bring_to_front()

    root.attributes("-topmost", True)

    start_dir = initial_dir if initial_dir and os.path.exists(initial_dir) else None

    folder_path = filedialog.askdirectory(
        title="Select Export Folder", initialdir=start_dir
    )

    output = json.dumps(os.path.abspath(folder_path) if folder_path else "")

    sys.stdout.write(output + "\n")
    sys.stdout.flush()
    root.destroy()
