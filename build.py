import functools
import glob
import os
import platform
import shutil
import subprocess

import PyInstaller.__main__

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


def get_macos_target_arch():
    """Return the macOS target architecture for packaging."""
    return os.environ.get("NEGPY_MACOS_ARCH", platform.machine())


# Basic PyInstaller arguments
params = [
    ENTRY_POINT,
    f"--name={APP_NAME}",
    "--onedir",
    "--windowed",  # GUI app, no console
    "--clean",
    "--noconfirm",
    # Hidden imports
    "--hidden-import=rawpy",
    "--hidden-import=cv2",
    "--hidden-import=numpy",
    "--hidden-import=numba",
    "--hidden-import=PIL",
    "--hidden-import=PIL.Image",
    "--hidden-import=PIL.ImageCms",
    "--hidden-import=imageio",
    "--hidden-import=imageio.v3",
    "--hidden-import=tifffile",
    "--hidden-import=imagecodecs",
    "--hidden-import=jinja2",
    "--hidden-import=PyQt6",
    "--hidden-import=qtawesome",
    # Scanner support: bundle the python-sane C extension but NOT libsane.so.1.
    # libsane.so.1 must come from the host so SANE can find its backend plugins
    # in /usr/lib/sane/. See libs_to_remove in package_linux().
    # Requires: uv sync --group sane before building on Linux/macOS.
    *([] if is_windows else ["--hidden-import=sane", "--hidden-import=_sane"]),
    # Camera scanning: see collect_gphoto2_plugins() — the plugin trees need their
    # directory layout preserved, which --collect-all does not do.
    *([] if is_windows else ["--collect-all=gphoto2"]),
    # pieusb scanner support (all platforms).
    # libusb_package's own PyInstaller hook drops the bundled libusb at the bundle
    # root, but get_library_path() resolves it with importlib_resources against the
    # *package* directory — so without this it finds nothing and pyusb falls back to
    # a system libusb that Windows does not have. --collect-all puts it where the
    # lookup actually looks.
    "--hidden-import=pieusb",
    "--collect-all=libusb_package",
    # Plustek / pyopticfilm: ship PyUSB + bundled libusb on Windows only.
    # Linux/macOS use host libusb via PyUSB (same stack SANE needs).
    *(
        [
            "--hidden-import=usb",
            "--hidden-import=usb.core",
            "--hidden-import=usb.backend.libusb1",
            "--hidden-import=pyopticfilm",
            "--collect-all=usb",
            "--collect-all=pyopticfilm",
        ]
        if is_windows
        else []
    ),
    # Exclude unused modules
    # Metadata
    "--copy-metadata=imageio",
    "--copy-metadata=rawpy",
    "--collect-all=wgpu",
    "--collect-all=rawpy",
    "--collect-all=imageio",
    "--collect-all=imagecodecs",
    # Data files
    "--add-data=negpy/features/exposure/shaders:negpy/features/exposure/shaders",
    "--add-data=negpy/features/geometry/shaders:negpy/features/geometry/shaders",
    "--add-data=negpy/features/toning/shaders:negpy/features/toning/shaders",
    "--add-data=negpy/features/lab/shaders:negpy/features/lab/shaders",
    "--add-data=negpy/features/lith/shaders:negpy/features/lith/shaders",
    "--add-data=negpy/features/cyanotype/shaders:negpy/features/cyanotype/shaders",
    "--add-data=negpy/features/finish/shaders:negpy/features/finish/shaders",
    "--add-data=negpy/desktop/view/styles:negpy/desktop/view/styles",
    "--add-data=icc:icc",
    "--add-data=media:media",
    "--add-data=crosstalk:crosstalk",
    "--add-data=gear:gear",
    "--add-data=VERSION:.",
    # The panel guides (section_help_dialog.py) render slices of it at runtime.
    "--add-data=docs/USER_GUIDE.md:docs",
]


def collect_gphoto2_plugins() -> None:
    """Ship libgphoto2's camera and I/O drivers with their directory layout intact.

    python-gphoto2 points `CAMLIBS`/`IOLIBS` at `<package>/libgphoto2/{camlibs,iolibs}`
    when it is imported, and libgphoto2 dlopen's every driver from there. PyInstaller's
    --collect-all flattens those .so files in among the other binaries, so the tree the
    library actually looks for is missing and *every* camera fails to connect. Re-add the
    two directories verbatim, at the path the env vars will resolve to.

    Camera scanning is an optional extra: `uv sync --group camera` before building, or the
    packaged app simply shows its setup hint.
    """
    if is_windows:
        return  # libgphoto2 has no Windows build
    try:
        import gphoto2
    except ImportError:
        print("gphoto2 not installed — packaging without camera scanning")
        return
    plugins = os.path.join(os.path.dirname(gphoto2.__file__), "libgphoto2")
    if not os.path.isdir(plugins):
        print(f"WARNING: gphoto2 plugins not found at {plugins} — camera scanning will not work")
        return
    params.append(f"--add-data={plugins}:gphoto2/libgphoto2")
    print(f"Bundling libgphoto2 drivers from {plugins}")


collect_gphoto2_plugins()

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

    macos_target_arch = get_macos_target_arch()
    params.append(f"--target-arch={macos_target_arch}")


def package_linux():
    """Package the built application into an AppImage."""
    print("Packaging for Linux (AppImage)...")
    dist_dir = os.path.join("dist", APP_NAME)
    appdir = os.path.join("dist", f"{APP_NAME}.AppDir")

    if os.path.exists(appdir):
        shutil.rmtree(appdir)

    # 1. Create AppDir structure
    shutil.copytree(dist_dir, appdir)

    # 2. De-bundle system graphics and UI libraries
    # This ensures the AppImage uses host drivers and platform plugins.
    libs_to_remove = [
        "libvulkan.so*",
        "libGL.so*",
        "libGLX.so*",
        "libEGL.so*",
        "libGLESv2.so*",
        "libgbm.so*",
        "libdrm.so*",
        "libX11*",
        "libXext.so*",
        "libXfixes.so*",
        "libXrender.so*",
        "libxshmfence.so*",
        "libstdc++.so*",
        "libz.so*",
        "libgcc_s.so*",
        "libdbus-1.so*",
        "libfontconfig.so*",
        "libfreetype.so*",
        "libexpat.so*",
        # Must use host libsane so SANE can locate backend plugins in /usr/lib/sane/.
        # libusb and libudev are transitive deps of libsane collected by PyInstaller;
        # bundling Ubuntu versions causes SANE backends on other distros to silently
        # find no USB devices (LD_LIBRARY_PATH serves wrong version first).
        "libsane.so*",
        "libusb-1.0.so*",
        "libusb-0.1.so*",
        "libudev.so*",
        "libjpeg.so*",
    ]
    print("De-bundling system libraries from AppDir...")
    for pattern in libs_to_remove:
        search_pattern = os.path.join(appdir, "**", pattern)
        for libpath in glob.glob(search_pattern, recursive=True):
            try:
                if os.path.isfile(libpath) or os.path.islink(libpath):
                    basename = os.path.basename(libpath)
                    # Safety check: Don't remove libraries with mangled names (containing '-')
                    # unless they are known system libraries or extensions.
                    # This protects Python wheels like Pillow/OpenCV.
                    system_prefixes = [
                        "dbus-",
                        "stdc++",
                        "gcc_s",
                        "wayland-",
                        "xkbcommon-",
                        "usb-",  # libusb-1.0.so* — system USB lib, not a wheel
                    ]
                    if "-" in basename and not any(p in basename for p in system_prefixes):
                        continue

                    os.remove(libpath)
                    print(f"  Removed: {os.path.relpath(libpath, appdir)}")
            except Exception as e:
                print(f"  Failed to remove {libpath}: {e}")

    # 3. Clear executable stack flag from Python shared library
    # Python 3.13's libpython sets PT_GNU_STACK RWE which modern kernels reject.
    print("Clearing executable stack flag from bundled Python library...")
    for libpath in glob.glob(os.path.join(appdir, "**", "libpython*.so*"), recursive=True):
        if os.path.isfile(libpath) and not os.path.islink(libpath):
            cleared = False
            for tool, args in [
                ("patchelf", ["patchelf", "--clear-execstack", libpath]),
                ("execstack", ["execstack", "-c", libpath]),
            ]:
                try:
                    subprocess.run(args, check=True, capture_output=True, text=True)
                    print(f"  Cleared execstack ({tool}): {os.path.relpath(libpath, appdir)}")
                    cleared = True
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            if not cleared:
                print(f"  WARNING: Could not clear execstack from {os.path.relpath(libpath, appdir)} — install patchelf or execstack")

    # 4. Add Desktop file and Icon
    shutil.copy("negpy.desktop", os.path.join(appdir, "negpy.desktop"))
    shutil.copy("media/icons/icon.png", os.path.join(appdir, "icon.png"))

    # 5. Create AppRun script
    apprun_path = os.path.join(appdir, "AppRun")
    with open(apprun_path, "w") as f:
        f.write("#!/bin/sh\n")
        f.write('HERE="$(dirname "$(readlink -f "${0}")")"\n')
        # Point to the bundled libraries
        f.write('export LD_LIBRARY_PATH="$HERE/_internal:$HERE:$LD_LIBRARY_PATH"\n')
        # Priority: Wayland then XCB. This is safer for modern distros while providing xcb fallback.
        f.write('export QT_QPA_PLATFORM="wayland;xcb"\n')
        # Disable X11 shared memory extension to prevent crashes with newer X servers
        f.write("export QT_X11_NO_MITSHM=1\n")
        # Hint WGPU to use Vulkan
        f.write('export WGPU_BACKEND_TYPE="Vulkan"\n')
        f.write(f'exec "${{HERE}}/{APP_NAME}" "$@"\n')
    os.chmod(apprun_path, 0o755)

    # 6. Run appimagetool
    try:
        tool = "./appimagetool-x86_64.AppImage"
        if not os.path.exists(tool):
            tool = "appimagetool"

        output_filename = os.path.join("dist", f"{APP_NAME}-{VERSION}-x86_64.AppImage")

        # Ensure ARCH is set for appimagetool, often required in CI
        env = os.environ.copy()
        env["ARCH"] = "x86_64"

        result = subprocess.run(
            [tool, appdir, output_filename],
            check=False,  # We handle check manually to print output
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            print(f"AppImageTool failed with exit code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise subprocess.CalledProcessError(
                result.returncode,
                result.args,
                output=result.stdout,
                stderr=result.stderr,
            )

        print(f"AppImage created: {output_filename}")
    except Exception as e:
        print(f"Error creating AppImage: {e}")
        raise


def package_windows():
    """Package the built application into an NSIS installer."""
    print(f"Packaging for Windows (NSIS) version {VERSION}...")

    cmd = "makensis"
    # Try to find makensis in common locations if not in PATH
    found_cmd = shutil.which(cmd) or shutil.which("makensis.exe")

    if not found_cmd:
        common_paths = [
            r"C:\Program Files (x86)\NSIS\makensis.exe",
            r"C:\Program Files\NSIS\makensis.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                cmd = p
                break
    else:
        cmd = found_cmd

    try:
        setup_name = f"{APP_NAME}-{VERSION}-Win64-Setup.exe"
        # On Windows, using shell=True helps resolving commands in PATH
        subprocess.run(
            [cmd, f"/DVERSION={VERSION}", f"/DOUTFILE={setup_name}", "installer.nsi"],
            check=True,
            shell=is_windows,
        )
        print(f"Windows Installer created: dist/{setup_name}")
    except Exception as e:
        print(f"Error creating Windows Installer: {e}")
        raise


# Every .so/.dylib under Contents/Frameworks known to link liblcms2.2.dylib.
# See LIBLCMS2_DYLIB_COLLISION.md for how this list and fix_lcms2_dylib_collision()
# were derived and verified.
_LCMS2_CONSUMER_GLOBS = [
    "PIL/_imagingcms*.so",
    "rawpy/libraw_r*.dylib",
    "imagecodecs/_cms.abi3.so",
    "imagecodecs/_jpeg2k.abi3.so",
    "libjxl_cms*.dylib",
    "cv2/*/libjxl_cms*.dylib",
]


def fix_lcms2_dylib_collision():
    """Repoint the canonical liblcms2.2.dylib at imagecodecs' own copy.

    Raises if imagecodecs' own copy can't be found unambiguously, or if any
    known consumer would be missing a symbol it needs from it — better a red
    build than a silently shipped repeat of this bug. See
    LIBLCMS2_DYLIB_COLLISION.md. Must run before codesign_macos_app() so the
    corrected symlink is covered by the final signature.
    """
    frameworks = os.path.join("dist", f"{APP_NAME}.app", "Contents", "Frameworks")
    canonical = os.path.join(frameworks, "liblcms2.2.dylib")
    matches = glob.glob(os.path.join(frameworks, "imagecodecs", "*", "liblcms2.2.dylib"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one imagecodecs liblcms2.2.dylib under {frameworks}, found {matches}")
    rel_target = os.path.relpath(matches[0], frameworks)
    if os.path.islink(canonical) or os.path.exists(canonical):
        os.remove(canonical)
    os.symlink(rel_target, canonical)
    print(f"Repointed {canonical} -> {rel_target} (imagecodecs' own liblcms2.2.dylib)")

    exports = _nm_symbols(matches[0], defined=True)
    gaps = []
    for pattern in _LCMS2_CONSUMER_GLOBS:
        for consumer in glob.glob(os.path.join(frameworks, pattern)):
            needed = {s for s in _nm_symbols(consumer, defined=False) if s.startswith("_cms")}
            missing = needed - exports
            if missing:
                gaps.append((consumer, sorted(missing)))
    if gaps:
        detail = "\n".join(f"  {os.path.relpath(c, frameworks)}: missing {m}" for c, m in gaps)
        raise RuntimeError(f"imagecodecs' liblcms2.2.dylib is missing symbols other consumers need:\n{detail}")


@functools.lru_cache(maxsize=None)
def _nm_symbols(path: str, *, defined: bool) -> frozenset[str]:
    """Return a Mach-O file's defined-exported (-gU) or undefined (-u) symbol names."""
    flag = "-gU" if defined else "-u"
    out = subprocess.run(["nm", flag, path], capture_output=True, text=True, check=True).stdout
    if defined:
        return frozenset(line.split()[-1] for line in out.splitlines() if line.strip())
    return frozenset(line.strip() for line in out.splitlines() if line.strip())


@functools.lru_cache(maxsize=None)
def _dylib_load_basenames(path: str) -> frozenset[str]:
    """Basenames a Mach-O file references via LC_LOAD_DYLIB (its real, linked-in dependencies)."""
    out = subprocess.run(["otool", "-L", path], capture_output=True, text=True, check=True).stdout
    return frozenset(os.path.basename(line.split()[0]) for line in out.splitlines()[1:] if line.strip())


def check_bundled_dylib_collisions():
    """Verify every same-basename dylib PyInstaller collapsed to one canonical
    copy still satisfies every real consumer's symbols.

    Generalizes fix_lcms2_dylib_collision() to the whole bundle: cv2, PIL,
    rawpy and imagecodecs each vendor their own copies of common libraries
    (libjpeg, libpng, libtiff, ...) under identical filenames, and PyInstaller
    keeps only one arbitrary pick as the canonical @rpath target for all of
    them -- the exact bug class in LIBLCMS2_DYLIB_COLLISION.md, which this
    scans for instead of relying on a curated list of known-risky basenames
    (that fix's own consumer glob missed a real libjpeg collision on first
    pass -- see the doc). Raises rather than warns, so a future dependency
    bump that silently drops a symbol fails the build instead of shipping
    broken silently.

    Does not check libraries PyInstaller fails to bundle at all (e.g. numba's
    optional libomp.dylib) -- that is a different failure mode (absence, not
    a collision) with no evidence of user impact; see LIBLCMS2_DYLIB_COLLISION.md.
    """
    frameworks = os.path.join("dist", f"{APP_NAME}.app", "Contents", "Frameworks")
    all_files = [
        os.path.join(dirpath, name)
        for dirpath, _, filenames in os.walk(frameworks)
        for name in filenames
        if name.endswith((".dylib", ".so"))
    ]
    by_basename: dict[str, list[str]] = {}
    for path in all_files:
        if not os.path.islink(path):
            by_basename.setdefault(os.path.basename(path), []).append(path)

    gaps = []
    for basename, copies in by_basename.items():
        canonical = os.path.join(frameworks, basename)
        if not os.path.islink(canonical):
            continue  # no top-level collapse for this basename -- nothing collided
        canonical_real = os.path.realpath(canonical)
        distinct = {p for p in copies if os.path.realpath(p) != canonical_real}
        if not distinct:
            continue  # every copy is byte-identical -- an arbitrary pick can't lose symbols
        exports = _nm_symbols(canonical_real, defined=True)
        provided_anywhere: set[str] = set(exports)
        for p in distinct:
            provided_anywhere |= _nm_symbols(p, defined=True)

        for consumer in all_files:
            if os.path.islink(consumer) or os.path.realpath(consumer) == canonical_real or consumer in distinct:
                continue
            if basename not in _dylib_load_basenames(consumer):
                continue
            needed = _nm_symbols(consumer, defined=False) & provided_anywhere
            missing = needed - exports
            if missing:
                gaps.append((consumer, basename, sorted(missing)))

    if gaps:
        detail = "\n".join(f"  {os.path.relpath(c, frameworks)} needs {b}: missing {m}" for c, b, m in gaps)
        raise RuntimeError(f"a bundled dylib collision leaves a consumer missing symbols:\n{detail}")


def codesign_macos_app():
    """Apply a fresh, consistent ad-hoc signature over the whole .app.

    PyInstaller's own signing pass does not reliably reach every binary it
    moves or rewrites under --collect-all.
    """
    app_path = os.path.join("dist", f"{APP_NAME}.app")
    print(f"Re-signing {app_path} (ad-hoc, deep)...")
    subprocess.run(["codesign", "--force", "--deep", "-s", "-", app_path], check=True)


def package_macos():
    """Package the built application into a DMG with Applications symlink."""
    print(f"Packaging for macOS (DMG) version {VERSION}...")
    app_path = os.path.join("dist", f"{APP_NAME}.app")
    dmg_name = f"{APP_NAME}-{VERSION}-macOS-{get_macos_target_arch()}.dmg"
    dmg_path = os.path.join("dist", dmg_name)
    temp_dmg_dir = os.path.join("dist", "dmg_temp")

    if os.path.exists(dmg_path):
        os.remove(dmg_path)
    if os.path.exists(temp_dmg_dir):
        shutil.rmtree(temp_dmg_dir)

    os.makedirs(temp_dmg_dir)

    try:
        # 1. Copy .app to temp dir (preserve symlinks for macOS bundles)
        shutil.copytree(app_path, os.path.join(temp_dmg_dir, f"{APP_NAME}.app"), symlinks=True)

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
            print(f"Contents of dist/{APP_NAME}: {os.listdir(f'dist/{APP_NAME}')[:10]}... (truncated)")
    else:
        print("ERROR: dist directory not found!")

    if is_linux:
        package_linux()
    elif is_windows:
        package_windows()
    elif is_macos:
        fix_lcms2_dylib_collision()
        check_bundled_dylib_collisions()
        codesign_macos_app()
        package_macos()


if __name__ == "__main__":
    build()
