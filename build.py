import ctypes
import glob
import importlib
import importlib.util
import os
import platform
import plistlib
import re
import shutil
import subprocess
from pathlib import Path

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

COOLSCAN_FEATURE_ENV = "NEGPY_ENABLE_COOLSCAN"
COOLSCAN_LIBUSB_ENV = "NEGPY_LIBUSB_DYLIB"
MACOS_CODESIGN_IDENTITY_ENV = "NEGPY_CODESIGN_IDENTITY"
MACOS_BUNDLE_IDENTIFIER = "org.negpy.NegPy"
MACOS_ENTITLEMENTS_FILE = str(Path(__file__).resolve().parent / "macos" / "NegPy.entitlements")
MACOS_RUNTIME_ENTITLEMENTS = {
    "com.apple.security.cs.allow-unsigned-executable-memory": True,
}
MACOS_CS_RUNTIME_FLAG = 0x10000
COOLSCAN_LIBUSB_DESTINATION = "coolscanpy/_native"
COOLSCAN_LIBUSB_FILENAME = "libusb-1.0.dylib"
CAPTURE_HELPER_FLAG = "--ls5000-capture-helper"
LIVE_ACCEPTANCE_FLAG = "--ls5000-live-acceptance"
PACKAGING_SMOKE_FLAG = "--negpy-packaging-smoke"
PACKAGING_SMOKE_OK = "NegPy frozen scanner packaging smoke passed"


def _coolscan_support_enabled() -> bool:
    """Enable the optional frozen scanner seam when its build group is present."""

    configured = os.environ.get(COOLSCAN_FEATURE_ENV)
    available = importlib.util.find_spec("coolscanpy") is not None
    if configured is None:
        enabled = available
    elif configured.strip().lower() in {"1", "true", "yes", "on"}:
        enabled = True
    elif configured.strip().lower() in {"0", "false", "no", "off"}:
        enabled = False
    else:
        raise RuntimeError(f"{COOLSCAN_FEATURE_ENV} must be one of 1/0, true/false, yes/no, or on/off")

    if enabled and not available:
        raise RuntimeError("Coolscan support was requested but coolscanpy is unavailable; sync the coolscan-roll build group first")
    if enabled and importlib.util.find_spec("portable_digital_ice") is None:
        raise RuntimeError("Coolscan support was requested but portable_digital_ice is unavailable; sync the fauxice build group first")
    return enabled


def resolve_macos_libusb_dylib() -> str:
    """Resolve one absolute, architecture-compatible libusb 1.0 dylib."""

    override = os.environ.get(COOLSCAN_LIBUSB_ENV)
    if override:
        if not os.path.isabs(override):
            raise RuntimeError(f"{COOLSCAN_LIBUSB_ENV} must be an absolute path")
        candidates = (override,)
    elif platform.machine() == "arm64":
        candidates = (
            "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",
            "/opt/local/lib/libusb-1.0.dylib",
            "/usr/local/opt/libusb/lib/libusb-1.0.dylib",
        )
    else:
        candidates = (
            "/usr/local/opt/libusb/lib/libusb-1.0.dylib",
            "/opt/local/lib/libusb-1.0.dylib",
            "/opt/homebrew/opt/libusb/lib/libusb-1.0.dylib",
        )

    rejected: list[str] = []
    for candidate in candidates:
        # Preserve the stable alias instead of resolving Homebrew's symlink to
        # libusb-1.0.0.dylib: PyInstaller uses the source basename at the
        # requested destination, and the frozen resolver requires this exact
        # package-owned filename.
        path = Path(os.path.abspath(candidate))
        if not path.is_file():
            rejected.append(f"{candidate} (missing)")
            continue
        if path.name != COOLSCAN_LIBUSB_FILENAME:
            rejected.append(f"{path} (filename must be {COOLSCAN_LIBUSB_FILENAME})")
            continue
        architecture = subprocess.run(
            ["/usr/bin/lipo", str(path), "-verify_arch", platform.machine()],
            check=False,
            capture_output=True,
            text=True,
        )
        if architecture.returncode != 0:
            rejected.append(f"{path} (wrong architecture)")
            continue
        return str(path)

    details = "; ".join(rejected)
    raise RuntimeError(
        "Coolscan support requires an architecture-compatible libusb 1.0 dylib. "
        f"Install it with Homebrew or set {COOLSCAN_LIBUSB_ENV} to an absolute path. "
        f"Checked: {details}"
    )


COOLSCAN_SUPPORT_ENABLED = _coolscan_support_enabled()
MACOS_LIBUSB_SOURCE = resolve_macos_libusb_dylib() if is_macos and COOLSCAN_SUPPORT_ENABLED else None
MACOS_CODESIGN_IDENTITY = os.environ.get(MACOS_CODESIGN_IDENTITY_ENV, "-").strip()
if is_macos and not MACOS_CODESIGN_IDENTITY:
    raise RuntimeError(f"{MACOS_CODESIGN_IDENTITY_ENV} must not be empty")

_REQUIRED_COOLSCAN_API = {
    "coolscanpy": (
        "DigitalIceAcquisition",
        "DigitalIceAcquisitionEvidence",
        "build_digital_ice_acquisition_evidence",
    ),
    "coolscanpy.protocol.ls5000_single_pass.density": (
        "NikonDensityEvidence",
        "NikonExactBuilderEvidence",
        "build_nikon_density_evidence",
        "build_nikon_exact_builder_evidence",
    ),
    "coolscanpy.protocol.ls5000_single_pass.bundle": ("verify_capture_bundle",),
}


def preflight_coolscan_runtime() -> str:
    """Reject a stale or unsealed Coolscan checkout before PyInstaller runs."""

    loaded: dict[str, object] = {}
    missing: list[str] = []
    for module_name, symbols in _REQUIRED_COOLSCAN_API.items():
        module = importlib.import_module(module_name)
        loaded[module_name] = module
        missing.extend(f"{module_name}.{symbol}" for symbol in symbols if not callable(getattr(module, symbol, None)))
    if missing:
        raise RuntimeError("Coolscan build API is stale or incomplete: " + ", ".join(missing))

    bundle_module = loaded["coolscanpy.protocol.ls5000_single_pass.bundle"]
    verify_capture_bundle = getattr(bundle_module, "verify_capture_bundle")
    bundle_sha256 = verify_capture_bundle(require_python_sources=True)
    if (
        not isinstance(bundle_sha256, str)
        or len(bundle_sha256) != 64
        or any(character not in "0123456789abcdef" for character in bundle_sha256)
    ):
        raise RuntimeError("Coolscan capture bundle returned an invalid SHA-256 identity")
    return bundle_sha256


# Basic PyInstaller arguments
params = [
    ENTRY_POINT,
    f"--name={APP_NAME}",
    "--onedir",
    "--windowed",  # GUI app, no console
    "--clean",
    "--noconfirm",
    *([f"--osx-bundle-identifier={MACOS_BUNDLE_IDENTIFIER}"] if is_macos else []),
    *([f"--osx-entitlements-file={MACOS_ENTITLEMENTS_FILE}"] if is_macos else []),
    *([f"--codesign-identity={MACOS_CODESIGN_IDENTITY}"] if is_macos and MACOS_CODESIGN_IDENTITY != "-" else []),
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
    # The frozen app re-enters itself with this worker flag. Keep the worker
    # explicit even though --collect-all also finds it, so that entry contract
    # cannot disappear in a future package layout change.
    *(["--hidden-import=coolscanpy.protocol.ls5000_single_pass.worker"] if COOLSCAN_SUPPORT_ENABLED else []),
    *(["--hidden-import=negpy.services.roll.live_acceptance"] if COOLSCAN_SUPPORT_ENABLED else []),
    # Scanner support: bundle the python-sane C extension but NOT libsane.so.1.
    # libsane.so.1 must come from the host so SANE can find its backend plugins
    # in /usr/lib/sane/. See libs_to_remove in package_linux().
    # Requires: uv sync --group scanner before building on Linux/macOS.
    *([] if is_windows else ["--hidden-import=sane", "--hidden-import=_sane"]),
    # Camera scanning: see collect_gphoto2_plugins() — the plugin trees need their
    # directory layout preserved, which --collect-all does not do.
    *([] if is_windows else ["--collect-all=gphoto2"]),
    # Exclude unused modules
    # Metadata
    "--copy-metadata=imageio",
    "--copy-metadata=rawpy",
    "--collect-all=wgpu",
    "--collect-all=rawpy",
    "--collect-all=imageio",
    "--collect-all=imagecodecs",
    *(["--collect-all=coolscanpy"] if COOLSCAN_SUPPORT_ENABLED else []),
    "--collect-all=portable_digital_ice",
    *([f"--add-binary={MACOS_LIBUSB_SOURCE}:{COOLSCAN_LIBUSB_DESTINATION}"] if MACOS_LIBUSB_SOURCE is not None else []),
    # Data files
    "--add-data=negpy/features/exposure/shaders:negpy/features/exposure/shaders",
    "--add-data=negpy/features/geometry/shaders:negpy/features/geometry/shaders",
    "--add-data=negpy/features/toning/shaders:negpy/features/toning/shaders",
    "--add-data=negpy/features/retouch/shaders:negpy/features/retouch/shaders",
    "--add-data=negpy/features/lab/shaders:negpy/features/lab/shaders",
    "--add-data=negpy/features/finish/shaders:negpy/features/finish/shaders",
    "--add-data=negpy/desktop/view/styles:negpy/desktop/view/styles",
    "--add-data=negpy/assets/portable_builder:negpy/assets/portable_builder",
    "--add-data=negpy/assets/portable_cms:negpy/assets/portable_cms",
    "--add-data=negpy/services/roll/portable_oracle_evaluator.py:negpy/services/roll",
    "--add-data=icc:icc",
    "--add-data=media:media",
    "--add-data=crosstalk:crosstalk",
    "--add-data=gear:gear",
    "--add-data=VERSION:.",
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


def package_macos():
    """Package the built application into a DMG with Applications symlink."""
    print(f"Packaging for macOS (DMG) version {VERSION}...")
    app_path = os.path.join("dist", f"{APP_NAME}.app")
    dmg_name = f"{APP_NAME}-{VERSION}-macOS-{platform.machine()}.dmg"
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


def verify_macos_runtime_entitlements(app_path: str) -> dict[str, bool]:
    """Fail closed unless the signed host has the one required JIT exception."""

    app_path = os.path.abspath(app_path)
    completed = subprocess.run(
        [
            "/usr/bin/codesign",
            "--display",
            "--entitlements",
            "-",
            "--xml",
            app_path,
        ],
        check=True,
        capture_output=True,
    )
    try:
        embedded = plistlib.loads(completed.stdout)
    except (plistlib.InvalidFileException, ValueError, TypeError) as error:
        raise RuntimeError("signed NegPy app has unreadable hardened-runtime entitlements") from error
    if embedded != MACOS_RUNTIME_ENTITLEMENTS:
        raise RuntimeError(
            f"signed NegPy app has unexpected hardened-runtime entitlements: expected {MACOS_RUNTIME_ENTITLEMENTS!r}, got {embedded!r}"
        )
    return embedded


def verify_macos_hardened_runtime(app_path: str) -> int:
    """Fail closed unless the host CodeDirectory enables hardened runtime."""

    app_path = os.path.abspath(app_path)
    completed = subprocess.run(
        ["/usr/bin/codesign", "--display", "--verbose=4", app_path],
        check=True,
        capture_output=True,
        text=True,
    )
    details = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r"\bflags=0x([0-9a-fA-F]+)\b", details)
    flags = int(match.group(1), 16) if match is not None else 0
    if flags & MACOS_CS_RUNTIME_FLAG == 0:
        raise RuntimeError("signed NegPy app is missing the hardened-runtime CodeDirectory flag")
    return flags


def sign_macos_scanner_runtime(app_path: str) -> str:
    """Sign the bundled libusb dylib, reseal the app, and verify both."""

    app_path = os.path.abspath(app_path)

    frameworks_libusb = os.path.join(
        app_path,
        "Contents",
        "Frameworks",
        COOLSCAN_LIBUSB_DESTINATION,
        COOLSCAN_LIBUSB_FILENAME,
    )
    resources_libusb = os.path.join(
        app_path,
        "Contents",
        "Resources",
        COOLSCAN_LIBUSB_DESTINATION,
        COOLSCAN_LIBUSB_FILENAME,
    )
    bundled_libusb = next(
        (path for path in (frameworks_libusb, resources_libusb) if os.path.isfile(path)),
        None,
    )
    if bundled_libusb is None:
        raise RuntimeError(
            f"PyInstaller did not place the required libusb dylib at {COOLSCAN_LIBUSB_DESTINATION}/{COOLSCAN_LIBUSB_FILENAME}"
        )

    identity = os.environ.get(MACOS_CODESIGN_IDENTITY_ENV, "-").strip()
    if not identity:
        raise RuntimeError(f"{MACOS_CODESIGN_IDENTITY_ENV} must not be empty")
    timestamp_option = "--timestamp=none" if identity == "-" else "--timestamp"
    codesign = "/usr/bin/codesign"

    subprocess.run(
        [codesign, "--force", "--sign", identity, timestamp_option, bundled_libusb],
        check=True,
    )
    subprocess.run(
        [
            codesign,
            "--force",
            "--sign",
            identity,
            timestamp_option,
            # PyInstaller applies this same policy to the initial bundle
            # signature. Reapply it explicitly here because changing the
            # bundled libusb invalidates the outer seal; relying on ambient
            # entitlement preservation made Developer ID builds fragile.
            "--entitlements",
            MACOS_ENTITLEMENTS_FILE,
            "--preserve-metadata=requirements,flags,runtime",
            app_path,
        ],
        check=True,
    )
    subprocess.run(
        [codesign, "--verify", "--strict", "--verbose=2", bundled_libusb],
        check=True,
    )
    subprocess.run(
        [codesign, "--verify", "--deep", "--strict", "--verbose=2", app_path],
        check=True,
    )
    verify_macos_runtime_entitlements(app_path)
    if identity != "-":
        verify_macos_hardened_runtime(app_path)
    return bundled_libusb


def smoke_macos_scanner_runtime(app_path: str) -> dict[str, str]:
    """Exercise the signed frozen seams without opening Qt or enumerating USB."""

    app_path = os.path.abspath(app_path)

    executable = os.path.join(app_path, "Contents", "MacOS", APP_NAME)
    bundled_libusb = os.path.join(
        app_path,
        "Contents",
        "Frameworks",
        COOLSCAN_LIBUSB_DESTINATION,
        COOLSCAN_LIBUSB_FILENAME,
    )
    if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise RuntimeError(f"frozen NegPy executable is missing or not executable: {executable}")
    if not os.path.isfile(bundled_libusb):
        raise RuntimeError(f"frozen libusb is missing: {bundled_libusb}")

    # Loading the library and resolving symbols does not initialize a libusb
    # context and therefore cannot enumerate or claim any USB device.
    libusb = ctypes.CDLL(bundled_libusb)
    for symbol in ("libusb_init", "libusb_exit"):
        if not hasattr(libusb, symbol):
            raise RuntimeError(f"frozen libusb is missing required symbol {symbol}")

    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    for variable in (
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
    ):
        environment.pop(variable, None)

    def run_offline(arguments: list[str], *, label: str) -> str:
        if LIVE_ACCEPTANCE_FLAG in arguments:
            raise RuntimeError(f"{label} attempted to enable live acceptance")
        if "--live" in arguments:
            raise RuntimeError(f"{label} attempted to enable live scanner access")
        completed = subprocess.run(
            [executable, *arguments],
            cwd=os.path.dirname(app_path),
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{label} failed with exit code {completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        return completed.stdout

    packaging_output = run_offline([PACKAGING_SMOKE_FLAG], label="frozen import/assets smoke")
    if PACKAGING_SMOKE_OK not in packaging_output:
        raise RuntimeError("frozen import/assets smoke did not emit its success receipt")

    helper_output = run_offline(
        [
            CAPTURE_HELPER_FLAG,
            "--frame",
            "18",
            "--boundary-offset-rows",
            "0",
            "--confirm-full-capture",
        ],
        label="frozen LS-5000 helper dry-run",
    )
    for expected in ("619458560 bytes", "dry run only; scanner was not accessed"):
        if expected not in helper_output:
            raise RuntimeError(f"frozen LS-5000 helper dry-run did not emit {expected!r}")

    return {
        "executable": executable,
        "libusb": bundled_libusb,
        "packaging_output": packaging_output,
        "helper_output": helper_output,
    }


def build():
    print(f"Building {APP_NAME} for {system}...")
    if COOLSCAN_SUPPORT_ENABLED:
        bundle_sha256 = preflight_coolscan_runtime()
        print(f"Verified Coolscan capture bundle: {bundle_sha256}")
    print("PyInstaller parameters:", params)

    PyInstaller.__main__.run(params)

    if is_macos and COOLSCAN_SUPPORT_ENABLED:
        sign_macos_scanner_runtime(os.path.join("dist", f"{APP_NAME}.app"))
        smoke_macos_scanner_runtime(os.path.join("dist", f"{APP_NAME}.app"))

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
        package_macos()


if __name__ == "__main__":
    build()
