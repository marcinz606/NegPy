# DarkroomPy Desktop Integration

This document outlines the architecture and workflows for running DarkroomPy as a standalone desktop application using **Electron** and a **PyInstaller sidecar**.

## Architecture Overview

DarkroomPy is built as a hybrid desktop application:
- **Frontend**: Electron provides the native window shell and manages the lifecycle of the backend.
- **Backend**: The original Python/Streamlit logic is bundled into a single-file executable using PyInstaller.
- **Communication**: Electron spawns the Python binary as a background process and loads the Streamlit UI (localhost:8501) into its main window.

## Development Workflow

### 1. Docker (Standard)
Your existing Docker workflow remains unchanged.
```bash
python start.py
# OR
docker-compose up
```

### 2. Electron Development
To run the Electron shell while developing:
1. Ensure your local virtual environment is set up:
   ```bash
   pip install -r requirements.txt
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. Start the application:
   ```bash
   npm start
   ```
*Note: In development mode, Electron uses the Python executable inside your `./venv` to run the code directly.*

## Packaging and Distribution

The build process involves two steps: bundling the Python engine and then packaging the Electron app.

### Manual Build (Linux/Local)
```bash
npm run dist
```
This runs `build_backend.py` (PyInstaller) and then `electron-builder`. Output is found in `dist/`.

### Automated Build (GitHub Actions)
Pushing a tag (e.g., `v1.0.0`) triggers `.github/workflows/release.yml`, which builds installers for all three platforms simultaneously.

## GitHub Releases & Automated Builds

The project is configured to automatically build and package the application for Windows, macOS, and Linux whenever a new version tag is pushed to GitHub.

### 1. Preparing a Release
Before triggering a build, it is good practice to tag your code for versioning:

```bash
git tag v1.0.0
git push origin v1.0.0
```

### 2. Triggering the Build (Manual)
Unlike standard tasks, the release build does **not** start automatically when you push a tag. You must trigger it manually:
1. Go to your GitHub repository in your browser.
2. Click on the **Actions** tab.
3. Select the **Build and Release** workflow from the left sidebar.
4. Click the **Run workflow** button.
5. Select the **tag** or **branch** you want to build from the dropdown.
6. Enter the version name (e.g., `v1.0.0`) in the input field.
7. Select the **Platform** you want to build (All, Windows, Linux, or macOS).
8. Click **Run workflow**.

### 3. Collecting Artifacts
Once the workflow completes:
- **Draft Release**: By default, `electron-builder` will create a Draft Release in the **Releases** section of your GitHub repo.
- **Assets**: The `.exe`, `.dmg`, and `.AppImage` files will be automatically uploaded to that release.
- **Publishing**: Review the draft, add your release notes, and click **Publish release** to make it visible to the public.

---

## Troubleshooting & Technical Tips

### Persistence & Data
The application uses a dedicated directory for user settings, databases, and exports. 
- **Path**: `~/Documents/DarkroomPy` (Platform-agnostic).
- This is controlled via the `DARKROOM_USER_DIR` environment variable, passed from Electron to the Python backend.

### Linux
- **Targets**: AppImage and .deb.
- **Dependencies**: Requires `libgl1` (OpenCV) and `build-essential` if compiling native Node modules.
- **Note**: The `.deb` build requires a valid author email in `package.json`.

### Windows
- **Targets**: NSIS Installer (.exe).
- **Backend**: PyInstaller creates `backend.exe`.
- **Process Management**: On Windows, Electron uses `taskkill` to ensure the Python sidecar is fully terminated when the window is closed.

### macOS
- **Targets**: DMG.
- **Architecture**: The current workflow builds for the architecture of the runner (Intel or Apple Silicon).
- **Signing**: For production distribution outside of GitHub, the app would require an Apple Developer ID and notarization.

## Troubleshooting & Technical Tips

### Common Build Issues

- **`PackageNotFoundError: No package metadata was found for streamlit/imageio`**: 
  Streamlit and Imageio use `importlib.metadata` at runtime to check their versions. PyInstaller must be explicitly told to copy this metadata using the `--copy-metadata` flag in `build_backend.py`.

- **`ModuleNotFoundError` in Bundled App**:
  PyInstaller's static analysis often misses dynamic imports used by Streamlit. All major libraries (rawpy, cv2, numpy, PIL, scipy, matplotlib) should be added to the `hiddenimports` list in the build script.

- **`RuntimeError: server.port does not work when global.developmentMode is true`**:
  In a bundled environment, Streamlit must be forced into production mode. We pass `--global.developmentMode=false` in `run_app.py` to allow custom port configuration.

- **`FileNotFoundError` for bundled assets (like `icc/`)**:
  When packaged, the app runs from a temporary directory (`_MEIPASS`). `run_app.py` must change the working directory to this path (`os.chdir(bundle_dir)`) so that relative paths in the Python code correctly find bundled assets.

- **`ENOTDIR` when spawning backend**:
  Electron cannot set its `cwd` to a path inside the `app.asar` archive. In `desktop/main.js`, we detect `isPackaged` and set the `cwd` to `process.resourcesPath` to ensure the process starts in a physical directory.

### Native Dependencies
- **Linux**: The AppImage bundles most things, but the host system still needs `libgl1` (for OpenCV) and standard graphics drivers.
- **Windows**: Ensure C++ Redistributables are installed if you encounter DLL loading errors.
