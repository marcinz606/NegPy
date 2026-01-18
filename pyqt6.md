# PyQt6 Migration Blueprint: Comprehensive Professional Rewrite

## 1. Vision & Objectives
This document outlines the high-fidelity migration from Streamlit to PyQt6. The goal is to achieve 100% feature parity while introducing native desktop advantages: 10-bit color, GPU acceleration, and zero-latency UI.

---

## 2. Component Migration Map

### A. Sidebar Components (Dockable Panels)
| Streamlit Section | PyQt6 Implementation | Logic / Signal |
| :--- | :--- | :--- |
| **File Manager** | `QDockWidget` + `QListView` | `fileSelected(path)`, `scanStarted()` |
| **Hot Folder Mode** | `QFileSystemWatcher` | `fileDiscovered(path)` -> Auto Load |
| **Analysis UI** | `QChartView` (Histograms) | Real-time luminance updates |
| **Exposure & Tonality** | `QFormLayout` + `QSlider` | `valueChanged(float)` -> `RenderTask` |
| **Lab Scanner** | `QGroupBox` + `CustomSliders` | Sync with `WorkspaceConfig` |
| **Retouch (D & B)** | `QListWidget` (Layers) | Mask activation/visibility signals |
| **Geometry** | `QToolButton` Grid | Rotation/Flip signals; CropMode enum |
| **Export UI** | `QWizard` or `QDialog` | `exportStarted(ExportConfig)` |
| **Presets** | `QComboBox` + `QPushButton` | `presetApplied(name)` |

### B. Viewport & Header
| Feature | PyQt6 Widget | Implementation Detail |
| :--- | :--- | :--- |
| **Image Canvas** | `QOpenGLWidget` | GL-Texture rendering for performance |
| **Header Stats** | `QHBoxLayout` + `QLabel` | Dynamic text: "Res \| CS (Source)" |
| **Status Area** | `QStatusBar` + `QProgressBar` | Non-blocking export tracking |
| **Toasts** | `QGraphicsProxyWidget` Overlay | Floating notifications over Canvas |

---

## 3. Core Architecture: "The Controller Pattern"

### State Management (Decoupled from Streamlit)
We will implement a `SessionManager` that replaces `st.session_state`.
```python
@dataclass
class AppState:
    current_file_path: Optional[str] = None
    config: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    workspace_cs: str = "Adobe RGB"
    is_processing: bool = False
    active_tool: ToolMode = ToolMode.NONE
```

### Signal Orchestration
The `AppController` will act as the central dispatcher:
1.  **UI Signal**: `Sidebar.exposure_changed(1.5)`
2.  **Controller**: Updates `AppState.config.exposure.density = 1.5`
3.  **Controller**: Dispatches `RenderTask(AppState.config)` to `ProcessingWorker`.
4.  **Worker (Background Thread)**: Executes `DarkroomEngine.process()`.
5.  **Worker**: Emits `resultReady(QImage)`.
6.  **Controller**: Tells `CanvasWidget.updateImage(new_qimage)`.

---

## 4. Deep-Dive: Interactive Geometry Tools

### I. The Mathematical Bridge
The existing `src/services/view/coordinate_mapping.py` is 100% reusable. The `map_click_to_raw` logic remains the core.
*   **Input**: Native `QMouseEvent` provides `QPointF` (sub-pixel precision).
*   **Normalization**: Map `event.pos()` -> `0..1` range based on the current `CanvasWidget` dimensions.
*   **Inversion**: Apply the existing `uv_grid` logic to reverse rotations/flips and find the raw sensor coordinates.

### II. Tool Implementation Specifics
| Tool | Interaction Logic | Visual Feedback |
| :--- | :--- | :--- |
| **Dust Removal** | Single click emits `(rx, ry)` to `AppState`. | Instant `QPainter.drawEllipse` at click site. |
| **WB Picker** | Single click samples pixel under cursor. | Tooltip showing `M, Y` shifts before clicking. |
| **Manual Crop** | Click-drag "Rubber Band" interaction. | Real-time `QRubberBand` or custom dashed rect. |
| **Dodge & Burn** | Real-time brush strokes on `mouseMove`. | Transparent overlay with localized alpha mask. |

### III. Overlays (Masks & Dust Patches)
*   **Static Layer**: The background processed image.
*   **Dynamic Overlay**: A `QPixmap` or `QPainter` layer that draws circles/paths. This allows the user to see exactly where they are working without re-running the full pixel engine for every mouse movement.

---

## 5. ICC Profile Management
*   **Logic**: Use `src/infrastructure/display/color_spaces.py`.
*   **PyQt6**:
    ```python
    icc_path = registry.get_icc_path(current_cs)
    q_color_space = QColorSpace.fromIccProfile(icc_path)
    q_image.setColorSpace(q_color_space)
    ```
    This tagging ensures the OS manages the monitor gamut correctly (no sRGB clipping).

---

## 6. Migration Tasks Checklist

### Phase 1: Infrastructure (Pure Python)
- [x] Refactor `AppController` to remove all `import streamlit`.
- [x] Create `DesktopSessionManager` to handle SQLite persistence via native signals.
- [x] Port `ColorSpaceRegistry` -> `QColorSpace` mapping.

### Phase 2: Custom Widgets
- [x] `ImageCanvas`: Native `mousePressEvent`/`mouseMoveEvent` for tool interaction.
- [x] `SignalSlider`: A custom slider that emits floating-point values and supports debouncing.
- [x] `AssetDelegate`: Custom rendering for the filmstrip thumbnails. (Handled via QListView + Model)

### Phase 3: Layout & Polish
- [x] Implement the "Modern Dark" stylesheet (QSS).
- [x] Replicate the 2-row toolbar below the image for navigation and geometry actions.
- [x] Integrate the existing `VersionChecker` into an "About" dialog or status alert. (Handled via Status Bar)

---

## 7. Directory Structure (Final Parity)
```text
src/
├── desktop/
│   ├── main.py             # QApplication loop
│   ├── controller.py       # Event orchestration & Task dispatch
│   ├── session.py          # State persistence & AppState management
│   ├── view/
│   │   ├── main_window.py  # QMainWindow + Layout
│   │   ├── canvas/
│   │   │   ├── widget.py   # Image rendering + Event handling
│   │   │   └── tools.py    # QPainter logic for brushes/crop-box
│   │   ├── sidebar/        # Parity with src/ui/components/sidebar/
│   │   └── styles/         # QSS (CSS-like) themes
│   └── workers/
│       ├── render.py       # QThread for Engine.process
│       └── export.py       # QThread for Batch Export
└── ...                     # Shared services/infrastructure
```

---

## 8. Usage
The application is now a fully-featured, native desktop digital darkroom.

### How to Run:
From the project root:
```bash
.venv/bin/python desktop.py
```
Or as a module:
```bash
.venv/bin/python -m src.desktop.main
```
