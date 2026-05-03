# Finish Feature: Vignette Effect — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a vignette overlay effect as a new `finish` feature stage applied after crop.

**Architecture:** New `negpy/features/finish/` module (models, logic, processor) following the established feature pattern. Wired into both CPU and GPU pipelines as the final creative stage. Sidebar sliders in the existing lab sidebar under EFFECTS.

**Tech Stack:** Python 3.12+, numpy, numba, cv2, dataclasses, WGSL, PyQt6

---

### Task 1: FinishConfig model

**Files:**
- Create: `negpy/features/finish/__init__.py`
- Create: `negpy/features/finish/models.py`

- [ ] **Step 1: Create `__init__.py`**

```bash
touch negpy/features/finish/__init__.py
```

- [ ] **Step 2: Write `models.py`**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FinishConfig:
    """
    Post-crop finishing effects (vignette).
    """

    vignette_strength: float = 0.0  # [-1.0, 1.0]  0 = off, neg = darken, pos = brighten
    vignette_size: float = 0.5      # [0.0, 1.0]   midpoint of falloff gradient
```

- [ ] **Step 3: Commit**

```bash
git add negpy/features/finish/__init__.py negpy/features/finish/models.py
git commit -m "feat: add FinishConfig frozen dataclass"
```

---

### Task 2: Vignette logic (numpy)

**Files:**
- Create: `negpy/features/finish/logic.py`
- Create: `tests/test_finish_logic.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest
import numpy as np
from negpy.features.finish.logic import apply_vignette


class TestVignette(unittest.TestCase):
    def _gradient_image(self) -> np.ndarray:
        """100x100 mid-gray image for reliable vignette testing."""
        return np.full((100, 100, 3), 0.5, dtype=np.float32)

    def test_noop_when_strength_zero(self) -> None:
        """Strength 0.0 returns image unchanged."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=0.0, size=0.5)
        np.testing.assert_array_equal(res, img)

    def test_output_shape_and_range(self) -> None:
        """Output keeps same shape and stays in [0, 1]."""
        img = self._gradient_image()
        for strength in [-0.5, 0.5, -1.0, 1.0]:
            for size in [0.0, 0.5, 1.0]:
                res = apply_vignette(img, strength, size)
                self.assertEqual(res.shape, img.shape)
                self.assertGreaterEqual(float(res.min()), 0.0)
                self.assertLessEqual(float(res.max()), 1.0)

    def test_darken_corners_darker_than_center(self) -> None:
        """Negative strength darkens corners more than center."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=-1.0, size=0.5)
        # Corner pixel (0,0) should be darker than center (50,50)
        corner_luma = float(res[0, 0].mean())
        center_luma = float(res[50, 50].mean())
        self.assertLess(corner_luma, center_luma)

    def test_brighten_corners_brighter_than_center(self) -> None:
        """Positive strength brightens corners more than center."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=1.0, size=0.5)
        corner_luma = float(res[0, 0].mean())
        center_luma = float(res[50, 50].mean())
        self.assertGreater(corner_luma, center_luma)

    def test_center_unaffected(self) -> None:
        """Center pixel should be unchanged regardless of strength."""
        img = self._gradient_image()
        for strength in [-1.0, -0.5, 0.5, 1.0]:
            res = apply_vignette(img, strength, size=0.5)
            np.testing.assert_array_almost_equal(res[50, 50], img[50, 50], decimal=5)

    def test_size_zero_affects_entire_image(self) -> None:
        """Size=0 means falloff starts from center — everything affected."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=-1.0, size=0.0)
        # Center should be darkened too
        center_luma = float(res[50, 50].mean())
        self.assertLess(center_luma, 0.5)

    def test_size_one_only_affects_extreme_corners(self) -> None:
        """Size=1 means falloff starts at edges — most pixels unaffected."""
        img = self._gradient_image()
        res = apply_vignette(img, strength=-1.0, size=1.0)
        # Most of the image should be near 0.5
        center_luma = float(res[50, 50].mean())
        self.assertAlmostEqual(center_luma, 0.5, delta=0.01)
        # Extreme corner should still be darkened
        corner_luma = float(res[0, 0].mean())
        self.assertLess(corner_luma, center_luma)

    def test_non_square_image(self) -> None:
        """Works correctly on non-square images."""
        img = np.full((50, 200, 3), 0.5, dtype=np.float32)
        res = apply_vignette(img, strength=-1.0, size=0.5)
        self.assertEqual(res.shape, img.shape)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_finish_logic.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'negpy.features.finish.logic'`

- [ ] **Step 3: Write `logic.py`**

```python
import numpy as np
from negpy.domain.types import ImageBuffer
from negpy.kernel.image.validation import ensure_image


def apply_vignette(img: ImageBuffer, strength: float, size: float) -> ImageBuffer:
    """
    Radial vignette overlay using cosine falloff.

    Args:
        img: Float32 RGB image [0, 1].
        strength: [-1, 1]. Negative = darken edges, positive = brighten edges, 0 = no effect.
        size: [0, 1]. Midpoint of gradient falloff. 0 = starts at center, 1 = starts at edges.

    Returns:
        Modified ImageBuffer with vignette applied.
    """
    if strength == 0.0:
        return img

    h, w = img.shape[:2]
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5

    # Euclidean distance from center, normalized so corners = 1.0
    y_coords = np.arange(h, dtype=np.float32)
    x_coords = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    dy = (yy - cy) / max(cy, 1.0)
    dx = (xx - cx) / max(cx, 1.0)
    dist = np.sqrt(dx**2 + dy**2)  # range [0, 1]

    # Remap: t=0 at the midpoint (size), t=1 at farthest edge
    t = (dist - size) / max(1.0 - size, 1e-6)
    t = np.clip(t, 0.0, 1.0)

    # Smooth cosine falloff
    factor = 0.5 * (1.0 - np.cos(t * np.pi))

    strength_abs = abs(strength)

    if strength < 0.0:
        # Darken: multiply toward black
        result = img * (1.0 - factor[:, :, np.newaxis] * strength_abs)
    else:
        # Brighten: blend toward white
        result = img + (1.0 - img) * factor[:, :, np.newaxis] * strength_abs

    return ensure_image(np.clip(result, 0.0, 1.0))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_finish_logic.py -v
```
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add negpy/features/finish/logic.py tests/test_finish_logic.py
git commit -m "feat: add vignette logic with cosine radial falloff"
```

---

### Task 3: FinishProcessor

**Files:**
- Create: `negpy/features/finish/processor.py`
- Create: `tests/test_finish_processor.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
import numpy as np
from negpy.domain.interfaces import PipelineContext
from negpy.features.finish.models import FinishConfig
from negpy.features.finish.processor import FinishProcessor


class TestFinishProcessor(unittest.TestCase):
    def _gradient_image(self) -> np.ndarray:
        return np.full((100, 100, 3), 0.5, dtype=np.float32)

    def _context(self) -> PipelineContext:
        return PipelineContext(original_size=(100, 100), scale_factor=1.0, process_mode="C41")

    def test_noop_when_strength_zero(self) -> None:
        """Processor returns image unchanged when vignette strength is 0."""
        img = self._gradient_image()
        config = FinishConfig(vignette_strength=0.0, vignette_size=0.5)
        processor = FinishProcessor(config)
        ctx = self._context()
        res = processor.process(img, ctx)
        np.testing.assert_array_equal(res, img)

    def test_applies_effect_when_nonzero(self) -> None:
        """Processor darkens corners when strength is negative."""
        img = self._gradient_image()
        config = FinishConfig(vignette_strength=-1.0, vignette_size=0.5)
        processor = FinishProcessor(config)
        ctx = self._context()
        res = processor.process(img, ctx)
        # Corner should be darker than center
        self.assertLess(float(res[0, 0].mean()), float(res[50, 50].mean()))

    def test_preserves_image_type(self) -> None:
        """Output is float32 in [0, 1]."""
        img = self._gradient_image()
        config = FinishConfig(vignette_strength=0.5, vignette_size=0.5)
        processor = FinishProcessor(config)
        ctx = self._context()
        res = processor.process(img, ctx)
        self.assertEqual(res.dtype, np.float32)
        self.assertGreaterEqual(float(res.min()), 0.0)
        self.assertLessEqual(float(res.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_finish_processor.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'negpy.features.finish.processor'`

- [ ] **Step 3: Write `processor.py`**

```python
from negpy.domain.interfaces import PipelineContext
from negpy.domain.types import ImageBuffer
from negpy.features.finish.logic import apply_vignette
from negpy.features.finish.models import FinishConfig


class FinishProcessor:
    def __init__(self, config: FinishConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        """
        Apply vignette overlay to the image.
        """
        if self.config.vignette_strength == 0.0:
            return image
        return apply_vignette(image, self.config.vignette_strength, self.config.vignette_size)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_finish_processor.py -v
```
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add negpy/features/finish/processor.py tests/test_finish_processor.py
git commit -m "feat: add FinishProcessor as thin wrapper for vignette"
```

---

### Task 4: Wire FinishConfig into WorkspaceConfig

**Files:**
- Modify: `negpy/domain/models.py`

- [ ] **Step 1: Add import and field to WorkspaceConfig**

```python
# At top of file, add import (after existing feature imports):
from negpy.features.finish.models import FinishConfig

# In WorkspaceConfig dataclass, add new field (after 'toning'):
    finish: FinishConfig = field(default_factory=FinishConfig)
```

Run a quick import check:

```bash
uv run python -c "from negpy.domain.models import WorkspaceConfig; c = WorkspaceConfig(); print(c.finish)"
```
Expected: `FinishConfig(vignette_strength=0.0, vignette_size=0.5)`

- [ ] **Step 2: Run existing tests to check for regressions**

```bash
uv run pytest tests/test_lab_logic.py tests/test_exposure_logic.py -v -x
```
Expected: all existing tests PASS (no breakage)

- [ ] **Step 3: Commit**

```bash
git add negpy/domain/models.py
git commit -m "feat: add FinishConfig field to WorkspaceConfig"
```

---

### Task 5: Wire FinishProcessor into CPU engine

**Files:**
- Modify: `negpy/services/rendering/engine.py`

- [ ] **Step 1: Add import and insert processor stage**

At the top of `engine.py`, add the import:
```python
from negpy.features.finish.processor import FinishProcessor
```

In `process()`, after the CropProcessor line (currently line 144):
```python
current_img = CropProcessor(settings.geometry).process(current_img, context)
```
Add:
```python
current_img = FinishProcessor(settings.finish).process(current_img, context)
```

The modified section will read:
```python
current_img = CropProcessor(settings.geometry).process(current_img, context)
current_img = FinishProcessor(settings.finish).process(current_img, context)
```

- [ ] **Step 2: Verify import and syntax**

```bash
uv run python -c "from negpy.services.rendering.engine import DarkroomEngine; print('OK')"
```
Expected: `OK` (no import errors)

- [ ] **Step 3: Run engine-related tests**

```bash
uv run pytest tests/ -v -x -k "not test_gpu and not gpu" --ignore=tests/test_finish_logic.py --ignore=tests/test_finish_processor.py
```
Expected: existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add negpy/services/rendering/engine.py
git commit -m "feat: add FinishProcessor stage after CropProcessor in CPU engine"
```

---

### Task 6: GPU WGSL shader for vignette

**Files:**
- Create: `negpy/features/finish/shaders/__init__.py`
- Create: `negpy/features/finish/shaders/finish.wgsl`

- [ ] **Step 1: Create shader directory and init**

```bash
mkdir -p negpy/features/finish/shaders
touch negpy/features/finish/shaders/__init__.py
```

- [ ] **Step 2: Write the WGSL shader**

```wgsl
struct FinishUniforms {
    vignette_strength: f32,
    vignette_size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: FinishUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0).rgb;

    // Center and max distance
    let center = vec2<f32>(f32(dims.x) * 0.5, f32(dims.y) * 0.5);
    let max_dist = length(center);
    let px = vec2<f32>(f32(coords.x), f32(coords.y));
    let d = length(px - center) / max_dist;

    // Remap: t=0 at midpoint, t=1 at farthest edge
    let t = clamp((d - params.vignette_size) / max(1e-6, 1.0 - params.vignette_size), 0.0, 1.0);

    // Smooth cosine falloff
    let factor = 0.5 * (1.0 - cos(t * 3.14159265));

    let strength_abs = abs(params.vignette_strength);
    if (params.vignette_strength < 0.0) {
        color = color * (1.0 - factor * strength_abs);
    } else if (params.vignette_strength > 0.0) {
        color = color + (1.0 - color) * factor * strength_abs;
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
```

- [ ] **Step 3: Commit**

```bash
git add negpy/features/finish/shaders/__init__.py negpy/features/finish/shaders/finish.wgsl
git commit -m "feat: add GPU vignette shader (WGSL compute)"
```

---

### Task 7: Wire vignette into GPU engine

**Files:**
- Modify: `negpy/services/rendering/gpu_engine.py`

This is the most invasive change. Four locations in `gpu_engine.py` are modified:

**7a. Add shader path registration** (in `__init__`):

- [ ] **Step 1: Add `finish` entry to `self._shaders`**

```python
# In __init__, add after "toning" line and before "metrics":
"finish": get_resource_path(os.path.join("negpy", "features", "finish", "shaders", "finish.wgsl")),
```

**7b. Add uniform slot** (in `__init__`):

- [ ] **Step 2: Add `finish` to `_uniform_names`**

```python
# Add "finish" after "toning" and before "layout":
self._uniform_names = [
    "geometry",
    "normalization",
    "exposure",
    "clahe_u",
    "retouch_u",
    "lab",
    "toning",
    "finish",
    "layout",
]
```

**7c. Add uniform size** (in `_get_uniform_binding`):

- [ ] **Step 3: Add `finish` entry in sizes dict**

```python
sizes = {
    "geometry": 32,
    "normalization": 112,
    "exposure": 160,
    "clahe_u": 32,
    "retouch_u": 40,
    "lab": 96,
    "toning": 64,
    "finish": 32,
    "layout": 48,
}
```

**7d. Add finish uniform packing** (in `_upload_unified_uniforms`):

- [ ] **Step 4: Pack finish uniforms**

After the toning packing section (after `t_data` is defined), insert:

```python
f_data = (
    struct.pack("ff", float(settings.finish.vignette_strength), float(settings.finish.vignette_size))
    + b"\x00" * 24
)
```

Then update the buffer packing loop:
```python
# OLD:
full_buffer = bytearray()
for d in [g_data, n_data, e_data, c_data, r_u_data, l_data, t_data, y_data]:
    ...
# NEW:
full_buffer = bytearray()
for d in [g_data, n_data, e_data, c_data, r_u_data, l_data, t_data, f_data, y_data]:
    ...
```

**7e. Update `_detect_invalidated_stage`** (return indices shift):

- [ ] **Step 5: Shift stage indices**

Current code (lines 101-124):
```python
if last.toning != settings.toning:
    return 5
if last.export != settings.export:
    return 6
return 7  # Nothing changed
```

Change to:
```python
if last.toning != settings.toning:
    return 5
if last.finish != settings.finish:
    return 6
if last.export != settings.export:
    return 7
return 8  # Nothing changed
```

**7f. Insert finish dispatch pass** (in `process_to_texture`):

- [ ] **Step 6: Add finish texture and dispatch between toning and layout**

After the toning dispatch (after line 520, before the layout section), insert:

```python
# --- Finish (Vignette) ---
tex_finish = self._get_intermediate_texture(
    crop_w,
    crop_h,
    wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.TEXTURE_BINDING,
    "finish_tex",
)
if start_stage <= 6:
    self._dispatch_pass(
        enc,
        "finish",
        [
            (0, tex_toning.view),
            (1, tex_finish.view),
            (2, self._get_uniform_binding("finish")),
        ],
        crop_w,
        crop_h,
    )
    tex_for_layout = tex_finish
else:
    tex_for_layout = tex_toning
```

Then in the layout section (currently line 522-541), change:
```python
# OLD:
        tex_final = self._get_intermediate_texture(
            paper_w, paper_h, ..., "final",
        )
        if start_stage <= 6:
            self._dispatch_pass(
                enc, "layout",
                [
                    (0, tex_toning.view),
                    ...
```
Replace `tex_toning.view` with `tex_for_layout.view`:
```python
# NEW:
        tex_final = self._get_intermediate_texture(
            paper_w, paper_h, ..., "final",
        )
        if start_stage <= 7:
            self._dispatch_pass(
                enc, "layout",
                [
                    (0, tex_for_layout.view),
                    ...
```

Also, the `content_rect` assignment for the non-layout path (currently line 544):
```python
# OLD:
    tex_final, content_rect = tex_toning, (0, 0, crop_w, crop_h)
# NEW:
    tex_final, content_rect = tex_for_layout, (0, 0, crop_w, crop_h)
```

- [ ] **Step 7: Verify import and syntax**

```bash
uv run python -c "from negpy.services.rendering.gpu_engine import GPUEngine; print('OK')"
```
Expected: `OK` (no import errors)

- [ ] **Step 8: Run existing tests to check for regressions**

```bash
uv run pytest tests/ -v -x --ignore=tests/test_finish_logic.py --ignore=tests/test_finish_processor.py -k "not gpu"
```
Expected: all non-GPU tests PASS

- [ ] **Step 9: Commit**

```bash
git add negpy/services/rendering/gpu_engine.py
git commit -m "feat: add vignette finish stage to GPU pipeline"
```

---

### Task 8: Add vignette sliders to Lab sidebar

**Files:**
- Modify: `negpy/desktop/view/sidebar/lab.py`

- [ ] **Step 1: Add vignette sliders under "EFFECTS"**

After the glow/halation row (line 53), add vignette row:

```python
# After: self.layout.addLayout(row4)
# Add:
self.layout.addWidget(section_subheader("VIGNETTE"))

row5 = QHBoxLayout()
self.vignette_strength_slider = CompactSlider("Strength", -1.0, 1.0, conf.vignette_strength)
self.vignette_size_slider = CompactSlider("Size", 0.0, 1.0, conf.vignette_size)
row5.addWidget(self.vignette_strength_slider)
row5.addWidget(self.vignette_size_slider)
self.layout.addLayout(row5)
```

- [ ] **Step 2: Wire signal connections**

In `_connect_signals()`, add after the halation slider connections (after line 112):

```python
self.vignette_strength_slider.valueChanged.connect(
    lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, vignette_strength=v)
)
self.vignette_strength_slider.valueCommitted.connect(
    lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, vignette_strength=v)
)

self.vignette_size_slider.valueChanged.connect(
    lambda v: self.update_config_section("finish", persist=False, readback_metrics=False, vignette_size=v)
)
self.vignette_size_slider.valueCommitted.connect(
    lambda v: self.update_config_section("finish", persist=True, readback_metrics=True, vignette_size=v)
)
```

- [ ] **Step 3: Wire sync_ui**

In `sync_ui()`, after the halation slider setValue call (line 127), add:

```python
conf_finish = self.state.config.finish
self.vignette_strength_slider.setValue(conf_finish.vignette_strength)
self.vignette_size_slider.setValue(conf_finish.vignette_size)
```

- [ ] **Step 4: Update `block_signals` list**

In `block_signals()`, add the new sliders to the widgets list:

```python
widgets = [
    ...,
    self.vignette_strength_slider,
    self.vignette_size_slider,
]
```

- [ ] **Step 5: Add tooltips** (in `controls_panel.py`'s `apply_shortcut_tooltips`)

In `negpy/desktop/view/sidebar/controls_panel.py`, after the halation tooltip (line 193), add:

```python
lab.vignette_strength_slider.setToolTip(
    tooltip_with_shortcut("Vignette strength: negative = darken edges, positive = brighten edges", ["vignette_str_inc", "vignette_str_dec"])
)
lab.vignette_size_slider.setToolTip(
    tooltip_with_shortcut("Vignette size: how far the vignette extends from center", ["vignette_size_inc", "vignette_size_dec"])
)
```

- [ ] **Step 6: Verify Python syntax**

```bash
uv run python -c "from negpy.desktop.view.sidebar.lab import LabSidebar; print('OK')"
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add negpy/desktop/view/sidebar/lab.py negpy/desktop/view/sidebar/controls_panel.py
git commit -m "feat: add vignette strength and size sliders to Lab sidebar"
```

---

### Task 9: Session reset and ControlsPanel integration

**Files:**
- Modify: `negpy/desktop/session.py`
- Modify: `negpy/desktop/view/sidebar/controls_panel.py`

- [ ] **Step 1: Add `finish` to `reset_section` in session.py**

In `reset_section()`, add the FinishConfig import and entry:

```python
from negpy.features.finish.models import FinishConfig

defaults = {
    ...,
    "retouch": RetouchConfig(),
    "finish": FinishConfig(),
}
```

- [ ] **Step 2: Add reset wiring in controls_panel.py**

Add in `_connect_signals()` after the retouch reset line (line 154):
```python
self.lab_section.reset_requested.connect(lambda: self.controller.session.reset_section("finish"))
```

Note: Since vignette sliders live in the lab sidebar, the reset button on the lab section will also reset the finish config. If this is not desired (user wants separate reset), we would need a separate FinishSidebar with its own section. For now, resetting lab resets finish as well, which is reasonable since they share the sidebar.

- [ ] **Step 3: Commit**

```bash
git add negpy/desktop/session.py negpy/desktop/view/sidebar/controls_panel.py
git commit -m "feat: add finish section reset support"
```

---

### Task 10: Final verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v -x -k "not gpu" --ignore=tests/test_finish_logic.py --ignore=tests/test_finish_processor.py
uv run pytest tests/test_finish_logic.py tests/test_finish_processor.py -v
```
Expected: ALL tests PASS

- [ ] **Step 2: Run lint + type check**

```bash
make lint
make type
```
Expected: no new errors introduced (fix any that appear)

- [ ] **Step 3: Run the app (manual smoke test)**

```bash
make run
```
Verify:
- [ ] Lab sidebar has "VIGNETTE" section with Strength and Size sliders
- [ ] Dragging Strength to negative darkens edges
- [ ] Dragging Strength to positive brightens edges  
- [ ] Size slider controls the midpoint radius
- [ ] No crashes on image load or slider drag

---

### File map (summary)

```
negpy/features/finish/
├── __init__.py                        (created)
├── models.py                          (created)
├── logic.py                           (created)
├── processor.py                       (created)
└── shaders/
    ├── __init__.py                    (created)
    └── finish.wgsl                    (created)

negpy/domain/models.py                 (modified: +import, +field)
negpy/services/rendering/engine.py     (modified: +import, +stage call)
negpy/services/rendering/gpu_engine.py (modified: +shader, +uniform, +dispatch)
negpy/desktop/view/sidebar/lab.py      (modified: +sliders, +signals, +sync)
negpy/desktop/view/sidebar/controls_panel.py (modified: +tooltips, +reset wiring)
negpy/desktop/session.py               (modified: +finish in reset_section)

tests/
├── test_finish_logic.py               (created)
└── test_finish_processor.py           (created)
```
