# Finish Feature: Vignette Effect

Date: 2026-05-03

## Summary

Add a new `finish` feature module that applies a vignette overlay as the final creative stage in the pipeline. Follows the existing multi-stage architecture (config → logic → processor → CPU/GPU engines).

## Motivation

The lab feature currently handles color/detail/sharpening/glow/halation effects. A vignette is a creative overlay applied after cropping — it doesn't belong in lab or toning. Lightroom places vignette in its own "Effects" panel as a post-crop adjustment. We mirror that separation.

## Architecture

### New module: `negpy/features/finish/`

```
negpy/features/finish/
├── __init__.py
├── models.py       # FinishConfig frozen dataclass
├── logic.py        # apply_vignette() pure numpy function
└── processor.py    # FinishProcessor thin wrapper
```

### Config (`models.py`)

```python
@dataclass(frozen=True)
class FinishConfig:
    vignette_strength: float = 0.0   # [-1.0, 1.0], 0 = off, negative = darken, positive = brighten
    vignette_size: float = 0.5       # [0.0, 1.0], midpoint of falloff gradient
```

### Logic (`logic.py`)

```python
def apply_vignette(img: ImageBuffer, strength: float, size: float) -> ImageBuffer:
```

Algorithm:
1. Compute per-pixel Euclidean distance from image center, normalized to [0, 1] (corner = 1.0)
2. Rescale distance: `t = (d - size) / max(1e-6, 1.0 - size)` — so t=0 at the midpoint and t=1 at the farthest edge
3. Clamp t to [0, 1] so pixels inside the midpoint radius are unaffected
4. Falloff: `factor = 0.5 * (1 - cos(t * π))` — smooth S-curve transition
5. If strength < 0 (darken): `result = color * (1 - factor * |strength|)`
6. If strength > 0 (brighten): `result = color + (1 - color) * factor * strength`
7. strength == 0: return input unchanged (fast path)

### Processor (`processor.py`)

```python
class FinishProcessor:
    def __init__(self, config: FinishConfig):
        self.config = config

    def process(self, image: ImageBuffer, context: PipelineContext) -> ImageBuffer:
        if self.config.vignette_strength == 0.0:
            return image
        return apply_vignette(image, self.config.vignette_strength, self.config.vignette_size)
```

### Workspace integration

Add `finish: FinishConfig = field(default_factory=FinishConfig)` to `WorkspaceConfig` in `negpy/domain/models.py`. The `to_dict()` and `from_flat_dict()` already iterate all fields — no changes needed there.

### CPU pipeline (`engine.py`)

Insert after `CropProcessor` (line 144):

```python
# After: current_img = CropProcessor(settings.geometry).process(current_img, context)
# Add:
current_img = FinishProcessor(settings.finish).process(current_img, context)
```

No caching stage — vignette is fast enough to recompute each frame. (If performance becomes an issue later, it can be wrapped in `_run_stage`.)

### GPU pipeline (`gpu_engine.py`)

Add a `finish` WGSL compute shader that runs between toning and layout. The shader reads the toned/cropped content texture and writes back to the same dimensions.

Shader: `negpy/features/finish/shaders/finish.wgsl`

- Input: `texture_2d<f32>` (content at full resolution)
- Output: `texture_storage_2d<rgba32float, write>`
- Uniforms: `vignette_strength: f32`, `vignette_size: f32`
- Dispatch: full content dimensions, 8×8 workgroups

GPU engine changes:
1. Add shader path to `self._shaders`
2. Add uniform slot to `_uniform_names` (appended after "layout")
3. Add uniform packing in `_upload_unified_uniforms`
4. Add texture for finish stage: same dimensions as `tex_toning`
5. Add dispatch pass between toning and layout
6. Update `_detect_invalidated_stage` to re-run finish when `settings.finish` changes

### Sidebar (`negpy/desktop/view/sidebar/lab.py`)

Add vignette sliders under the existing "EFFECTS" section header:

```python
# After the glow/halation row, add:
self.layout.addWidget(section_subheader("VIGNETTE"))

self.vignette_strength_slider = CompactSlider("Strength", -1.0, 1.0, conf.vignette_strength)
self.vignette_size_slider = CompactSlider("Size", 0.0, 1.0, conf.vignette_size)
```

Signal wiring follows the same `valueChanged`/`valueCommitted` pattern with `update_config_section("finish", ...)`.

## Data flow

```
User drags slider
  → AppController.update_config_section("finish", vignette_strength=v)
    → dataclasses.replace(config, finish=replace(config.finish, vignette_strength=v))
    → emit config_updated
    → worker renders via DarkroomEngine or GPUEngine
      → FinishProcessor.process(img, context)
        → apply_vignette(img, strength, size)
    → emit image_updated
```

## Testing

- Unit tests for `apply_vignette()`: verify identity at strength=0, darkening at corners, brightening at corners, midpoint behavior
- Integration: verify `FinishProcessor` passes through when strength=0, applies effect otherwise
- Manual: visual check against sample negatives

## Out of scope

- Moving border from `ExportConfig` to `FinishConfig`. Border is wired into the GPU layout pass and export settings. This is a larger refactor.
- Additional vignette shapes (rectangular, elliptical). The radial vignette is the standard.
- Vignette feather/roundness controls. `size` provides effective midpoint control which is the primary creative parameter.
