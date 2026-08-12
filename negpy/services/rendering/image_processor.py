import os
import io
import ctypes
import cv2
import rawpy
import tifffile
import imagecodecs
import numpy as np
from dataclasses import replace as dc_replace
from functools import lru_cache
from PIL import Image, ImageCms
from typing import Callable, Tuple, Optional, Any, Dict, List, Sequence
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.config import APP_CONFIG
from negpy.domain.types import ImageBuffer
from negpy.domain.models import (
    WorkspaceConfig,
    ExportConfig,
    ExportFormat,
    ExportResolutionMode,
    ColorSpace,
)
from negpy.features.altprocess.models import AltProcess
from negpy.features.process.models import ProcessMode
from negpy.features.process.logic import effective_linear_raw, linear_raw_token
from negpy.features.process.sensor import apply_sensor_correction, effective_sensor_matrix, sensor_token
from negpy.features.exposure.models import RenderIntent
from negpy.features.flatfield.logic import apply_flatfield, flatfield_token
from negpy.features.retouch.logic import (
    apply_hair_inpaint,
    apply_ir_attenuation,
    apply_score_repair,
    compute_dust_stats,
    detect_luma_score,
    film_scale,
    downsample_ir,
    hair_bake_token,
    ir_bake_token,
    ir_defect_score,
    ir_detect_cutoff,
    ir_detect_target,
    ir_ratio_and_gain,
    lines_to_score,
    manual_bake_token,
    repair_components,
    route_wide_defects,
    strokes_to_score,
)
from negpy.features.retouch import openice
from negpy.features.retouch.models import IR_METHOD_OPENICE
from negpy.features.rgbscan.logic import merge_rgb_triplet, rgbscan_token
from negpy.features.rgbscan.models import RgbScanConfig, is_rgb_triplet
from negpy.features.stitch.logic import stitch_composite
from negpy.features.hdr.logic import merge_bracket
from negpy.features.hdr.models import hdr_active, hdr_token
from negpy.features.stitch.models import stitch_has_triplets, stitch_token
from negpy.domain.interfaces import PipelineContext
from negpy.services.rendering.engine import DarkroomEngine
from negpy.services.rendering.gpu_engine import GPUEngine
from negpy.infrastructure.gpu.device import GPUDevice
from negpy.kernel.image.logic import (
    apply_exif_orientation,
    float_to_uint8,
    float_to_uint16,
    ensure_rgb,
    uint16_to_float32,
    float_to_uint_luma,
    working_oetf_decode,
)
from negpy.infrastructure.loaders.factory import loader_factory
from negpy.infrastructure.loaders.helpers import (
    NonStandardFileWrapper,
    camera_wb_multipliers,
    camera_xyz_matrix,
    get_best_demosaic_algorithm,
    is_xtrans,
)
from negpy.services.export.print import PrintService
from negpy.infrastructure.display.color_spaces import ColorSpaceRegistry, WORKING_COLOR_SPACE
from negpy.infrastructure.display.icc_lut import DEFAULT_LUT_SIZE, apply_icc_u16_rgb

# Preview soft-proof LUT grid. Finer than the display LUT because the proof clips at
# the output gamut boundary, and interpolating across that kink is where the error is.
PROOF_LUT_SIZE = 65

logger = get_logger(__name__)

# (photometric, primaries, transfer) for JXL's enumerated color encoding (D65 white
# only, no ICC). Other spaces must hard-fail. Transfers verified against the bundled
# icc/*.icc: Rec 2020 uses the Rec.709/BT.2020 OETF (BT709, not sRGB) and
# GrayGamma2.2.icc holds the sRGB TRC despite its name (SRGB, not gamma 2.2).
_JXL_COLOR = {
    ColorSpace.SRGB.value: ("RGB", "SRGB", "SRGB"),
    ColorSpace.P3_D65.value: ("RGB", "P3", "SRGB"),
    ColorSpace.REC2020.value: ("RGB", "BT2100", "BT709"),
    ColorSpace.GREYSCALE.value: ("GRAY", None, "SRGB"),
}


def _use_half_size_decode(raw: Any, linear_raw: bool) -> bool:
    """Mirrors the preview fast path (PreviewManager): rawpy half_size aliases the
    X-Trans 6x6 CFA on linear (no-camera-WB) decodes, so those stay full-res."""
    return not isinstance(raw, NonStandardFileWrapper) and not (is_xtrans(raw) and linear_raw)


def _downsample_to_long_edge(buf: np.ndarray, long_px: int) -> np.ndarray:
    """Shrink so the long edge is at most long_px; never upscales."""
    h, w = buf.shape[:2]
    long_edge = max(h, w)
    if long_px <= 0 or long_edge <= long_px:
        return buf
    s = long_px / long_edge
    return cv2.resize(buf, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def _detection_downsample(buf: np.ndarray) -> np.ndarray:
    """Dust detection always runs at preview scale so preview and export produce
    the identical region set (WYSIWYG) and full-res export detection stays cheap."""
    h, w = buf.shape[:2]
    long_edge = max(h, w)
    if long_edge <= APP_CONFIG.preview_render_size:
        return buf
    s = APP_CONFIG.preview_render_size / long_edge
    return cv2.resize(buf, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def _part_params(params: WorkspaceConfig, index: int) -> WorkspaceConfig:
    """Params for stitch part ``index``, carrying that part's own R/G/B exposures.

    Empty ``stitch_triplets`` (composites registered before triplet support) falls back
    to ``params.rgbscan``."""
    triplets = params.stitch.stitch_triplets
    if index >= len(triplets):
        return params
    green, blue = triplets[index]
    rgbscan = (
        RgbScanConfig(enabled=True, green_path=green, blue_path=blue, align=params.stitch.stitch_align)
        if (green and blue)
        else RgbScanConfig()
    )
    return dc_replace(params, rgbscan=rgbscan)


class ImageProcessor:
    """
    Coordinates multi-backend image processing.
    Seamlessly switches between CPU (DarkroomEngine) and GPU (GPUEngine).
    """

    def __init__(self) -> None:
        self.engine_cpu = DarkroomEngine()
        self.engine_gpu: Optional[GPUEngine] = None

        # Last decoded full-res source (decode+flatfield) — skips re-decode on repeat export.
        # One entry only (full-res buffers are large); treated read-only downstream.
        self._source_cache_key: Optional[tuple] = None
        self._source_cache_value: Optional[Tuple[np.ndarray, Optional[np.ndarray], str]] = None
        # Decoder XYZ->camera matrix per source path, filled during decode. Small and
        # append-only: one 3x3 per file the session has exported.
        self._cam_xyz_by_path: Dict[str, Tuple[Optional[list], Optional[list]]] = {}

        # Flat-field + sensor-unmix corrected source: full-buffer passes that no
        # creative slider moves.
        self._precorrect_key: Optional[tuple] = None
        self._precorrect_value: Optional[np.ndarray] = None

        # Source-space dust detection cache. Geometry deliberately excluded from
        # the key (strokes are source-normalized); resolution excluded so an
        # export reuses the exact preview-detected regions.
        self._retouch_detect_key: Optional[tuple] = None
        self._retouch_detect_value: Optional[tuple] = None
        # Threshold-independent stat maps — survive threshold-slider drags.
        self._dust_stats_key: Optional[tuple] = None
        self._dust_stats_value: Optional[tuple] = None
        # IR ratio + gain map at detection scale, keyed on source only (survives
        # threshold drags); shared by the attenuation bake and IR detection.
        self._ir_gain_key: Optional[tuple] = None
        self._ir_gain_value: Optional[tuple] = None
        # Inpainted source for hairs, keyed on (source+detection params, buffer res)
        # so creative-slider drags reuse it instead of re-inpainting every frame.
        self._hair_key: Optional[tuple] = None
        self._hair_value: Optional[np.ndarray] = None
        # IR-baked source (division + score-weighted fill) and its routed mask, keyed
        # like _hair so creative-slider drags reuse the baked buffer.
        self._ir_recon_key: Optional[tuple] = None
        self._ir_recon_value: Optional[tuple] = None
        # Repaired source for detected specks and for painted strokes. Same keying as
        # _hair: a creative-slider drag reuses the buffer, a new stroke invalidates it
        # through the manual token in source_hash.
        self._luma_key: Optional[tuple] = None
        self._luma_value: Optional[np.ndarray] = None
        self._manual_key: Optional[tuple] = None
        self._manual_value: Optional[tuple] = None
        # The OpenICE method's whole result, on its own slot: the two IR methods share no
        # state, so whichever loses can be deleted without unpicking the other.
        self._ice_key: Optional[tuple] = None
        self._ice_value: Optional[tuple] = None

        # Called with a label when a slow uncached step starts, for a UI busy cue. Set by
        # RenderWorker; the caller runs on the render thread, so keep it to a signal emit.
        self.on_slow_step: Optional[Callable[[str], None]] = None

        if APP_CONFIG.use_gpu:
            gpu = GPUDevice.get()
            if gpu.is_available:
                self.engine_gpu = GPUEngine()
                logger.info("ImageProcessor: Acceleration backend ready")
            else:
                logger.warning("ImageProcessor: GPU unavailable, using CPU fallback")

    @property
    def backend_name(self) -> str:
        if self.engine_gpu:
            return self.engine_gpu.gpu.backend_name or "WEBGPU"
        return "CPU"

    @staticmethod
    def _is_flat(settings: WorkspaceConfig) -> bool:
        """Flat (digital-intermediate) renders run on the CPU engine only, so the
        master is numerically exact and never subject to the looser GPU parity."""
        return settings.exposure.render_intent == RenderIntent.FLAT

    def _slow_step(self, label: str) -> None:
        if self.on_slow_step is not None:
            self.on_slow_step(label)

    def _ir_ratio_gain(self, ir_buffer: np.ndarray, img: np.ndarray, source_key: str) -> tuple:
        """Cached (ratio_det, gain_det, degenerate, gammas) at detection scale."""
        # Detection follows the buffer it will repair (capped): the score is upsampled onto
        # that buffer, so a coarse detection writes a fat mask and averages over a wide
        # support — see _IR_MAX_UPSAMPLE.
        target = ir_detect_target(max(img.shape[:2]), APP_CONFIG.preview_render_size)
        # Key on the source shape — downsample_ir is deterministic in it, so this
        # discriminates the same as the detection shape but resolves before the
        # downsample runs. _ir_bake and _detect_luma both call this per render;
        # keying on the result made the second call repay a full-res erode to build
        # a key it then hit.
        key = (source_key, ir_buffer.shape, target)
        if key == self._ir_gain_key and self._ir_gain_value is not None:
            return self._ir_gain_value
        # Min-preserving (not _downsample_to_long_edge) — INTER_AREA averages a sub-pixel
        # hair's dip away. No-op on the preview path, where preview_ir already arrives
        # min-pooled at this scale.
        ir_det = downsample_ir(np.ascontiguousarray(ir_buffer, dtype=np.float32), target)
        val = ir_ratio_and_gain(ir_det, _downsample_to_long_edge(img, target))
        self._ir_gain_key = key
        self._ir_gain_value = val
        return val

    def _ir_bake(
        self,
        img: np.ndarray,
        ir_buffer: Optional[np.ndarray],
        settings: WorkspaceConfig,
        source_key: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool, Optional[np.ndarray]]:
        """IR correction baked in source transmittance space before the engine (mirrors
        apply_flatfield; the GPU re-uploads source each frame, so the bake reaches it
        parity-free): division attenuation for semi-transparent dust, score-weighted
        fill for opaque cores. Returns (corrected_img, corrected_mask_or_None, degenerate,
        routed_mask_or_None) — the routed mask goes to _hair_inpaint."""
        ret = settings.retouch
        if self._is_flat(settings) or ir_buffer is None or not ret.ir_dust_remove:
            return img, None, False, None
        if ret.ir_method == IR_METHOD_OPENICE:
            return self._ir_bake_openice(img, ir_buffer, ret, source_key)
        ratio_det, gain_det, degenerate, _ = self._ir_ratio_gain(ir_buffer, img, source_key)
        if degenerate:
            return img, None, True, None
        key = (source_key, round(float(ret.ir_threshold), 6), bool(ret.ir_attenuation), img.shape)
        if key == self._ir_recon_key and self._ir_recon_value is not None:
            out, routed = self._ir_recon_value
        else:
            self._slow_step("removing IR dust")
            score_det = ir_defect_score(ratio_det, ir_detect_cutoff(ret.ir_threshold, ret.ir_attenuation))
            out = apply_ir_attenuation(img, gain_det) if ret.ir_attenuation else img
            out = apply_score_repair(out, score_det)
            routed = route_wide_defects(score_det)
            self._ir_recon_key = key
            self._ir_recon_value = (out, routed)
        return out, (ratio_det < 0.97), False, routed

    def _ir_bake_openice(
        self, img: np.ndarray, ir_buffer: np.ndarray, ret: Any, source_key: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool, Optional[np.ndarray]]:
        """The OpenICE method (``features/retouch/openice.py``). Its weight ramp is measured
        off the buffer's own IR noise, so calibration is per resolution and the whole result
        caches under one key, unlike the NegPy method's split gain/recon."""
        key = (source_key, round(float(ret.ir_threshold), 6), img.shape, ir_buffer.shape)
        if key == self._ice_key and self._ice_value is not None:
            return self._ice_value
        self._slow_step("removing IR dust")
        h, w = img.shape[:2]
        long_edge = max(h, w)
        dims = None
        if long_edge > APP_CONFIG.preview_render_size:
            s = APP_CONFIG.preview_render_size / long_edge
            dims = (max(1, round(w * s)), max(1, round(h * s)))
        val = openice.run(img, ir_buffer, float(ret.ir_threshold), dims)
        self._ice_key = key
        self._ice_value = val
        return val

    def _hair_inpaint(self, img: np.ndarray, hair_masks: List[np.ndarray], cache_key: str) -> np.ndarray:
        """Structure-following inpaint of detected hairs, baked into the source before
        the engine (like _ir_bake; the GPU re-uploads source each frame, so it reaches
        both paths parity-free). Cached per (source+params, resolution)."""
        ckey = (cache_key, img.shape)
        if ckey == self._hair_key and self._hair_value is not None:
            return self._hair_value
        self._slow_step("repairing dust")
        out = apply_hair_inpaint(img, hair_masks)
        self._hair_key = ckey
        self._hair_value = out
        return out

    def _detect_luma(
        self,
        settings: WorkspaceConfig,
        img: np.ndarray,
        source_key: str,
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """Source-space luma dust detection → ``(detection-scale score, hair masks)``.
        IR defects never come through here — _ir_bake repairs those."""
        ret = settings.retouch
        if self._is_flat(settings) or not ret.dust_remove:
            return None, []

        key = (
            source_key,
            round(float(ret.dust_threshold), 6),
            int(ret.dust_size),
            settings.process.process_mode,
        )
        if key == self._retouch_detect_key and self._retouch_detect_value is not None:
            return self._retouch_detect_value

        stats_key = (source_key, int(ret.dust_size))
        if stats_key == self._dust_stats_key and self._dust_stats_value is not None:
            stats = self._dust_stats_value
        else:
            stats = compute_dust_stats(_detection_downsample(img), ret.dust_size)
            self._dust_stats_key = stats_key
            self._dust_stats_value = stats
        score, hair_luma = detect_luma_score(_detection_downsample(img), ret.dust_threshold, ret.dust_size, stats=stats)
        value = (score, [hair_luma] if hair_luma is not None else [])
        self._retouch_detect_key = key
        self._retouch_detect_value = value
        return value

    def _luma_bake(self, img: np.ndarray, score: Optional[np.ndarray], cache_key: str) -> np.ndarray:
        """Detected specks repaired into the linear source, ahead of the meters (mirrors
        _ir_bake). Cached per (source+detection params, resolution)."""
        if score is None:
            return img
        ckey = (cache_key, img.shape)
        if ckey == self._luma_key and self._luma_value is not None:
            return self._luma_value
        self._slow_step("repairing dust")
        out = np.asarray(repair_components(img, score))
        self._luma_key = ckey
        self._luma_value = out
        return out

    def _manual_bake(self, img: np.ndarray, settings: WorkspaceConfig, source_key: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Painted strokes and traced scratch lines repaired into the linear source, by the
        same fill the IR and luma paths use (mirrors _ir_bake; the GPU re-uploads source each
        frame, so the bake reaches both engines parity-free). Both hold raw-frame coordinates,
        so this runs before geometry and needs no mapping. Returns the buffer and the mask of
        defects too wide for the fill, for the caller to inpaint."""
        ret = settings.retouch
        lines = getattr(ret, "scratch_lines", [])
        if self._is_flat(settings) or not (ret.manual_heal_strokes or ret.manual_dust_spots or lines):
            return img, None
        key = (source_key, img.shape)
        if key == self._manual_key and self._manual_value is not None:
            return self._manual_value
        # One pass for both hand-placed sources: whichever calls a pixel more damaged wins,
        # and the fill sees every hole at once.
        parts = [
            s
            for s in (
                strokes_to_score(img, ret.manual_heal_strokes, ret.manual_dust_spots),
                lines_to_score(img, lines, getattr(ret, "scratch_threshold", 0.5)),
            )
            if s is not None
        ]
        score = parts[0] if len(parts) == 1 else (np.minimum(*parts) if parts else None)
        if score is None:
            value: Tuple[np.ndarray, Optional[np.ndarray]] = (img, None)
        else:
            self._slow_step("repairing dust")
            # floor=False: a scratch has lost emulsion and reads brighter than the film
            # around it, so a painted repair must be free to darken as well as lighten.
            # Film-footprint supports, not the buffer's own pixels: a painted score arrives at
            # full buffer resolution, where the fill's fine rungs would be 3-9 px — grain scale,
            # sitting entirely inside a defect that is 47 px wide at export. Their candidate is
            # then the defect's own value and the repair only half-lands. The IR path never hits
            # this because its score is measured coarse and carries the same factor here.
            value = (
                np.asarray(repair_components(img, score, floor=False, factor=film_scale(img.shape[:2]))),
                route_wide_defects(score, budget=None),
            )
        self._manual_key = key
        self._manual_value = value
        return value

    def run_pipeline(
        self,
        img: ImageBuffer,
        settings: WorkspaceConfig,
        source_hash: str,
        render_size_ref: float,
        metrics: Optional[Dict[str, Any]] = None,
        prefer_gpu: bool = True,
        readback_metrics: bool = True,
        ir_buffer: Optional[np.ndarray] = None,
        crop_preview_full: bool = False,
        wants_uv_grid: bool = True,
        skip_flatfield: bool = False,
        cam_xyz: Optional[list] = None,
        camera_wb: Optional[list] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes rendering pipeline. Returns result (ndarray/GPUTexture) and metrics.

        ``skip_flatfield``: the export CPU fallbacks pass an already-flat-fielded buffer.
        """
        # Flat-field is a source pre-correction (before geometry/crop); folding its token
        # into source_hash invalidates the engine cache when it changes. Stitch buffers
        # arrive per-part flat-fielded (both decode paths) — correcting the composite
        # canvas as one frame would stretch the gain map across the seam.
        # Shape is in the key: HQ re-decodes the same file larger under an unchanged
        # source_hash.
        precorrect_key = (
            source_hash,
            img.shape,
            skip_flatfield,
            flatfield_token(settings.flatfield),
            sensor_token(settings.process),
            rgbscan_token(settings.rgbscan),
            stitch_token(settings.stitch),
            hdr_token(settings.hdr),
        )
        if self._precorrect_key == precorrect_key and self._precorrect_value is not None:
            img = self._precorrect_value
        else:
            source = img
            if not skip_flatfield and not settings.stitch.stitch_enabled:
                img = apply_flatfield(img, settings.flatfield)
            # Sensor unmix is a source pre-correction like flat-field. skip_flatfield buffers
            # come from _load_source_f32, which already applied it; triplet composites take
            # each channel from its own single-band exposure, so unmixing them would inject
            # crosstalk that was never captured.
            if not skip_flatfield and not is_rgb_triplet(settings.rgbscan) and not stitch_has_triplets(settings.stitch):
                img = apply_sensor_correction(img, effective_sensor_matrix(settings.process))
            # Both no-op'd: caching would pin a second reference to the same buffer.
            if img is not source:
                self._precorrect_key = precorrect_key
                self._precorrect_value = img
        h_orig, w_cols = img.shape[:2]
        # Fold the buffer resolution into source_hash: toggling HQ re-decodes the same
        # file at full resolution with unchanged settings, so without this the engine
        # cache reports "nothing changed" and returns the stale low-res render instead
        # of re-rendering the new full-res buffer.
        base_hash = (
            source_hash
            + flatfield_token(settings.flatfield)
            + rgbscan_token(settings.rgbscan)
            + stitch_token(settings.stitch)
            + hdr_token(settings.hdr)
            + linear_raw_token(settings.process, settings.exposure.render_intent)
            + sensor_token(settings.process)
            + ir_bake_token(settings.retouch, ir_buffer is not None)
            + manual_bake_token(settings.retouch)
        )

        # Bake the IR correction before detection so meters/stats see the corrected buffer.
        want_ir = settings.retouch.ir_dust_remove and ir_buffer is not None and not self._is_flat(settings)
        img, ir_corrected_mask, ir_degenerate, ir_routed = self._ir_bake(img, ir_buffer, settings, base_hash)

        orig_ret = settings.retouch
        detected_dust, hair_masks = self._detect_luma(settings, img, base_hash)
        img = self._luma_bake(img, detected_dust, base_hash + hair_bake_token(orig_ret))
        img, manual_routed = self._manual_bake(img, settings, base_hash)
        extra = [m for m in (ir_routed, manual_routed) if m is not None]
        if extra:
            hair_masks = hair_masks + extra  # never mutate the cached list
        # Inpaint long/twisted hairs into the source (both engines see it — the token
        # invalidates the base stage when detection params change; empty otherwise).
        hair_token = hair_bake_token(orig_ret) if hair_masks else ""
        if hair_masks:
            img = self._hair_inpaint(img, hair_masks, base_hash + hair_token)

        source_hash = base_hash + hair_token + f"|res{w_cols}x{h_orig}"

        scale_factor = max(h_orig, w_cols) / float(APP_CONFIG.preview_render_size)

        context = PipelineContext(
            scale_factor=scale_factor,
            original_size=(h_orig, w_cols),
            process_mode=settings.process.process_mode,
            crop_preview_full=crop_preview_full,
            wants_uv_grid=wants_uv_grid,
            cam_xyz=cam_xyz,
            camera_wb=camera_wb,
        )
        if metrics:
            context.metrics.update(metrics)
        # Display-overlay data: the detection-scale set that was repaired. Absent when
        # detection is off (so the overlay draws nothing).
        if detected_dust is not None:
            context.metrics["detected_dust_mask"] = detected_dust < 1.0
        # Overlay wash over the inpainted hairs (they emit no stroke capsules).
        if hair_masks:
            context.metrics["hair_inpaint_masks"] = hair_masks
        # Overlay data for the IR-corrected regions (division + fill) + the B&W/Kodachrome guard.
        if want_ir:
            context.metrics["ir_degenerate"] = ir_degenerate
            if ir_corrected_mask is not None:
                context.metrics["ir_corrected_mask"] = ir_corrected_mask

        if self._is_flat(settings) or crop_preview_full:
            # The crop tool's "show full uncropped frame" preview only needs a single
            # CPU render per settings change (dragging only moves an overlay rect), so
            # we sidestep the GPU engine's ROI-fused compute dispatch entirely here.
            prefer_gpu = False

        if prefer_gpu and self.engine_gpu:
            try:
                processed, gpu_metrics = self.engine_gpu.process_to_texture(
                    img,
                    settings,
                    scale_factor=scale_factor,
                    render_size_ref=render_size_ref,
                    readback_metrics=readback_metrics,
                    source_hash=source_hash,
                    analysis_source_hash=source_hash,
                    cam_xyz=cam_xyz,
                    camera_wb=camera_wb,
                )
                context.metrics.update(gpu_metrics)
                return processed, context.metrics
            except Exception:
                logger.exception("Hardware acceleration failed, falling back to CPU")
                context.metrics["gpu_fallback"] = True

        processed = self.engine_cpu.process(img, settings, source_hash, context)
        return processed, context.metrics

    def buffer_to_pil(self, buffer: Any, settings: WorkspaceConfig, bit_depth: int = 8) -> Image.Image:
        """Converts float32 buffer to calibrated PIL Image."""
        if not isinstance(buffer, np.ndarray):
            raise ValueError("Direct GPU textures cannot be converted to PIL without readback.")

        t = settings.toning
        # Any toner or split tint makes a B&W print chromatic — collapsing to a
        # single luma plane here would silently discard the toning.
        is_toned = (
            t.selenium_strength != 0.0
            or t.sepia_strength != 0.0
            or t.gold_strength != 0.0
            or t.blue_strength != 0.0
            or t.copper_strength != 0.0
            or t.vanadium_strength != 0.0
            or t.shadow_tint_strength != 0.0
            or t.highlight_tint_strength != 0.0
            or settings.altproc.alt_process != AltProcess.NONE
        )
        is_bw = settings.process.process_mode == ProcessMode.BW and not is_toned

        if is_bw:
            img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=bit_depth)
            return Image.fromarray(img_int)

        if bit_depth == 8:
            return Image.fromarray(float_to_uint8(buffer))
        elif bit_depth == 16:
            if buffer.ndim == 2 or (buffer.ndim == 3 and buffer.shape[2] == 1):
                return Image.fromarray(float_to_uint16(buffer))
            return Image.fromarray(float_to_uint8(buffer))
        raise ValueError(f"Unsupported bit depth: {bit_depth}")

    def _decode_sensor_rgb(
        self, file_path: str, linear_raw: bool, fast: bool = False, wb_override: Optional[Sequence[float]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode one RAW to sensor-native (output_color=raw), linear uint16 RGB.

        `fast` allows a half-size decode (contact-sheet tiles); ignored where
        half_size would distort colors (see _use_half_size_decode).

        `wb_override` decodes on someone else's white balance instead of this file's own.
        Frames of one bracket must share a scale or the exposure ratios solved between them
        absorb the difference, and `use_camera_wb` reads each *file's* as-shot multipliers —
        which differ per frame on a camera left in auto white balance.

        Returns (rgb_uint16, loader_metadata).
        """
        ctx_mgr, metadata = loader_factory.get_loader(file_path, linear_raw=linear_raw)
        with ctx_mgr as raw:
            algo = get_best_demosaic_algorithm(raw)
            user_wb = [1, 1, 1, 1] if linear_raw else (list(wb_override) if wb_override is not None else None)
            post_kw: Dict[str, Any] = {"half_size": True} if fast and _use_half_size_decode(raw, linear_raw) else {}
            rgb = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                adjust_maximum_thr=0.0,  # fixed white level, never the frame's own max
                use_camera_wb=not linear_raw and wb_override is None,
                user_wb=user_wb,
                output_bps=16,
                output_color=rawpy.ColorSpace.raw,
                demosaic_algorithm=algo,
                user_flip=0,
                **post_kw,
            )
            rgb = ensure_rgb(rgb)
            # Sensor-native decode leaves the buffer in camera primaries; the transparency
            # transfer needs the matrix to reach the working space (see capture_color).
            metadata["cam_xyz"] = camera_xyz_matrix(raw)
            metadata["camera_wb"] = camera_wb_multipliers(raw)
        return rgb, metadata

    def _load_source_f32(
        self, file_path: str, params: WorkspaceConfig, fast_decode: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Decode a source file to a flatfield-corrected, EXIF-oriented float32 buffer.

        A stitch composite decodes every part and assembles them by replaying the
        registration stored in ``params.stitch``, each part against its own rgbscan
        config rather than the primary's.

        Returns (f32_buffer, ir_buffer, source_color_space).
        """
        is_triplet = is_rgb_triplet(params.rgbscan) or stitch_has_triplets(params.stitch)
        # Narrowband triplet channels don't survive half_size CFA binning.
        fast_decode = fast_decode and not is_triplet

        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            mtime = 0.0
        cache_key = (
            file_path,
            mtime,
            effective_linear_raw(params.process, params.exposure.render_intent),
            rgbscan_token(params.rgbscan),
            stitch_token(params.stitch),
            hdr_token(params.hdr),
            flatfield_token(params.flatfield),
            sensor_token(params.process),
            fast_decode,
        )
        if cache_key == self._source_cache_key and self._source_cache_value is not None:
            return self._source_cache_value

        if params.stitch.stitch_enabled and params.stitch.stitch_paths:
            parts, irs = [], []
            source_cs = WORKING_COLOR_SPACE
            for i, path in enumerate((file_path, *params.stitch.stitch_paths)):
                f32, ir, cs = self._decode_oriented_f32(path, _part_params(params, i), fast_decode)
                if i == 0:
                    source_cs = cs
                parts.append(f32)
                irs.append(ir)
            f32_buffer, ir_full = stitch_composite(parts, irs, params.stitch)
            result = (f32_buffer, ir_full, source_cs)
        else:
            result = self._decode_oriented_f32(file_path, params, fast_decode)

        self._source_cache_key = cache_key
        self._source_cache_value = result
        return result

    def _decode_oriented_f32(
        self, file_path: str, params: WorkspaceConfig, fast_decode: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Single-file decode tail: sensor RGB -> float32 -> EXIF orientation -> flatfield.

        Stitch registration is estimated on buffers produced here, so any decode
        the transforms are replayed against must come through here too.
        """
        linear_raw = effective_linear_raw(params.process, params.exposure.render_intent)
        rgbcfg = params.rgbscan
        # A bracket wins over a triplet: the two are refused together in the UI, and the
        # export decode branches in this order — an asset that somehow carries both must
        # not assemble differently on the two paths.
        is_triplet = is_rgb_triplet(rgbcfg) and not hdr_active(params.hdr)

        rgb, metadata = self._decode_sensor_rgb(file_path, linear_raw, fast=fast_decode)
        # No embedded profile (scanner-raw linear, sensor-native RAW) → the buffer is
        # already in the working space, so "Same as Source" exports without converting.
        source_cs = str(metadata.get("color_space") or WORKING_COLOR_SPACE)
        # Memoized rather than returned: the 3-tuple return is unpacked positionally in
        # several callers, and only the export render needs this.
        self._cam_xyz_by_path[file_path] = (metadata.get("cam_xyz"), metadata.get("camera_wb"))
        ir_full = metadata.get("ir")

        if is_triplet:
            # Assemble one frame from the R/G/B exposures; the primary (red) file is
            # already decoded above, so reuse it and decode only green/blue.
            for label, path in (("green", rgbcfg.green_path), ("blue", rgbcfg.blue_path)):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"RGB-scan {label} exposure not found: {path}")

            def _decode(path: str) -> np.ndarray:
                if path == file_path:
                    return rgb
                return self._decode_sensor_rgb(path, linear_raw)[0]

            rgb = merge_rgb_triplet(_decode, file_path, rgbcfg.green_path, rgbcfg.blue_path, align=rgbcfg.align)

        if hdr_active(params.hdr):
            # Merged straight to float32: the recovered detail sits below the reference
            # exposure's quantization step, so a uint16 round-trip would discard it.
            # Before flat-field and the sensor unmix, both of which are applied below —
            # the decode pins the white level (adjust_maximum_thr=0.0), so saturation is
            # exactly 1.0 and the merge's exclusion threshold means what it says. A gain
            # map applied first moves that point and skews the weights.
            #
            # Every frame decodes on the reference's white balance, never its own. The
            # transfer path already decodes neutral (effective_linear_raw), so this is
            # normally moot -- but it is pinned here rather than left to that, because a
            # bracket whose frames sit on different white balances solves wrong ratios and
            # says nothing about it.
            bracket_wb = None if linear_raw else metadata.get("camera_wb")
            f32_buffer = merge_bracket(
                # fast_decode must ride along: a half-size primary against full-size
                # siblings is a shape mismatch, not a slow merge.
                lambda p: rgb if p == file_path else self._decode_sensor_rgb(p, linear_raw, fast=fast_decode, wb_override=bracket_wb)[0],
                file_path,
                params.hdr,
            )
        else:
            f32_buffer = uint16_to_float32(rgb)

        if ir_full is not None and ir_full.shape[:2] != f32_buffer.shape[:2]:
            # Defensive: no current IR carrier half-sizes (libraw ignores half_size on
            # stacked LinearRaw, NonStandardFileWrapper is excluded from fast decode), but a
            # mismatched plane must be rescaled or the IR bake silently skips it. Routed
            # through downsample_ir, not INTER_AREA, so a sub-pixel hair keeps its dip.
            ih, iw = f32_buffer.shape[:2]
            ir_full = downsample_ir(ir_full, max(ih, iw), dims=(iw, ih))

        orientation = metadata.get("orientation", 1)
        f32_buffer = apply_exif_orientation(f32_buffer, orientation)
        f32_buffer = apply_flatfield(f32_buffer, params.flatfield)
        if not is_triplet:
            f32_buffer = apply_sensor_correction(f32_buffer, effective_sensor_matrix(params.process))
        if ir_full is not None:
            ir_full = apply_exif_orientation(ir_full, orientation)
        return f32_buffer, ir_full, source_cs

    @staticmethod
    def _slice_half_source(
        f32_buffer: np.ndarray,
        ir_full: Optional[np.ndarray],
        half: int,
        split_x: float,
        crop_rect: Optional[tuple[float, float, float, float]] = None,
        gutter_thickness: float = 0.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Slice a decoded source down to one half-frame; copies so the shared
        per-file decode cache is never mutated downstream. No-op when half == 0."""
        if not half:
            return f32_buffer, ir_full
        from negpy.services.assets.half_frame import slice_half

        f32_buffer = np.ascontiguousarray(slice_half(f32_buffer, half, split_x, crop_rect=crop_rect, gutter_thickness=gutter_thickness))
        if ir_full is not None:
            ir_full = np.ascontiguousarray(slice_half(ir_full, half, split_x, crop_rect=crop_rect, gutter_thickness=gutter_thickness))
        return f32_buffer, ir_full

    def process_export(
        self,
        file_path: str,
        params: WorkspaceConfig,
        export_settings,  # ExportConfig or ExportPreset
        source_hash: str,
        metrics: Optional[Dict[str, Any]] = None,
        prefer_gpu: bool = True,
        bounds_override: Optional[Any] = None,
        working_color_space: str = WORKING_COLOR_SPACE,
        half: int = 0,
        split_x: float = 0.5,
        crop_rect: Optional[tuple[float, float, float, float]] = None,
        gutter_thickness: float = 0.0,
    ) -> Tuple[Optional[bytes], str]:
        """Performs high-resolution export with color management."""
        try:
            # Ensure both GPU and CPU paths use the same export settings.
            params = dc_replace(params, export=export_settings)

            f32_buffer, ir_full, source_cs = self._load_source_f32(file_path, params)
            f32_buffer, ir_full = self._slice_half_source(
                f32_buffer, ir_full, half, split_x, crop_rect=crop_rect, gutter_thickness=gutter_thickness
            )
            target_cs = export_settings.export_color_space
            if target_cs == ColorSpace.SAME_AS_SOURCE.value:
                target_cs = source_cs
            color_space = str(target_cs)

            detect_key = (
                source_hash
                + flatfield_token(params.flatfield)
                + rgbscan_token(params.rgbscan)
                + stitch_token(params.stitch)
                + hdr_token(params.hdr)
                + linear_raw_token(params.process, params.exposure.render_intent)
                + sensor_token(params.process)
                + ir_bake_token(params.retouch, ir_full is not None)
                + manual_bake_token(params.retouch)
            )
            f32_buffer, _, _, ir_routed = self._ir_bake(f32_buffer, ir_full, params, detect_key)
            orig_ret = params.retouch
            detected, hair_masks = self._detect_luma(params, f32_buffer, detect_key)
            f32_buffer = self._luma_bake(f32_buffer, detected, detect_key + hair_bake_token(orig_ret))
            f32_buffer, manual_routed = self._manual_bake(f32_buffer, params, detect_key)
            extra = [m for m in (ir_routed, manual_routed) if m is not None]
            if extra:
                hair_masks = hair_masks + extra
            if hair_masks:
                f32_buffer = self._hair_inpaint(f32_buffer, hair_masks, detect_key + hair_bake_token(orig_ret))

            h_raw, w_raw = f32_buffer.shape[:2]
            export_scale = max(h_raw, w_raw) / float(APP_CONFIG.preview_render_size)

            if self._is_flat(params):
                prefer_gpu = False

            if prefer_gpu and self.engine_gpu:
                buffer, gpu_metrics = self.engine_gpu.process(
                    f32_buffer,
                    params,
                    scale_factor=export_scale,
                    bounds_override=bounds_override,
                    readback_metrics=False,
                    cam_xyz=self._cam_xyz_by_path.get(file_path, (None, None))[0],
                    camera_wb=self._cam_xyz_by_path.get(file_path, (None, None))[1],
                )
            else:
                buffer, _ = self.run_pipeline(
                    f32_buffer,
                    params,
                    source_hash,
                    render_size_ref=float(APP_CONFIG.preview_render_size),
                    metrics=metrics or {"log_bounds": bounds_override} if bounds_override else metrics,
                    prefer_gpu=False,
                    wants_uv_grid=False,
                    skip_flatfield=True,  # f32_buffer already flat-fielded by _load_source_f32
                    cam_xyz=self._cam_xyz_by_path.get(file_path, (None, None))[0],
                    camera_wb=self._cam_xyz_by_path.get(file_path, (None, None))[1],
                )
                buffer = self._apply_scaling_and_border_f32(buffer, params, params.export)
                # Release full-res arrays pinned in the CPU stage cache.
                self.engine_cpu.cache.clear()

            return self._encode_export(buffer, export_settings, color_space, working_color_space)

        except Exception as e:
            logger.error(f"Export pipeline failed: {e}")
            return None, str(e)

    def _encode_export(
        self,
        buffer: np.ndarray,
        export_settings,
        color_space: str,
        working_color_space: str = WORKING_COLOR_SPACE,
    ) -> Tuple[bytes, str]:
        """Encodes a processed float buffer to the target format's file bytes.

        Input ICC overrides the source, output ICC the destination; both are always
        applied so the file matches the preview.
        """
        fmt = export_settings.export_fmt
        icc_input = export_settings.icc_input_path
        icc_output = export_settings.icc_output_path

        # A target with no ICC profile (ACES/XYZ, or a stale custom name) can be
        # neither converted to nor tagged — the file would silently carry untagged
        # working-space pixels. Export the working space itself, correctly tagged,
        # instead. An output override supplies its own destination, so it exempts.
        if ColorSpaceRegistry.get_icc_path(color_space) is None and not (icc_output and os.path.exists(icc_output)):
            logger.warning(f"No ICC profile available for '{color_space}'; exporting as {working_color_space} instead")
            color_space = working_color_space

        is_greyscale = color_space == ColorSpace.GREYSCALE.value

        if fmt == ExportFormat.TIFF:
            if is_greyscale:
                img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=16)
                img_out, icc_bytes = self._apply_color_management_u16_greyscale(
                    img_int, working_color_space, color_space, icc_output, icc_input
                )
            else:
                img_int = float_to_uint16(buffer)
                img_out, icc_bytes = self._apply_color_management_u16(img_int, working_color_space, color_space, icc_output, icc_input)

            output_buf = io.BytesIO()
            tifffile.imwrite(
                output_buf,
                img_out,
                photometric="rgb" if img_out.ndim == 3 else "minisblack",
                iccprofile=icc_bytes,
                compression="zlib",
                predictor=True,
            )
            return output_buf.getvalue(), "tiff"
        elif fmt == ExportFormat.PNG:
            if is_greyscale:
                # PIL "I;16" supports 16-bit greyscale, so keep full bit depth here.
                img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=16)
                img_out, icc_bytes = self._apply_color_management_u16_greyscale(
                    img_int, working_color_space, color_space, icc_output, icc_input
                )
                pil_img = Image.fromarray(img_out)
            else:
                # PIL has no 16-bit RGB mode, so RGB PNG is 8-bit (TIFF is the 16-bit
                # lossless path). Mirror the JPEG branch for color management.
                img_int = float_to_uint8(buffer)
                pil_img, icc_bytes = self.apply_color_management(
                    Image.fromarray(img_int), working_color_space, color_space, icc_output, icc_input
                )
            output_buf = io.BytesIO()
            save_kwargs: Dict[str, Any] = {"format": "PNG", "compress_level": 6}
            if icc_bytes:
                save_kwargs["icc_profile"] = icc_bytes
            pil_img.save(output_buf, **save_kwargs)
            return output_buf.getvalue(), "png"
        elif fmt == ExportFormat.JXL:
            tag = _JXL_COLOR.get(color_space)
            if tag is None:
                raise ValueError(
                    f"JPEG XL export does not support the {color_space} color space. "
                    "Use sRGB, P3 D65, Rec 2020, or Greyscale, or pick another format."
                )
            photometric, primaries, transfer = tag
            # 16-bit, color-managed to target; ICC discarded (libjxl tags enumeratively).
            if is_greyscale:
                img_int = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=16)
                img_out, _icc = self._apply_color_management_u16_greyscale(img_int, working_color_space, color_space, icc_output, icc_input)
            else:
                img_int = float_to_uint16(buffer)
                img_out, _icc = self._apply_color_management_u16(img_int, working_color_space, color_space, icc_output, icc_input)
            bits = imagecodecs.jpegxl_encode(
                np.ascontiguousarray(img_out),
                bitspersample=16,
                photometric=photometric,
                primaries=primaries,
                transfer=transfer,
                lossless=export_settings.jxl_lossless,
                distance=None if export_settings.jxl_lossless else export_settings.jxl_distance,
                effort=export_settings.jxl_effort,
                numthreads=0,  # all cores; single-threaded otherwise (~7x slower)
            )
            return bytes(bits), "jxl"
        elif fmt == ExportFormat.WEBP:
            # 8-bit only (WebP has no higher bit depth). Lossy or lossless via a
            # flag; PIL embeds the ICC profile for any color space.
            if is_greyscale:
                # apply_color_management can't transform an L image with the RGB
                # working profile (lcms refuses the transform, leaving the file
                # untagged in the working TRC); use the synthetic grey re-encode.
                pil_img, icc_bytes = self._greyscale_to_pil_u8(buffer, working_color_space, color_space, icc_output, icc_input)
            else:
                pil_img, icc_bytes = self.apply_color_management(
                    Image.fromarray(float_to_uint8(buffer)), working_color_space, color_space, icc_output, icc_input
                )
            if max(pil_img.size) > 16383:
                raise ValueError("WebP max dimension is 16383 px; use TIFF/PNG for larger exports.")
            output_buf = io.BytesIO()
            save_kwargs: Dict[str, Any] = {
                "format": "WEBP",
                "lossless": export_settings.webp_lossless,
                "quality": export_settings.webp_quality,
                "method": export_settings.webp_method,
            }
            if icc_bytes:
                save_kwargs["icc_profile"] = icc_bytes
            pil_img.save(output_buf, **save_kwargs)
            return output_buf.getvalue(), "webp"
        else:
            if is_greyscale:
                # Same constraint as the WebP branch: greyscale can't go through the
                # RGB CMS transform, so re-encode at 16-bit and downconvert.
                pil_img, icc_bytes = self._greyscale_to_pil_u8(buffer, working_color_space, color_space, icc_output, icc_input)
            else:
                pil_img, icc_bytes = self.apply_color_management(
                    Image.fromarray(float_to_uint8(buffer)),
                    working_color_space,
                    color_space,
                    icc_output,
                    icc_input,
                )
            output_buf = io.BytesIO()
            self._save_to_pil_buffer(pil_img, output_buf, export_settings, icc_bytes)
            return output_buf.getvalue(), "jpg"

    def render_display_array(
        self,
        file_path: str,
        params: WorkspaceConfig,
        source_hash: str,
        target_long_px: int,
        prefer_gpu: bool = True,
        working_color_space: str = WORKING_COLOR_SPACE,
        fast_decode: bool = False,
        half: int = 0,
        split_x: float = 0.5,
        crop_rect: Optional[tuple[float, float, float, float]] = None,
        gutter_thickness: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Render a file (with its edits) to a small sRGB uint8 RGB array for tiling.

        Mirrors the export render path but at small resolution and in display space,
        so a contact-sheet tile matches the on-canvas look. Returns None on failure.

        The source is shrunk to target_long_px *before* the pipeline runs, and the
        paper layout is bounded to the same size. Rendering the full-res source and
        discarding all but a tile cost ~3.5GB peak RSS (CPU) / ~8GB commit (GPU) per
        24MP frame, so a large roll exhausted memory and tiles were silently dropped.
        """
        try:
            from negpy.infrastructure.display.color_mgmt import apply_display_transform

            f32_buffer, ir_full, _ = self._load_source_f32(file_path, params, fast_decode=fast_decode)
            f32_buffer, ir_full = self._slice_half_source(
                f32_buffer, ir_full, half, split_x, crop_rect=crop_rect, gutter_thickness=gutter_thickness
            )

            # Proof scale: everything downstream only needs target_long_px. The
            # cached source buffer is shared, so resize (never mutate) it.
            f32_buffer = _downsample_to_long_edge(f32_buffer, target_long_px)
            if ir_full is not None and ir_full.shape[:2] != f32_buffer.shape[:2]:
                th, tw = f32_buffer.shape[:2]
                ir_full = cv2.resize(ir_full, (tw, th), interpolation=cv2.INTER_AREA)
            # Each frame of a contact sheet is decoded once, so the full-res source
            # cache only pins ~300MB (24MP) across the next frame's decode.
            self._source_cache_key = None
            self._source_cache_value = None
            self._precorrect_key = None
            self._precorrect_value = None

            # A Print/Target-px export setting sizes the paper from print_size x DPI,
            # which would re-inflate the tile to full print resolution right after the
            # downsample. Bound it with the virtual-DPI trick the UI print preview uses
            # (PrintService.preview_paper_layout) so borders stay proportional.
            if params.export.export_resolution_mode != ExportResolutionMode.ORIGINAL.value:
                virtual_dpi = max(1, int((target_long_px * 2.54) / max(0.1, params.export.export_print_size)))
                params = dc_replace(
                    params,
                    export=dc_replace(
                        params.export,
                        export_resolution_mode=ExportResolutionMode.PRINT.value,
                        export_dpi=virtual_dpi,
                    ),
                )

            h_raw, w_raw = f32_buffer.shape[:2]
            scale_factor = max(1.0, max(h_raw, w_raw) / float(target_long_px))

            detect_key = (
                source_hash
                + flatfield_token(params.flatfield)
                + rgbscan_token(params.rgbscan)
                + stitch_token(params.stitch)
                + hdr_token(params.hdr)
                + linear_raw_token(params.process, params.exposure.render_intent)
                + sensor_token(params.process)
                + ir_bake_token(params.retouch, ir_full is not None)
                + manual_bake_token(params.retouch)
            )
            f32_buffer, _, _, ir_routed = self._ir_bake(f32_buffer, ir_full, params, detect_key)
            orig_ret = params.retouch
            detected, hair_masks = self._detect_luma(params, f32_buffer, detect_key)
            f32_buffer = self._luma_bake(f32_buffer, detected, detect_key + hair_bake_token(orig_ret))
            f32_buffer, manual_routed = self._manual_bake(f32_buffer, params, detect_key)
            extra = [m for m in (ir_routed, manual_routed) if m is not None]
            if extra:
                hair_masks = hair_masks + extra
            if hair_masks:
                f32_buffer = self._hair_inpaint(f32_buffer, hair_masks, detect_key + hair_bake_token(orig_ret))

            if self._is_flat(params):
                prefer_gpu = False

            if prefer_gpu and self.engine_gpu:
                buffer, _ = self.engine_gpu.process(
                    f32_buffer,
                    params,
                    scale_factor=scale_factor,
                    readback_metrics=False,
                    cam_xyz=self._cam_xyz_by_path.get(file_path, (None, None))[0],
                    camera_wb=self._cam_xyz_by_path.get(file_path, (None, None))[1],
                )
            else:
                buffer, _ = self.run_pipeline(
                    f32_buffer,
                    params,
                    source_hash,
                    render_size_ref=float(target_long_px),
                    prefer_gpu=False,
                    wants_uv_grid=False,
                    skip_flatfield=True,  # f32_buffer already flat-fielded by _load_source_f32
                    cam_xyz=self._cam_xyz_by_path.get(file_path, (None, None))[0],
                    camera_wb=self._cam_xyz_by_path.get(file_path, (None, None))[1],
                )
                buffer = self._apply_scaling_and_border_f32(buffer, params, params.export)
                self.engine_cpu.cache.clear()

            if isinstance(buffer, np.ndarray) and buffer.ndim == 3 and buffer.shape[2] == 4:
                buffer = buffer[:, :, :3]
            buffer = apply_display_transform(buffer, working_color_space)
            # Downsample after the display transform, matching where the sheet
            # compositor resamples, so the tile looks identical — just smaller.
            buffer = _downsample_to_long_edge(buffer, target_long_px)
            return float_to_uint8(buffer)
        except Exception as e:
            logger.error(f"Contact-sheet tile render failed for {file_path}: {e}")
            return None

    def _apply_scaling_and_border_f32(self, img: np.ndarray, params: WorkspaceConfig, export_settings: ExportConfig) -> np.ndarray:
        """CPU fallback for layout application."""
        result, _ = PrintService.apply_layout(
            img,
            export_settings,
            border_size=params.finish.border_size,
            border_color=PrintService.effective_border_color(params.finish, params.toning),
            finish=params.finish,
        )
        return result

    def _get_target_icc_bytes(self, color_space: str, icc_path: Optional[str]) -> Optional[bytes]:
        """Loads ICC profile data for embedding (custom output profile or target space)."""
        if icc_path and os.path.exists(icc_path):
            with open(icc_path, "rb") as f:
                return f.read()
        path = ColorSpaceRegistry.get_icc_path(color_space)
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        return None

    @staticmethod
    def _has_custom_icc(input_icc_path: Optional[str], output_icc_path: Optional[str]) -> bool:
        """True when an input or output ICC override file is present."""
        return bool((input_icc_path and os.path.exists(input_icc_path)) or (output_icc_path and os.path.exists(output_icc_path)))

    @staticmethod
    def _resolve_src_profile(working_color_space: str, input_icc_path: Optional[str]) -> Any:
        """Source profile: input ICC override if present, else the working space."""
        if input_icc_path and os.path.exists(input_icc_path):
            return ImageCms.getOpenProfile(input_icc_path)
        path_src = ColorSpaceRegistry.get_icc_path(working_color_space)
        return ImageCms.getOpenProfile(path_src) if path_src and os.path.exists(path_src) else ImageCms.createProfile("sRGB")

    @staticmethod
    def _resolve_dst_profile(color_space: str, output_icc_path: Optional[str]) -> Any:
        """Destination profile: output ICC override if present, else the target space (or None)."""
        if output_icc_path and os.path.exists(output_icc_path):
            return ImageCms.getOpenProfile(output_icc_path)
        path_dst = ColorSpaceRegistry.get_icc_path(color_space)
        return ImageCms.getOpenProfile(path_dst) if path_dst and os.path.exists(path_dst) else None

    @staticmethod
    def _is_print_profile(profile: Any) -> bool:
        """True for a paper/printer output profile (gets a paper-white soft proof)."""
        device_class = (getattr(profile.profile, "device_class", "") or "").strip()
        color_space = (getattr(profile.profile, "xcolor_space", "") or "").strip()
        return device_class == "prtr" or color_space == "CMYK"

    def _apply_color_management_u16_rgb(
        self,
        img_u16: np.ndarray,
        working_color_space: str,
        color_space: str,
        output_icc_path: Optional[str],
        input_icc_path: Optional[str] = None,
        lut_size: int = DEFAULT_LUT_SIZE,
    ) -> Tuple[np.ndarray, Optional[bytes]]:
        """ICC RGB transform for 16-bit arrays (PIL has no 16-bit RGB mode).

        Source is the input override or the working space; destination is the output
        override or the target space. One src→dst transform, so the embedded profile
        matches the pixels.

        Trilinear LUT approximation, not the exact per-pixel evaluation
        _apply_color_management_u16 gets from lcms2 via imagecodecs — this is the
        fallback for when that codec is unavailable (see its docstring), so accuracy
        trades off against not depending on it. ``lut_size`` lets an export ask for a
        finer grid than the interactive display LUT needs.
        """
        has_custom = self._has_custom_icc(input_icc_path, output_icc_path)
        if not has_custom and working_color_space == color_space:
            return img_u16, self._get_target_icc_bytes(color_space, None)
        try:
            p_src = self._resolve_src_profile(working_color_space, input_icc_path)
            p_dst = self._resolve_dst_profile(color_space, output_icc_path)
            if p_dst is None:
                return img_u16, self._get_target_icc_bytes(color_space, None)
            result = apply_icc_u16_rgb(
                img_u16,
                p_src,
                p_dst,
                ImageCms.Intent.RELATIVE_COLORIMETRIC,
                ImageCms.Flags.BLACKPOINTCOMPENSATION,
                size=lut_size,
            )
            return result, self._get_target_icc_bytes(color_space, output_icc_path)
        except Exception as e:
            logger.error(f"CMS transformation failed: {e}")
            return img_u16, None

    def _apply_color_management_u16_greyscale(
        self,
        img_u16: np.ndarray,
        working_color_space: str,
        color_space: str,
        output_icc_path: Optional[str],
        input_icc_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[bytes]]:
        """Re-encode a (H,W) uint16 luma buffer to the tagged grey profile's TRC.

        The buffer is luma in the working TRC. The bundled GrayGamma2.2.icc actually
        carries the sRGB TRC despite its name (verified against the profile; see the
        _JXL_COLOR note), and JXL tags greyscale as SRGB transfer — so encode with the
        sRGB OETF, not a pure 2.2 power, or the pixels won't match their own tag in
        the shadows. An RGB working profile can't drive a 1-channel ICC transform,
        hence this synthetic re-encode instead of lcms.
        """
        if working_color_space == color_space:
            return img_u16, self._get_target_icc_bytes(color_space, output_icc_path)
        lin = np.clip(np.asarray(working_oetf_decode(img_u16.astype(np.float32) / 65535.0)), 0.0, 1.0)
        gray = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
        out = np.clip(gray * 65535.0 + 0.5, 0.0, 65535.0).astype(np.uint16)
        return out, self._get_target_icc_bytes(color_space, output_icc_path)

    def _greyscale_to_pil_u8(
        self,
        buffer: np.ndarray,
        working_color_space: str,
        color_space: str,
        output_icc_path: Optional[str],
        input_icc_path: Optional[str] = None,
    ) -> Tuple[Image.Image, Optional[bytes]]:
        """Greyscale buffer -> color-managed 8-bit L image for JPEG/WebP.

        Runs the 16-bit grey re-encode first so the 8-bit result carries the same
        TRC (and profile bytes) as the greyscale TIFF/PNG of the same edit.
        """
        img16 = float_to_uint_luma(np.ascontiguousarray(buffer), bit_depth=16)
        img16, icc_bytes = self._apply_color_management_u16_greyscale(
            img16, working_color_space, color_space, output_icc_path, input_icc_path
        )
        img8 = np.clip(np.round(img16.astype(np.float32) / 257.0), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(img8), icc_bytes

    def _apply_color_management_u16(
        self,
        img_u16: np.ndarray,
        working_color_space: str,
        color_space: str,
        output_icc_path: Optional[str],
        input_icc_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[bytes]]:
        """ICC RGB transform for 16-bit arrays using lcms2 via imagecodecs.

        PIL has no 16-bit RGB mode so we use imagecodecs.cms_transform
        (already a dependency for JXL) which evaluates lcms2 at full
        16-bit precision on numpy arrays directly.

        cms_transform's underlying dlopen can fail (see
        LIBLCMS2_DYLIB_COLLISION.md for one real cause); imagecodecs' lazy
        resolver only surfaces that as a generic ImportError, discarding the
        real reason. Fall back to the LUT-based transform rather than
        silently export unmanaged pixels.
        """
        has_custom = self._has_custom_icc(input_icc_path, output_icc_path)
        if not has_custom and working_color_space == color_space:
            return img_u16, self._get_target_icc_bytes(color_space, None)

        try:
            src_bytes = self._get_target_icc_bytes(working_color_space, input_icc_path)
            dst_bytes = self._get_target_icc_bytes(color_space, output_icc_path)
            if src_bytes is None or dst_bytes is None:
                logger.warning("CMS skipped: ICC profile not found")
                return img_u16, self._get_target_icc_bytes(color_space, output_icc_path)

            try:
                result = imagecodecs.cms_transform(
                    np.ascontiguousarray(img_u16),
                    src_bytes,
                    dst_bytes,
                    colorspace="RGB",
                    outcolorspace="RGB",
                    intent=1,  # RELATIVE_COLORIMETRIC
                    flags=0x2000,  # BLACKPOINTCOMPENSATION (matches PIL's value)
                )
            except ImportError as e:
                self._log_cms_codec_dlopen_error()
                logger.warning(
                    f"imagecodecs CMS codec unavailable on this build ({e}); falling back to the LUT-based ICC transform for this export"
                )
                return self._apply_color_management_u16_rgb(
                    img_u16,
                    working_color_space,
                    color_space,
                    output_icc_path,
                    input_icc_path,
                    lut_size=PROOF_LUT_SIZE,
                )
            # imagecodecs outputs uint16 [0,65535] directly
            return result, self._get_target_icc_bytes(color_space, output_icc_path)
        except Exception as e:
            logger.error(f"CMS transformation failed: {e}")
            return img_u16, None

    @staticmethod
    def _log_cms_codec_dlopen_error() -> None:
        """Log the real dlopen error behind a cms_transform ImportError, if any.

        Best-effort diagnostic only, never raises: reproduces imagecodecs'
        lazy import via ctypes.CDLL, which does not discard the OSError.
        """
        try:
            codec_path = os.path.join(os.path.dirname(imagecodecs.__file__), "_cms.abi3.so")
            ctypes.CDLL(codec_path)
        except OSError as dlopen_error:
            logger.warning(f"underlying dlopen error for imagecodecs' CMS codec: {dlopen_error}")
        except Exception:
            pass

    def apply_color_management(
        self,
        pil_img: Image.Image,
        working_color_space: str,
        color_space: str,
        output_icc_path: Optional[str],
        input_icc_path: Optional[str] = None,
    ) -> Tuple[Image.Image, Optional[bytes]]:
        """ICC transform for export. Source is the input override or working space;
        destination is the output override or target space."""
        has_custom = self._has_custom_icc(input_icc_path, output_icc_path)
        if not has_custom and working_color_space == color_space:
            return pil_img, self._get_target_icc_bytes(color_space, None)

        try:
            p_src = self._resolve_src_profile(working_color_space, input_icc_path)
            p_dst = self._resolve_dst_profile(color_space, output_icc_path)
            if p_dst is None:
                return pil_img, self._get_target_icc_bytes(color_space, None)

            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB" if pil_img.mode != "I;16" else "L")

            result_pil = ImageCms.profileToProfile(
                pil_img,
                p_src,
                p_dst,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode="RGB" if pil_img.mode != "L" else "L",
                flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
            )
            if result_pil:
                pil_img = result_pil
            return pil_img, self._get_target_icc_bytes(color_space, output_icc_path)
        except Exception as e:
            logger.error(f"CMS transformation failed: {e}")
            return pil_img, None

    @staticmethod
    def soft_proof_preview(
        pil_img: Image.Image,
        working_color_space: str,
        input_icc_path: Optional[str],
        output_icc_path: Optional[str],
        monitor_icc_bytes: Optional[bytes] = None,
    ) -> Image.Image:
        """Soft-proof the preview into display space.

        For a paper/printer output profile, simulate the print on screen (paper white +
        ink) via a proof transform. For an export color space, do a gamut-only proof
        (relative colorimetric + BPC) ending at the display. ``display`` is the monitor
        profile when detected (``monitor_icc_bytes``), else sRGB. The output always
        lands in display space — otherwise it would leak output-space numbers to the
        screen and shift per output space (issue #243). The caller shows the result raw
        (no further display transform).
        """
        try:
            from negpy.infrastructure.display.color_mgmt import open_profile_from_bytes

            # littleCMS needs RGB against the RGB working/output profiles.
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            p_src = ImageProcessor._resolve_src_profile(working_color_space, input_icc_path)
            # Custom output profile, or the working space when only an input is set.
            p_dst = ImageProcessor._resolve_dst_profile(working_color_space, output_icc_path)
            if p_dst is None:
                return pil_img
            # Display the proof lands on: the monitor profile when detected, else sRGB.
            p_display = open_profile_from_bytes(monitor_icc_bytes) if monitor_icc_bytes else ImageCms.createProfile("sRGB")

            if ImageProcessor._is_print_profile(p_dst):
                # Paper/printer profile: simulate the print on screen (paper white + ink)
                # via a proof transform — relative-colorimetric source→paper, then
                # absolute-colorimetric paper→display so the paper white/Dmax show.
                # Handles RGB and CMYK paper profiles (proof space is internal).
                proof = ImageCms.buildProofTransform(
                    p_src,
                    p_display,
                    p_dst,
                    "RGB",
                    "RGB",
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    proofRenderingIntent=ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
                    flags=ImageCms.Flags.SOFTPROOFING,
                )
                result = ImageCms.applyTransform(pil_img, proof)
                return result if result is not None else pil_img

            # Export color space / display-class profile: gamut-only proof. GRAY
            # destinations need an "L" intermediate.
            dst_space = (getattr(p_dst.profile, "xcolor_space", "RGB ") or "RGB ").strip()
            out_mode = "L" if dst_space == "GRAY" else "RGB"
            result = ImageCms.profileToProfile(
                pil_img,
                p_src,
                p_dst,
                renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                outputMode=out_mode,
                flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
            )
            if result is None:
                return pil_img
            result = result if result.mode == "RGB" else result.convert("RGB")
            # Final output → display transform so the proof is shown in display space
            # rather than reinterpreted by the viewer. Always runs (not just when a
            # monitor profile is known): without it the proof would leak output-space
            # numbers to the screen and shift per output space (issue #243). Skipped for
            # GRAY outputs, whose `result` is no longer in `p_dst`'s space after RGB-ising.
            if out_mode == "RGB":
                proofed = ImageCms.profileToProfile(
                    result,
                    p_dst,
                    p_display,
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB",
                    flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
                )
                if proofed is not None:
                    result = proofed
            return result
        except Exception as e:
            logger.error(f"Soft-proof preview failed: {e}")
            return pil_img

    @staticmethod
    @lru_cache(maxsize=8)
    def soft_proof_lut(
        working_color_space: str,
        input_icc_path: Optional[str],
        output_icc_path: Optional[str],
        monitor_icc_bytes: Optional[bytes] = None,
        size: int = PROOF_LUT_SIZE,
    ) -> Optional[np.ndarray]:
        """``soft_proof_preview`` baked into an (N,N,N,3) LUT, for the preview only.

        Built by pushing the identity grid through ``soft_proof_preview`` itself, so
        the print-profile / export-space / GRAY branches cannot drift from it. Export
        keeps the exact per-pixel transform.
        """
        try:
            axis = np.linspace(0, 255, size).round().astype(np.uint8)
            r, g, b = np.meshgrid(axis, axis, axis, indexing="ij")
            grid = np.ascontiguousarray(np.stack((r, g, b), axis=-1)).reshape(size, size * size, 3)
            proofed = ImageProcessor.soft_proof_preview(
                Image.fromarray(grid, mode="RGB"),
                working_color_space,
                input_icc_path,
                output_icc_path,
                monitor_icc_bytes,
            )
            lut = np.asarray(proofed, dtype=np.float32).reshape(size, size, size, 3) / 255.0
            return np.ascontiguousarray(lut)
        except Exception as e:
            logger.warning("Soft-proof LUT build failed, falling back to the per-pixel transform", exc_info=e)
            return None

    def _save_to_pil_buffer(
        self,
        pil_img: Image.Image,
        buf: io.BytesIO,
        export_settings,
        icc_bytes: Optional[bytes],
    ) -> None:
        """Encodes PIL image to byte stream."""
        fmt = "JPEG" if export_settings.export_fmt == ExportFormat.JPEG else "TIFF"
        quality = getattr(export_settings, "jpeg_quality", 95)
        pil_img.save(
            buf,
            format=fmt,
            quality=quality,
            subsampling=0,
            dpi=(export_settings.export_dpi, export_settings.export_dpi),
            icc_profile=icc_bytes,
            compression="tiff_lzw" if fmt == "TIFF" else None,
        )

    def cleanup(self, release_source_cache: bool = True, collect: bool = True, retain: Any = None) -> None:
        """Evacuates transient GPU resources; ``retain`` survives the teardown."""
        if release_source_cache:
            self._source_cache_key = None
            self._source_cache_value = None
            self._precorrect_key = None
            self._precorrect_value = None
        if self.engine_gpu:
            self.engine_gpu.cleanup(collect=collect, retain=retain)

    def destroy_all(self) -> None:
        """Teardown GPU engine."""
        if self.engine_gpu:
            self.engine_gpu.destroy_all()
