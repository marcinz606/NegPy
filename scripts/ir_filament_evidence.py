"""Measure NegPy's IR dust repair on filament (hair/scratch) defects.

Dev tool, not shipped. The IR bake is judged by eye on the canvas, which makes
filament regressions invisible until they ship; this reproduces the bake outside
the GUI and puts numbers on it.

Two modes:

  # synthetic frame with ground truth (the only mode that can score accuracy)
  uv run python scripts/ir_filament_evidence.py

  # a real scan carrying an IR channel (no ground truth: coverage + grain only)
  uv run python scripts/ir_filament_evidence.py --file /path/to/scan.tif

Runs the bake twice, mirroring the app: once on a preview-scale buffer (mask 1:1
with the pixels) and once full-res (detection stays pinned at
``preview_render_size``, so the mask is upsampled onto the buffer). Crops land in
``debug/ir_filament/``.

Filament and speck are scored separately: they take different routes through
``route_ir_defects``, and only the filament is the problem.

Metrics
-------
overrepair    repaired area / true defect area. 1.0 is exact; a filament blurred
              into a wide band scores several times that.
grain         high-pass sigma inside the repair against the same measure on
              ground truth (on a real scan, against a clean annulus). 1.0 keeps
              grain, ~0 is the "plastic worm" a mean-based fill leaves. Reads
              *above* 1.0 when the repair leaves residual defect structure, so
              read it next to rmse, not alone.
rmse          against ground truth over the defect core, and separately over the
              part of the core crossing an image edge (where an isotropic fill
              bridges instead of continuing the edge). Synthetic only; `src`
              is the do-nothing baseline to beat.
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from negpy.kernel.system.config import APP_CONFIG  # noqa: E402

# No GPU work happens in the IR bake, and initializing it costs seconds per run.
APP_CONFIG.use_gpu = False

from negpy.domain.models import WorkspaceConfig  # noqa: E402
from negpy.features.retouch.logic import downsample_ir, ir_defect_score, ir_detect_cutoff, ir_ratio_and_gain  # noqa: E402
from negpy.features.retouch.models import RetouchConfig  # noqa: E402
from negpy.kernel.image.logic import working_oetf_encode  # noqa: E402
from negpy.services.rendering.image_processor import ImageProcessor, _detection_downsample  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug", "ir_filament")

# Synthetic fixture: physical attenuation of a neutral-density defect. The visible
# planes follow the IR transmittance raised to the refraction gammas the bake fits,
# so the division tier has something real to recover.
_SYNTH_GAMMAS = (1.2, 1.4, 1.55)
_SYNTH_IR_CLEAN = 0.90
_SYNTH_GRAIN_SIGMA = 0.013


def synthetic_frame(long_edge: int, seed: int = 7):
    """Film-like frame with three defects that each take a different route: an opaque
    3 px filament (scores at floor, routes to the inpaint), a semi-transparent one
    (partial score, so the fill owns it whatever the routing does), and a round speck
    as the compact control. All three cross the same hard tone edges.

    Returns ``(clean, source, ir, regions, edges)`` at full res: ground truth, the
    scan the bake sees, the IR plane, per-defect opacity maps in [0, 1], and the
    image edges the filaments have to be continued across.
    """
    h, w = int(long_edge * 2 / 3), int(long_edge)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    base = 0.28 + 0.30 * (xx / w) + 0.12 * np.sin(2.0 * np.pi * yy / h)
    edges = np.zeros((h, w), dtype=np.uint8)
    # Hard tone steps at three angles: the filament crosses each one, so a fill that
    # averages a disc bridges them and a structure-following one does not.
    for (x0, y0), (x1, y1), step in (
        ((int(0.30 * w), 0), (int(0.30 * w), h), 0.22),
        ((0, int(0.55 * h)), (w, int(0.40 * h)), -0.18),
        ((int(0.55 * w), 0), (w, int(0.85 * h)), 0.15),
    ):
        side = np.zeros((h, w), dtype=np.uint8)
        cv2.line(side, (x0, y0), (x1, y1), 1, 1)
        cv2.line(edges, (x0, y0), (x1, y1), 1, 3)
        # Fill one side of the line by flooding from a corner it does not touch.
        filled = side.copy()
        cv2.floodFill(filled, np.zeros((h + 2, w + 2), np.uint8), (0, h - 1), 2)
        base = base + step * (filled == 2).astype(np.float32)

    clean = np.repeat(base[..., None], 3, axis=2).astype(np.float32)
    clean[:, :, 0] *= 1.04
    clean[:, :, 2] *= 0.93
    # Grain: white noise blurred to a grain size, so a high-pass sigma actually
    # measures texture rather than per-pixel sensor noise.
    grain = cv2.GaussianBlur(rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32), (0, 0), 0.8)
    clean = np.clip(clean + grain * (_SYNTH_GRAIN_SIGMA / max(float(grain.std()), 1e-6)), 0.01, 1.0)

    def psf(core: np.ndarray) -> np.ndarray:
        return np.clip(cv2.GaussianBlur(core, (0, 0), 0.9) * 1.4, 0.0, 1.0)

    def curve(y_off: float, opacity: float) -> np.ndarray:
        core = np.zeros((h, w), dtype=np.float32)
        t = np.linspace(0.0, 1.0, 900)
        px = (0.12 + 0.76 * t) * w
        py = (y_off + 0.42 * np.sin(2.4 * np.pi * t) * t) * h
        cv2.polylines(core, [np.stack([px, py], -1).astype(np.int32)], False, 1.0, 3, cv2.LINE_AA)
        return psf(core) * opacity

    spk = np.zeros((h, w), dtype=np.float32)
    cv2.circle(spk, (int(0.20 * w), int(0.80 * h)), max(3, long_edge // 300), 1.0, -1)
    regions = {"opaque-fil": curve(0.30, 1.0), "faint-fil": curve(0.62, 0.45), "speck": psf(spk)}
    alpha = np.maximum.reduce(list(regions.values()))

    t_ir = 1.0 - 0.85 * alpha
    ir = (_SYNTH_IR_CLEAN * t_ir).astype(np.float32)
    source = clean.copy()
    for c in range(3):
        source[:, :, c] *= t_ir ** _SYNTH_GAMMAS[c]
    return clean, source.astype(np.float32), ir, regions, edges


def _detected_alpha(img: np.ndarray, ir: np.ndarray, threshold: float) -> np.ndarray:
    """Stand-in for ground-truth opacity on a real scan: the at-floor IR score,
    which is what the bake itself calls a defect."""
    ratio, _, _, _ = ir_ratio_and_gain(downsample_ir(ir, APP_CONFIG.preview_render_size), _detection_downsample(img))
    score = ir_defect_score(ratio, ir_detect_cutoff(threshold, True))
    at_floor = (score <= 0.05).astype(np.float32)
    return at_floor if at_floor.shape == img.shape[:2] else cv2.resize(at_floor, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)


def load_real(path: str):
    """``(source, ir)`` from a scan file, or exit if it carries no IR channel."""
    proc = ImageProcessor()
    img, ir, _ = proc._load_source_f32(path, WorkspaceConfig())
    if ir is None:
        sys.exit(f"{path} carries no IR channel — the IR bake would be skipped entirely.")
    return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(ir, dtype=np.float32)


def bake(img: np.ndarray, ir: np.ndarray, threshold: float, key: str):
    """Run the real ``_ir_bake`` + ``_hair_inpaint`` sequence. ``(out, routed)``."""
    proc = ImageProcessor()
    cfg = WorkspaceConfig(retouch=RetouchConfig(ir_dust_remove=True, ir_attenuation=True, ir_threshold=threshold))
    out, _, degenerate, routed = proc._ir_bake(img, ir, cfg, key)
    if degenerate:
        sys.exit("IR plane read as degenerate (B&W/Kodachrome ghost) — the bake self-skipped.")
    if routed is not None:
        out = proc._hair_inpaint(out, [routed], key)
    return np.ascontiguousarray(out, dtype=np.float32), routed


def _highpass_sigma(img: np.ndarray, sel: np.ndarray) -> float:
    if not sel.any():
        return float("nan")
    lum = img.mean(axis=2) if img.ndim == 3 else img
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 1.5)
    return float(hp[sel].std())


def _grow(mask: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0


def measure(src, out, clean, alpha, other, edges, routed, scale_px: float) -> dict:
    """Coverage, grain and (with ground truth) accuracy over one defect.

    ``other`` masks the frame's *other* defects out of every selection, so a speck
    can't contaminate the filament's numbers or vice versa.
    """
    r = max(1, int(round(scale_px)))
    elsewhere = ~_grow(other > 0.05, 4 * r)
    truth = (alpha > 0.05) & elsewhere
    # Core relative to this defect's own peak opacity — a semi-transparent filament
    # never reaches an absolute 0.5 but still has a centre line.
    core = (alpha >= 0.5 * float(alpha.max())) & elsewhere
    changed = (np.abs(out - src).max(axis=2) > 0.002) & _grow(truth, 12 * r) & elsewhere
    # Grain reference: ground truth in the same pixels, or (real scan) a clean annulus.
    ref = (
        _highpass_sigma(clean, core) if clean is not None else _highpass_sigma(out, _grow(truth, 8 * r) & ~_grow(truth, 2 * r) & elsewhere)
    )
    m = {
        "defect_px": int(truth.sum()),
        "repaired_px": int(changed.sum()),
        "overrepair": float(changed.sum() / max(truth.sum(), 1)),
        "grain": _highpass_sigma(out, core) / max(ref, 1e-9),
        "routed": bool(routed is not None and routed.any() and (_resize_mask(routed, alpha.shape) & core).any()),
    }
    if clean is None:
        return m

    def rmse(sel, a):
        return float(np.sqrt(np.mean((a[sel] - clean[sel]) ** 2))) if sel.any() else float("nan")

    near_edge = core & (_grow(edges > 0, 6 * r))
    m.update(
        rmse_src=rmse(core, src),
        rmse_out=rmse(core, out),
        rmse_edge_src=rmse(near_edge, src),
        rmse_edge_out=rmse(near_edge, out),
        residual_max=float(np.abs(out[core] - clean[core]).max()) if core.any() else float("nan"),
    )
    return m


def _resize_mask(mask: np.ndarray, shape: tuple) -> np.ndarray:
    if mask.shape[:2] == tuple(shape[:2]):
        return mask > 0
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def _png(path: str, img: np.ndarray) -> None:
    enc = np.asarray(working_oetf_encode(np.clip(np.ascontiguousarray(img, dtype=np.float32), 0.0, 1.0)))
    if enc.ndim == 3:
        enc = enc[:, :, ::-1]
    cv2.imwrite(path, (np.clip(enc, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


def dump(tag: str, crop, src, out, clean, routed) -> None:
    """Side-by-side crop around the busiest stretch of filament."""
    y0, y1, x0, x1 = crop
    os.makedirs(OUT_DIR, exist_ok=True)
    panels = [src[y0:y1, x0:x1], out[y0:y1, x0:x1]]
    if clean is not None:
        panels.append(clean[y0:y1, x0:x1])
    gap = np.ones((y1 - y0, 4, 3), dtype=np.float32)
    _png(os.path.join(OUT_DIR, f"{tag}.png"), np.concatenate([p for pair in zip(panels, [gap] * len(panels)) for p in pair][:-1], axis=1))
    if routed is not None:
        _png(os.path.join(OUT_DIR, f"{tag}_routed.png"), routed.astype(np.float32))


def _crop_window(alpha: np.ndarray, size: int) -> tuple:
    """Window centred on the densest defect neighbourhood."""
    h, w = alpha.shape
    size = min(size, h, w)
    dens = cv2.boxFilter((alpha > 0.05).astype(np.float32), -1, (size | 1, size | 1))
    cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return y0, y0 + size, x0, x0 + size


_REPORT_KEYS = ("defect_px", "repaired_px", "overrepair", "grain", "rmse_src", "rmse_out", "rmse_edge_src", "rmse_edge_out", "residual_max")


def _report(rows: list) -> None:
    keys = [k for k in _REPORT_KEYS if any(k in m for _, _, m in rows)]
    print(f"\n{'run':<14}{'defect':<11}{'routed':<8}" + "".join(f"{k:>14}" for k in keys))
    print("-" * (33 + 14 * len(keys)))
    for run, defect, m in rows:
        cells = "".join(f"{m[k]:>14.4g}" if k in m else f"{'-':>14}" for k in keys)
        print(f"{run:<14}{defect:<11}{('yes' if m['routed'] else 'no'):<8}{cells}")
    print(
        "\noverrepair 1.0 = exact, high = defect smeared into a band."
        "\ngrain 1.0 = texture matches ground truth; ~0 = smooth 'plastic worm', >1 = residual defect."
        "\nrmse_out must beat rmse_src (do nothing); rmse_edge_* isolates edge crossings."
        f"\ncrops: {OUT_DIR}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="scan with an IR channel; omitted = synthetic frame with ground truth")
    ap.add_argument("--threshold", type=float, default=RetouchConfig.ir_threshold, help="IR Threshold slider value")
    ap.add_argument("--long-edge", type=int, default=4800, help="synthetic frame long edge (px)")
    ap.add_argument("--crop", type=int, default=420, help="dumped crop size (px)")
    args = ap.parse_args()

    if args.file:
        src_full, ir_full = load_real(args.file)
        clean_full, edges_full = None, np.zeros(src_full.shape[:2], np.uint8)
        # No ground truth: the "defect" is whatever the detector itself scores at floor.
        regions_full = {"detected": _detected_alpha(src_full, ir_full, args.threshold)}
    else:
        clean_full, src_full, ir_full, regions_full, edges_full = synthetic_frame(args.long_edge)

    rows = []
    for name, scale in (("preview", 1.0), ("full-res", max(1.0, max(src_full.shape[:2]) / APP_CONFIG.preview_render_size))):
        shrink = scale == 1.0
        img = _detection_downsample(src_full) if shrink else src_full
        ir = downsample_ir(ir_full, APP_CONFIG.preview_render_size) if shrink else ir_full
        clean = None if clean_full is None else (_detection_downsample(clean_full) if shrink else clean_full)
        edges = (_detection_downsample(edges_full) if shrink else edges_full) > 0.3
        regions = {k: (_detection_downsample(v) if shrink else v) for k, v in regions_full.items()}

        out, routed = bake(img, ir, args.threshold, name)
        for defect, alpha in regions.items():
            other = np.maximum.reduce([v for k, v in regions.items() if k != defect]) if len(regions) > 1 else np.zeros_like(alpha)
            rows.append((name, defect, measure(img, out, clean, alpha, other, edges, routed, scale)))
            dump(f"{name}_{defect}", _crop_window(alpha, int(args.crop * scale)), img, out, clean, routed)
    _report(rows)


if __name__ == "__main__":
    main()
