"""Measure Optical Removal against the IR path on scans that carry an IR plane.

Dev tool, not shipped. Both paths feed the same fill, so the whole difference between them is
the score map each detector produces. The IR score is the baseline here: every IR component
the optical detector misses, and every mark it makes that IR does not, is listed and cropped.

  uv run python scripts/optical_vs_ir_evidence.py                       # samples/scans
  uv run python scripts/optical_vs_ir_evidence.py --files '/path/*.tif' --json out.json
  uv run python scripts/optical_vs_ir_evidence.py --compare baseline.json
  uv run python scripts/optical_vs_ir_evidence.py --min-pool             # erode before the resize
  uv run python scripts/optical_vs_ir_evidence.py --bake --top 4         # run both fills at preview scale

IR is a baseline, not truth: it misses bright light-piping scratches and carries a
misregistration skirt. "optical-only" is a review list, never a false-positive count.

Everything is measured on one detection plane (``--target`` long edge) so the masks are
pixel-aligned. Decoded planes are cached under ``debug/optical_vs_ir/cache``.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from negpy.kernel.system.config import APP_CONFIG  # noqa: E402

APP_CONFIG.use_gpu = False

from negpy.domain.models import WorkspaceConfig  # noqa: E402
from negpy.features.retouch.logic import (  # noqa: E402
    _IR_WRITE_HI,
    _is_hair,
    compute_dust_stats,
    detect_luma_score,
    downsample_ir,
    ir_defect_score,
    ir_detect_cutoff,
    ir_ratio_and_gain,
)
from negpy.features.retouch.models import RetouchConfig  # noqa: E402
from negpy.infrastructure.loaders.constants import is_ir_sidecar_path  # noqa: E402
from negpy.kernel.image.logic import working_oetf_encode  # noqa: E402
from negpy.services.rendering.image_processor import ImageProcessor, _downsample_to_long_edge  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "debug", "optical_vs_ir")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

THRESHOLDS = (0.2, 0.4, 0.66, 0.8, 0.95)
BUCKETS = ("small", "medium", "large", "hair")
BANDS = ("thin", "mid", "dense")
GREEN = np.array([0.22, 1.0, 0.08], np.float32)
MAGENTA = np.array([1.0, 0.0, 1.0], np.float32)


def planes(path: str, target: int, min_pool: bool) -> dict:
    """Detection-scale ``vis`` (plain INTER_AREA), ``small`` (what the optical detector sees),
    ``ratio`` and ``degenerate``. The IR ratio is always measured against ``vis`` so the truth
    does not move with ``--min-pool``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    cache = os.path.join(CACHE_DIR, f"{stem}_{target}_{int(min_pool)}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    img, ir, _ = ImageProcessor()._load_source_f32(path, WorkspaceConfig())
    if ir is None:
        return {"degenerate": np.array(True)}
    img = np.ascontiguousarray(img, dtype=np.float32)
    ir = np.ascontiguousarray(ir, dtype=np.float32)
    vis = np.ascontiguousarray(_downsample_to_long_edge(img, target), dtype=np.float32)
    small = np.ascontiguousarray(downsample_ir(img, target), dtype=np.float32) if min_pool else vis
    ratio, _, degenerate, _ = ir_ratio_and_gain(downsample_ir(ir, target), vis)
    out = {
        "vis": vis,
        "small": small,
        "ir": downsample_ir(ir, target),
        "ratio": ratio.astype(np.float32),
        "degenerate": np.array(degenerate),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, **out)
    return out


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    return cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0


def ir_components(truth: np.ndarray, band_plane: np.ndarray, excess: np.ndarray) -> tuple:
    """``(labels, rows, dropped)``: one row per IR component with area, bucket, band, centroid.
    Components on the frame border, and components where the visible density never rises
    above its surroundings, are rebate edges and misregistration skirts rather than dust;
    they are dropped and counted."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(truth.astype(np.uint8), connectivity=8)
    h, w = truth.shape
    peak = np.full(n, -1.0, np.float32)
    np.maximum.at(peak, labels.ravel(), excess.ravel())
    rows = []
    dropped = 0
    for i in range(1, n):
        x0, y0, bw, bh, area = (int(v) for v in stats[i])
        if x0 <= 1 or y0 <= 1 or x0 + bw >= w - 1 or y0 + bh >= h - 1 or peak[i] <= 0:
            labels[labels == i] = 0
            dropped += 1
            continue
        sub = labels[y0 : y0 + bh, x0 : x0 + bw] == i
        # ir_defect_score's 3x3 erode makes 9 px the smallest component.
        if _is_hair(sub, area):
            bucket = "hair"
        elif area <= 12:
            bucket = "small"
        elif area <= 40:
            bucket = "medium"
        else:
            bucket = "large"
        cx, cy = cents[i]
        d = float(band_plane[int(cy), int(cx)])
        band = "thin" if d < 0.33 else ("dense" if d > 0.66 else "mid")
        rows.append({"label": i, "area": area, "bucket": bucket, "band": band, "cx": float(cx), "cy": float(cy)})
    return labels, rows, dropped


def optical_masks(small: np.ndarray, thr: float, size: int, stats: tuple) -> tuple:
    """``(opt_any, opt_write)`` — every pixel the score touched, and every pixel the fill writes."""
    score, hair = detect_luma_score(small, thr, size, stats=stats)
    h, w = small.shape[:2]
    opt_any = np.zeros((h, w), bool)
    opt_write = np.zeros((h, w), bool)
    if score is not None:
        opt_any |= score < 1.0
        opt_write |= score < _IR_WRITE_HI
    if hair is not None:
        opt_any |= hair > 0
        opt_write |= hair > 0
    return opt_any, opt_write


def score_frame(
    truth: np.ndarray,
    labels: np.ndarray,
    rows: list,
    dropped: int,
    opt_any: np.ndarray,
    opt_write: np.ndarray,
    w_std: np.ndarray,
    ratio: np.ndarray,
) -> dict:
    hit_labels = set(np.unique(labels[_dilate(opt_any, 1) & truth]).tolist())
    n_lab = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=n_lab)
    covered = np.bincount(labels[opt_write & truth].ravel(), minlength=n_lab)
    per = {b: {"n": 0, "hit": 0} for b in BUCKETS}
    grid = {f"{b}/{d}": {"n": 0, "hit": 0} for b in BUCKETS for d in BANDS}
    coverage = []
    for r in rows:
        hit = r["label"] in hit_labels
        r["hit"] = hit
        for cell in (per[r["bucket"]], grid[f"{r['bucket']}/{r['band']}"]):
            cell["n"] += 1
            cell["hit"] += int(hit)
        if hit:
            r["coverage"] = float(covered[r["label"]] / max(areas[r["label"]], 1))
            coverage.append(r["coverage"])
    # Optical-only: whole optical components that touch no IR truth (a pixel subtraction
    # would count the pad ring around every real hit).
    n_opt, lab_opt, st_opt, cents_opt = cv2.connectedComponentsWithStats(opt_any.astype(np.uint8), connectivity=8)
    touches = np.zeros(n_opt, bool)
    touches[np.unique(lab_opt[_dilate(truth, 2)])] = True
    # The IR ratio under each optical-only mark: a dip short of the cutoff is dust IR saw but
    # did not call; ~1.0 means IR saw nothing there.
    ratio_min = np.full(n_opt, 2.0, np.float32)
    np.minimum.at(ratio_min, lab_opt.ravel(), ratio.ravel())
    only_rows = [
        {"area": int(st_opt[i, cv2.CC_STAT_AREA]), "cx": float(cents_opt[i][0]), "cy": float(cents_opt[i][1]), "ratio": float(ratio_min[i])}
        for i in range(1, n_opt)
        if not touches[i]
    ]
    only = ~touches[lab_opt] & opt_any
    missed_px = truth & ~opt_any
    return {
        "ir_components": len(rows),
        "ir_dropped": dropped,
        "component_recall": sum(r["hit"] for r in rows) / max(len(rows), 1),
        "pixel_recall": float((truth & opt_any).sum() / max(truth.sum(), 1)),
        "buckets": per,
        "grid": grid,
        "coverage": coverage,
        "coverage_median": float(np.median(coverage)) if coverage else float("nan"),
        "coverage_p10": float(np.percentile(coverage, 10)) if coverage else float("nan"),
        "optical_only_count": len(only_rows),
        "optical_only_area": int(only.sum()),
        "optical_only_ir_faint": sum(r["ratio"] < 0.9 for r in only_rows),
        "optical_only_clean_area": sum(r["area"] for r in only_rows if r["ratio"] >= 0.9),
        "optical_only_rows": only_rows,
        "w_std": {
            "missed_p50": float(np.median(w_std[missed_px])) if missed_px.any() else float("nan"),
            "missed_p90": float(np.percentile(w_std[missed_px], 90)) if missed_px.any() else float("nan"),
            "only_p50": float(np.median(w_std[only])) if only.any() else float("nan"),
            "only_p90": float(np.percentile(w_std[only], 90)) if only.any() else float("nan"),
            "frame_p50": float(np.median(w_std)),
            "frame_p90": float(np.percentile(w_std, 90)),
        },
    }


def _encode(img: np.ndarray) -> np.ndarray:
    return np.asarray(working_oetf_encode(np.clip(np.ascontiguousarray(img, dtype=np.float32), 0.0, 1.0)))


def _wash(enc: np.ndarray, mask: np.ndarray, color: np.ndarray) -> np.ndarray:
    out = enc.copy()
    out[mask] = out[mask] * 0.55 + color * 0.45
    return out


def _save(path: str, panels: list) -> None:
    gap = np.ones((panels[0].shape[0], 4, 3), np.float32)
    strip = np.concatenate([p for pair in zip(panels, [gap] * len(panels)) for p in pair][:-1], axis=1)
    cv2.imwrite(path, (np.clip(strip[:, :, ::-1], 0, 1) * 255 + 0.5).astype(np.uint8))


def _window(cx: float, cy: float, shape: tuple, size: int) -> tuple:
    h, w = shape[:2]
    size = min(size, h, w)
    y0 = int(np.clip(round(cy) - size // 2, 0, h - size))
    x0 = int(np.clip(round(cx) - size // 2, 0, w - size))
    return y0, y0 + size, x0, x0 + size


def dump_crops(
    stem: str, vis: np.ndarray, truth: np.ndarray, opt_any: np.ndarray, rows: list, only_rows: list, top: int, size: int
) -> None:
    enc = _encode(vis)
    ir_wash = _wash(enc, truth, MAGENTA)
    opt_wash = _wash(enc, opt_any, GREEN)
    misses = sorted((r for r in rows if not r["hit"]), key=lambda r: -r["area"])[:top]
    only = sorted(only_rows, key=lambda r: -r["area"])[:top]
    for tag, items in (("miss", misses), ("only", only)):
        for k, r in enumerate(items):
            y0, y1, x0, x1 = _window(r["cx"], r["cy"], vis.shape, size)
            name = (
                f"{stem}_{tag}{k}_{r.get('bucket', '')}{r.get('band', '')}_a{r['area']}"
                + (f"_r{r['ratio']:.2f}" if "ratio" in r else "")
                + ".png"
            )
            _save(os.path.join(OUT_DIR, name), [enc[y0:y1, x0:x1], ir_wash[y0:y1, x0:x1], opt_wash[y0:y1, x0:x1]])


def bake_ab(
    stem: str,
    small: np.ndarray,
    ir_det: np.ndarray,
    truth: np.ndarray,
    rows: list,
    thr: float,
    size: int,
    ir_thr: float,
    top: int,
    crop: int,
) -> dict:
    """Both fills at detection scale; how much of the IR repair the optical path reproduces."""
    proc = ImageProcessor()
    cfg = WorkspaceConfig(retouch=RetouchConfig(dust_remove=True, dust_threshold=thr, dust_size=size))
    score, hairs = proc._detect_luma(cfg, small, stem)
    out_opt = np.asarray(proc._luma_bake(small, score, stem), dtype=np.float32)
    if hairs:
        out_opt = np.asarray(proc._hair_inpaint(out_opt, hairs, stem + "h"), dtype=np.float32)
    cfg_ir = WorkspaceConfig(retouch=RetouchConfig(ir_dust_remove=True, ir_threshold=ir_thr))
    out_ir, _, degenerate, routed = proc._ir_bake(small, ir_det, cfg_ir, stem + "ir")
    out_ir = np.asarray(out_ir, dtype=np.float32)
    if routed is not None:
        out_ir = np.asarray(proc._hair_inpaint(out_ir, [routed], stem + "irh"), dtype=np.float32)
    lum = lambda a: np.clip(a.mean(axis=2), 1e-4, None)  # noqa: E731
    fix_ir = np.abs(np.log10(lum(out_ir) / lum(small)))[truth]
    gap = np.abs(np.log10(lum(out_opt) / lum(out_ir)))[truth]
    m = {
        "ir_correction_dex_rms": float(np.sqrt(np.mean(fix_ir**2))),
        "optical_vs_ir_dex_rms": float(np.sqrt(np.mean(gap**2))),
        "achieved_fraction": float(1.0 - gap.sum() / max(fix_ir.sum(), 1e-9)),
    }
    enc_s, enc_i, enc_o = _encode(small), _encode(out_ir), _encode(out_opt)
    for k, r in enumerate(sorted(rows, key=lambda r: -r["area"])[:top]):
        y0, y1, x0, x1 = _window(r["cx"], r["cy"], small.shape, crop)
        _save(
            os.path.join(OUT_DIR, f"{stem}_bake{k}_{r['bucket']}_a{r['area']}.png"),
            [enc_s[y0:y1, x0:x1], enc_i[y0:y1, x0:x1], enc_o[y0:y1, x0:x1]],
        )
    return m


def aggregate(frames: dict) -> dict:
    agg = {}
    for thr in THRESHOLDS:
        key = f"{thr}"
        per = {b: {"n": 0, "hit": 0} for b in BUCKETS}
        grid = {f"{b}/{d}": {"n": 0, "hit": 0} for b in BUCKETS for d in BANDS}
        cov, n, hit, only_n, only_a, only_f, only_ca = [], 0, 0, 0, 0, 0, 0
        for f in frames.values():
            s = f["sweep"][key]
            n += s["ir_components"]
            hit += round(s["component_recall"] * s["ir_components"])
            cov += s["coverage"]
            only_n += s["optical_only_count"]
            only_a += s["optical_only_area"]
            only_f += s["optical_only_ir_faint"]
            only_ca += s["optical_only_clean_area"]
            for b in BUCKETS:
                per[b]["n"] += s["buckets"][b]["n"]
                per[b]["hit"] += s["buckets"][b]["hit"]
            for g in grid:
                grid[g]["n"] += s["grid"][g]["n"]
                grid[g]["hit"] += s["grid"][g]["hit"]
        agg[key] = {
            "ir_components": n,
            "component_recall": hit / max(n, 1),
            "buckets": {b: v["hit"] / max(v["n"], 1) for b, v in per.items()},
            "bucket_n": {b: v["n"] for b, v in per.items()},
            "grid": {g: (v["hit"] / v["n"] if v["n"] else float("nan")) for g, v in grid.items()},
            "grid_n": {g: v["n"] for g, v in grid.items()},
            "coverage_median": float(np.median(cov)) if cov else float("nan"),
            "coverage_p10": float(np.percentile(cov, 10)) if cov else float("nan"),
            "optical_only_count": only_n,
            "optical_only_area": only_a,
            "optical_only_ir_faint": only_f,
            "optical_only_clean_area": only_ca,
        }
    return agg


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def report(frames: dict, agg: dict, default_thr: float, baseline: dict | None) -> None:
    head = (
        f"{'thr':>5}{'ncomp':>7}{'recall':>8}"
        + "".join(f"{b:>8}" for b in BUCKETS)
        + f"{'cov50':>7}{'cov10':>7}{'only#':>7}{'onlyA':>8}{'faint':>7}{'cleanA':>8}"
    )
    for stem, f in frames.items():
        print(f"\n== {stem}")
        print(head)
        for thr in THRESHOLDS:
            s = f["sweep"][f"{thr}"]
            cells = "".join(
                f"{(s['buckets'][b]['hit'] / s['buckets'][b]['n']) if s['buckets'][b]['n'] else float('nan'):>8.2f}" for b in BUCKETS
            )
            print(
                f"{thr:>5}{s['ir_components']:>7}{s['component_recall']:>8.2f}{cells}{s['coverage_median']:>7.2f}{s['coverage_p10']:>7.2f}{s['optical_only_count']:>7}{s['optical_only_area']:>8}{s['optical_only_ir_faint']:>7}{s['optical_only_clean_area']:>8}"
            )
        w = f["sweep"][f"{default_thr}"]["w_std"]
        print(
            f"   w_std p50/p90  missed {w['missed_p50']:.3f}/{w['missed_p90']:.3f}  only {w['only_p50']:.3f}/{w['only_p90']:.3f}  frame {w['frame_p50']:.3f}/{w['frame_p90']:.3f}"
        )
        if "bake" in f:
            print(
                f"   bake: IR correction {f['bake']['ir_correction_dex_rms']:.3f} dex rms, optical vs IR {f['bake']['optical_vs_ir_dex_rms']:.3f}, achieved {f['bake']['achieved_fraction']:.2f}"
            )

    print("\n== AGGREGATE (pooled components)")
    print(head)
    for thr in THRESHOLDS:
        a = agg[f"{thr}"]
        cells = "".join(f"{a['buckets'][b]:>8.2f}" for b in BUCKETS)
        print(
            f"{thr:>5}{a['ir_components']:>7}{a['component_recall']:>8.2f}{cells}{a['coverage_median']:>7.2f}{a['coverage_p10']:>7.2f}{a['optical_only_count']:>7}{a['optical_only_area']:>8}{a['optical_only_ir_faint']:>7}{a['optical_only_clean_area']:>8}"
        )
    a = agg[f"{default_thr}"]
    print("\n   bucket n: " + "  ".join(f"{b}={a['bucket_n'][b]}" for b in BUCKETS))
    print(f"   recall by bucket/band at thr {default_thr}:")
    print(f"{'':>8}" + "".join(f"{d:>14}" for d in BANDS))
    for b in BUCKETS:
        print(f"{b:>8}" + "".join(f"{a['grid'][f'{b}/{d}']:>9.2f}({a['grid_n'][f'{b}/{d}']:>3})" for d in BANDS))
    if baseline:
        print(f"\n== vs baseline at thr {default_thr}")
        b0 = baseline["aggregate"][f"{default_thr}"]
        for k in (
            "component_recall",
            "coverage_median",
            "coverage_p10",
            "optical_only_count",
            "optical_only_area",
            "optical_only_ir_faint",
            "optical_only_clean_area",
        ):
            print(f"   {k:<20}{_fmt(b0.get(k, float('nan'))):>10} -> {_fmt(a[k]):>10}")
        for b in BUCKETS:
            print(f"   {'recall ' + b:<20}{b0['buckets'][b]:>10.2f} -> {a['buckets'][b]:>10.2f}")
    print(f"\ncrops: {OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", default=[os.path.join(ROOT, "samples", "scans", "*.tif")], help="globs; IR sidecars are skipped")
    ap.add_argument("--target", type=int, default=APP_CONFIG.preview_render_size, help="detection long edge")
    ap.add_argument("--min-pool", action="store_true", help="erode by the resample footprint before the optical downsample")
    ap.add_argument("--size", type=int, default=RetouchConfig.dust_size)
    ap.add_argument("--threshold", type=float, default=RetouchConfig.dust_threshold, help="slider value the crops and bake use")
    ap.add_argument("--ir-threshold", type=float, default=RetouchConfig.ir_threshold)
    ap.add_argument("--top", type=int, default=8, help="crops per frame per list")
    ap.add_argument("--crop", type=int, default=160)
    ap.add_argument("--bake", action="store_true", help="also run both fills at detection scale")
    ap.add_argument("--json", help="write the per-frame and aggregate numbers here")
    ap.add_argument("--compare", help="a --json file from an earlier run to diff the aggregate against")
    args = ap.parse_args()

    paths = sorted(p for g in args.files for p in glob.glob(g) if not is_ir_sidecar_path(p))
    if not paths:
        sys.exit("no files")
    os.makedirs(OUT_DIR, exist_ok=True)
    thresholds = tuple(sorted(set(THRESHOLDS) | {args.threshold}))
    globals()["THRESHOLDS"] = thresholds

    frames = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        p = planes(path, args.target, args.min_pool)
        if bool(p["degenerate"]):
            print(f"{stem}: no usable IR plane, skipped")
            continue
        truth = ir_defect_score(p["ratio"], ir_detect_cutoff(args.ir_threshold, True)) <= 0.05
        stats = compute_dust_stats(p["small"], args.size)
        labels, rows, dropped = ir_components(truth, stats[1], stats[0] - stats[1])
        truth = labels > 0
        sweep = {}
        for thr in thresholds:
            opt_any, opt_write = optical_masks(p["small"], thr, args.size, stats)
            sweep[f"{thr}"] = score_frame(
                truth, labels, [dict(r) for r in rows] if thr != args.threshold else rows, dropped, opt_any, opt_write, stats[3], p["ratio"]
            )
            if thr == args.threshold:
                dump_crops(stem, p["vis"], truth, opt_any, rows, sweep[f"{thr}"]["optical_only_rows"], args.top, args.crop)
        f = {"sweep": sweep}
        if args.bake:
            f["bake"] = bake_ab(stem, p["small"], p["ir"], truth, rows, args.threshold, args.size, args.ir_threshold, args.top, args.crop)
        frames[stem] = f
        s = sweep[f"{args.threshold}"]
        print(
            f"{stem}: {s['ir_components']} IR comps ({s['ir_dropped']} dropped), recall {s['component_recall']:.2f}, cov50 {s['coverage_median']:.2f}, only {s['optical_only_count']}"
        )

    agg = aggregate(frames)
    baseline = None
    if args.compare:
        with open(args.compare) as fh:
            baseline = json.load(fh)
    report(frames, agg, args.threshold, baseline)
    if args.json:
        slim = {
            k: {
                "sweep": {
                    t: {kk: vv for kk, vv in s.items() if kk not in ("coverage", "optical_only_rows")} for t, s in f["sweep"].items()
                },
                **({"bake": f["bake"]} if "bake" in f else {}),
            }
            for k, f in frames.items()
        }
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "frames": slim, "aggregate": agg}, fh, indent=1)


if __name__ == "__main__":
    main()
