//! The panorama gain compensation of negpy/features/stitch/logic.py.

use rayon::prelude::*;

/// `gain_compensate`'s per-channel gains: the float64 mean of `reference` over the overlap
/// over that of `moving`, rows added in order. `None` under `min_overlap` pixels or for a
/// mean under 1e-6, where the part stays as it is.
pub fn compensation_gains(
    reference: &[f32],
    ref_mask: &[bool],
    moving: &[f32],
    mov_mask: &[bool],
    min_overlap: usize,
) -> Option<[f32; 3]> {
    let (mut a, mut b, mut n) = ([0f64; 3], [0f64; 3], 0usize);
    for (p, (&r, &m)) in ref_mask.iter().zip(mov_mask).enumerate() {
        if r && m {
            for c in 0..3 {
                a[c] += reference[3 * p + c] as f64;
                b[c] += moving[3 * p + c] as f64;
            }
            n += 1;
        }
    }
    if n < min_overlap {
        return None;
    }
    let (a, b) = (a.map(|v| v / n as f64), b.map(|v| v / n as f64));
    if b.iter().any(|&v| v < 1e-6) {
        return None;
    }
    Some([0, 1, 2].map(|c| (a[c] / b[c]) as f32))
}

/// One part's `seam_blend` weight: a cross-fade ramp of width `2 * feather` on the signed
/// distance to the cut, times a ramp of width `feather` up from the part's coverage edge, in
/// float32. The two distances to the cut run at once; the coverage distance runs after them.
pub fn seam_weight(
    seam: &[bool],
    valid: &[bool],
    (h, w): (usize, usize),
    feather: f64,
) -> Vec<f32> {
    use crate::distance::{Metric, distance_transform};
    use rayon::prelude::*;
    const BLOCK: usize = 1 << 14;
    let dist = |m: &[bool], inside: bool| {
        let src: Vec<u8> = m.par_iter().map(|&v| (v == inside) as u8).collect();
        distance_transform(&src, h, w, Metric::L2Mask3)
    };
    let span = (2.0 * feather) as f32;
    let (mut weight, beyond) = rayon::join(|| dist(seam, true), || dist(seam, false));
    weight
        .par_chunks_mut(BLOCK)
        .zip(beyond.par_chunks(BLOCK).zip(seam.par_chunks(BLOCK)))
        .for_each(move |(wt, (b, s))| {
            for ((wt, &b), &s) in wt.iter_mut().zip(b).zip(s) {
                *wt = if s {
                    (*wt / span + 0.5).clamp(0.0, 1.0)
                } else {
                    (0.5 - b / span).clamp(0.0, 1.0)
                };
            }
        });
    drop(beyond);
    let edge = dist(valid, true);
    let feather = feather as f32;
    crate::par_zip(&mut weight, &edge, move |wt, &e| {
        *wt *= (e / feather).clamp(0.0, 1.0)
    });
    weight
}

/// The weighted mean of HxWx3 parts, the weight of part `i` from `weight(i)`: parts added in
/// order as NumPy adds `img * weight[..., None]`, over `max(weight sum, 1e-6)`. One weight plane
/// is alive at a time.
fn weighted_mean(images: &[&[f32]], n: usize, weight: impl Fn(usize) -> Vec<f32>) -> Vec<f32> {
    use rayon::prelude::*;
    const BLOCK: usize = 1 << 13;
    let mut acc = vec![0f32; 3 * n];
    let mut sum = vec![0f32; n];
    for (i, img) in images.iter().enumerate() {
        let wt = weight(i);
        acc.par_chunks_mut(3 * BLOCK)
            .zip(sum.par_chunks_mut(BLOCK))
            .zip(img.par_chunks(3 * BLOCK).zip(wt.par_chunks(BLOCK)))
            .for_each(move |((a, s), (px, wt))| {
                for (((a, s), px), &wt) in a
                    .as_chunks_mut::<3>()
                    .0
                    .iter_mut()
                    .zip(s)
                    .zip(px.as_chunks::<3>().0)
                    .zip(wt)
                {
                    for c in 0..3 {
                        a[c] += px[c] * wt;
                    }
                    *s += wt;
                }
            });
    }
    let floor = 1e-6f32;
    acc.par_chunks_mut(3 * BLOCK)
        .zip(sum.par_chunks(BLOCK))
        .for_each(move |(a, s)| {
            for (a, &s) in a.as_chunks_mut::<3>().0.iter_mut().zip(s) {
                let d = if floor > s { floor } else { s };
                for v in a {
                    *v /= d;
                }
            }
        });
    acc
}

/// `feather_blend`: the parts weighted by the distance to their own coverage edge.
pub fn feather_blend(images: &[&[f32]], masks: &[&[bool]], (h, w): (usize, usize)) -> Vec<f32> {
    use crate::distance::{Metric, distance_transform};
    use rayon::prelude::*;
    weighted_mean(images, h * w, |i| {
        let src: Vec<u8> = masks[i].par_iter().map(|&v| v as u8).collect();
        distance_transform(&src, h, w, Metric::L2Mask3)
    })
}

/// `seam_blend`: the parts weighted by `seam_weight` with a band of `feather`.
pub fn seam_blend(
    images: &[&[f32]],
    masks: &[&[bool]],
    seams: &[&[bool]],
    (h, w): (usize, usize),
    feather: f64,
) -> Vec<f32> {
    weighted_mean(images, h * w, |i| {
        seam_weight(seams[i], masks[i], (h, w), feather)
    })
}

/// `build_proxy`'s stretch of a float32 plane: its 1st and 99th percentiles (float64, as for a
/// sequence of q) onto 0..1, clipped, raised to 1 / 2.2 and scaled to uint8 by truncation.
pub fn proxy_stretch(gray: &[f32], dst: &mut [u8]) {
    let p = crate::stats::np_percentiles_f64(gray, &[1.0, 99.0]);
    let (lo, hi) = (p[0], p[1]);
    let span = (hi - lo).max(1e-9);
    crate::par_zip(dst, gray, move |d, &v| {
        let t = ((v as f64 - lo) / span).clamp(0.0, 1.0);
        // NaN casts to 0, as NumPy's float to uint8 cast does on x86.
        *d = (libm::pow(t, 1.0 / 2.2) * 255.0) as u8;
    });
}

/// One part of `blend_ir`: where `sel` is set, the canvas takes the part's value, or the larger
/// of the two with `take_max`.
pub fn ir_blend_into(acc: &mut [f32], warped: &[f32], sel: &[bool], take_max: bool) {
    acc.par_chunks_mut(1 << 14)
        .zip(warped.par_chunks(1 << 14).zip(sel.par_chunks(1 << 14)))
        .for_each(|(a, (w, s))| {
            for ((a, &w), &s) in a.iter_mut().zip(w).zip(s) {
                if s {
                    *a = if take_max { a.max(w) } else { w };
                }
            }
        });
}

/// `blend_ir`'s last step: every canvas pixel no part reached (below 0) becomes 1.0.
pub fn ir_finish(acc: &mut [f32]) {
    acc.par_chunks_mut(1 << 14).for_each(|a| {
        for v in a {
            if *v < 0.0 {
                *v = 1.0;
            }
        }
    });
}

/// `(a & b).sum()` of two masks.
pub fn mask_overlap(a: &[bool], b: &[bool]) -> usize {
    a.par_chunks(1 << 16)
        .zip(b.par_chunks(1 << 16))
        .map(|(a, b)| a.iter().zip(b).filter(|(x, y)| **x & **y).count())
        .sum()
}

/// `img *= gains` per channel of an HxWx3 float32 image, in place.
pub fn scale_channels(img: &mut [f32], gains: [f32; 3]) {
    img.par_chunks_mut(3 << 14).for_each(|c| {
        for p in c.as_chunks_mut::<3>().0 {
            for k in 0..3 {
                p[k] *= gains[k];
            }
        }
    });
}

/// `np.clip(img, 0, 1)` in place; NaN stays.
pub fn clip_unit(img: &mut [f32]) {
    img.par_chunks_mut(1 << 14).for_each(|c| {
        for v in c {
            *v = v.clamp(0.0, 1.0);
        }
    });
}
