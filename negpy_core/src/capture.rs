//! Ports of the preview and capture metering in negpy/services/capture, negpy/services/scanning
//! and the scan preview widgets.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::stats::{np_median, np_percentiles, np_quantile, pairwise_sum_f32};

/// How `stretch_preview` maps a channel between its 1st and 99th percentiles.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stretch {
    /// `preview_positive` of a positive: `clip((v - lo) / (hi - lo), 0, 1) * 255`.
    Positive,
    /// `preview_positive` of a negative: `clip((hi - v) / (hi - lo), 0, 1) * 255`.
    Negative,
    /// `_preview_u8`: `clip((v - lo) / (hi - lo) * 255, 0, 255)`.
    Prescan,
}

/// The scan preview stretch of an HxWxC float32 image into `out_cn` uint8 channels, output channel
/// c reading input channel min(c, C - 1). A channel whose 99th percentile is not above its 1st is 0.
pub fn stretch_preview(img: &[f32], cn: usize, out_cn: usize, mode: Stretch, dst: &mut [u8]) {
    let bounds: Vec<Option<(f32, f32)>> = (0..cn)
        .into_par_iter()
        .map(|c| {
            let ch: Vec<f32> = img.iter().skip(c).step_by(cn).copied().collect();
            let p = np_percentiles(&ch, &[1.0, 99.0])?;
            (p[1] > p[0]).then_some((p[0], p[1]))
        })
        .collect();
    dst.par_chunks_mut(out_cn * 4096)
        .zip(img.par_chunks(cn * 4096))
        .for_each(|(d, s)| {
            for (o, p) in d.chunks_exact_mut(out_cn).zip(s.chunks_exact(cn)) {
                for (c, v) in o.iter_mut().enumerate() {
                    let src = c.min(cn - 1);
                    let Some((lo, hi)) = bounds[src] else {
                        *v = 0;
                        continue;
                    };
                    let x = p[src];
                    // NumPy's float32 to uint8 cast truncates.
                    *v = match mode {
                        Stretch::Positive => ((x - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0,
                        Stretch::Negative => ((hi - x) / (hi - lo)).clamp(0.0, 1.0) * 255.0,
                        Stretch::Prescan => ((x - lo) / (hi - lo) * 255.0).clamp(0.0, 255.0),
                    } as u8;
                }
            }
        });
}

/// `sharpness`: the float32 mean square of the 4-neighbour Laplacian over the squared float32
/// mean level, or 0 below `min_mean`.
pub fn sharpness(img: &[f32], h: usize, w: usize, min_mean: f64) -> f64 {
    let mean = (pairwise_sum_f32(img) / img.len() as f32) as f64;
    if mean < min_mean {
        return 0.0;
    }
    let mut sq = vec![0f32; (h - 2) * (w - 2)];
    sq.par_chunks_mut(w - 2).enumerate().for_each(|(i, row)| {
        let y = i + 1;
        let at = |yy: usize, x: usize| img[yy * w + x];
        for (j, d) in row.iter_mut().enumerate() {
            let x = j + 1;
            let lap = 4.0 * at(y, x) - at(y - 1, x) - at(y + 1, x) - at(y, x - 1) - at(y, x + 1);
            *d = lap * lap;
        }
    });
    (pairwise_sum_f32(&sq) / sq.len() as f32) as f64 / (mean * mean)
}

/// The share of pixels of an image with `cn` channels where any of the first three is at least `level`.
pub fn any_channel_at_least(img: &[f32], cn: usize, level: f32) -> f64 {
    let n = img.len() / cn;
    let hits: usize = img
        .par_chunks(cn * 4096)
        .map(|s| {
            s.chunks_exact(cn)
                .filter(|p| p[..cn.min(3)].iter().any(|&v| v >= level))
                .count()
        })
        .sum();
    hits as f64 / n as f64
}

/// `_to_grey`: `(r + g + b + 1) // 3` per pixel, summed in uint32.
pub fn to_grey<T: Copy + Send + Sync + Into<u32> + TryFrom<u32>>(
    rgb: &[T],
    cn: usize,
    dst: &mut [T],
) {
    dst.par_chunks_mut(4096)
        .zip(rgb.par_chunks(cn * 4096))
        .for_each(|(d, s)| {
            for (o, p) in d.iter_mut().zip(s.chunks_exact(cn)) {
                let sum: u32 = p.iter().map(|&v| v.into()).sum();
                // The average of the channels fits the type.
                *o = T::try_from((sum + 1) / 3).unwrap_or(p[0]);
            }
        });
}

/// `np.clip(img.astype(np.float32) / divisor, 0, 1)` in float32.
pub fn normalized<T: Copy + Send + Sync + Into<f64>>(img: &[T], divisor: f32, dst: &mut [f32]) {
    crate::par_zip(dst, img, move |d, &v| {
        // NaN stays, as np.clip keeps it.
        let x: f64 = v.into();
        *d = (x as f32 / divisor).clamp(0.0, 1.0);
    });
}

/// `meter_frame`'s picture statistics over rows y1..y2 and columns x1..x2 of an HxWx3 float32
/// frame: the share whose brightest channel reaches `sat`, and the 99.9th percentile per channel
/// of the unsaturated pixels (of every pixel when at least `blown_share` saturate, or all do).
pub fn meter_picture(
    img: &[f32],
    w: usize,
    (y1, y2, x1, x2): (usize, usize, usize, usize),
    sat: f32,
    blown_share: f64,
) -> (f64, bool, [f32; 3]) {
    let mut picture: Vec<[f32; 3]> = Vec::with_capacity((y2 - y1) * (x2 - x1));
    for y in y1..y2 {
        picture.extend_from_slice(img[(y * w + x1) * 3..(y * w + x2) * 3].as_chunks::<3>().0);
    }
    let saturated: Vec<bool> = picture
        .iter()
        .map(|p| p[0].max(p[1]).max(p[2]) >= sat)
        .collect();
    let n = picture.len();
    let hits = saturated.iter().filter(|&&s| s).count();
    let clipped = hits as f64 / n as f64;
    let blown = clipped >= blown_share;
    let keep_all = blown || hits == n;
    let thinnest = std::array::from_fn(|c| {
        let v: Vec<f32> = picture
            .iter()
            .zip(&saturated)
            .filter(|(_, s)| keep_all || !**s)
            .map(|(p, _)| p[c])
            .collect();
        np_percentiles(&v, &[99.9]).map_or(f32::NAN, |p| p[0])
    });
    (clipped, blown, thinnest)
}

/// `_peak_map(np.max(img, axis=2))`: the brightest channel of an HxWx3 float32 frame, max-pooled
/// in f x f blocks over the frame zero-padded to whole blocks.
pub fn peak_map(img: &[f32], h: usize, w: usize, f: usize) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = (h.div_ceil(f), w.div_ceil(f));
    let mut out = vec![0f32; ph * pw];
    out.par_chunks_mut(pw).enumerate().for_each(|(by, row)| {
        for y in by * f..((by + 1) * f).min(h) {
            for (x, p) in img[y * w * 3..(y + 1) * w * 3]
                .as_chunks::<3>()
                .0
                .iter()
                .enumerate()
            {
                let o = &mut row[x / f];
                *o = o.max(p[0].max(p[1]).max(p[2]));
            }
        }
    });
    (out, ph, pw)
}

/// `raw_channel_clip_fraction`'s two shares of the photosites of one CFA colour: at or above
/// `threshold` (None to skip), and on the saturation plateau. A sensor can saturate below its
/// published white level, but its sites still pile up at the highest code it reaches: the densest
/// code in the top `window` counts, `ratio` times denser than each of the next codes, with a
/// few noisy sites allowed above it. `guard` (the white level, 0 for none) keeps a frame far
/// from saturation from reading its noise floor as a plateau.
pub fn raw_clip_fractions(
    values: &[u16],
    threshold: Option<u16>,
    guard: u32,
    (window, min_sites, neighbours, ratio): (usize, usize, usize, f64),
) -> (f64, f64) {
    let n = values.len() as f64;
    let linearity = threshold.map_or(0.0, |t| {
        values.iter().filter(|&&v| v >= t).count() as f64 / n
    });
    let top = values.iter().copied().max().unwrap_or(0) as u32;
    if guard > 0 && top * 2 < guard {
        return (linearity, 0.0);
    }
    let lo = top.saturating_sub(window as u32);
    let mut counts = vec![0usize; (top - lo) as usize + 1];
    for &v in values {
        if v as u32 >= lo {
            counts[(v as u32 - lo) as usize] += 1;
        }
    }
    // Only codes the file holds count: a lossy raw encoding stores a sparse set of codes.
    let populated: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
    let peak_at = populated.iter().enumerate().fold(0, |b, (i, &p)| {
        if counts[p] > counts[populated[b]] {
            i
        } else {
            b
        }
    });
    let peak = counts[populated[peak_at]];
    if peak < min_sites {
        return (linearity, 0.0);
    }
    let density = populated[peak_at + 1..]
        .iter()
        .take(neighbours)
        .map(|&p| counts[p])
        .max()
        .unwrap_or(0);
    if (peak as f64) < ratio * (density as f64).max(1.0) {
        return (linearity, 0.0);
    }
    let anchor = lo + populated[peak_at] as u32;
    (
        linearity,
        values.iter().filter(|&&v| v as u32 >= anchor).count() as f64 / n,
    )
}

/// `_gutter_scan`'s column profile: every `step`th row of an HxW float32 plane over `peak`, then
/// the float32 mean of each column, rows added in order as NumPy reduces axis 0.
pub fn column_means(a: &[f32], w: usize, step: usize, peak: f32) -> Vec<f32> {
    let rows: Vec<&[f32]> = a.chunks_exact(w).step_by(step.max(1)).collect();
    let mut sum = vec![0f32; w];
    for r in &rows {
        for (s, &v) in sum.iter_mut().zip(*r) {
            *s += v / peak;
        }
    }
    sum.iter().map(|&s| s / rows.len() as f32).collect()
}

/// `_looks_border_crushed`'s two medians of an HxW float32 sample: of the border ring `m` wide
/// and of the central half.
pub fn ring_inner_medians(arr: &[f32], h: usize, w: usize, m: usize) -> (f32, f32) {
    let at = |y: usize| &arr[y * w..][..w];
    let mut ring: Vec<f32> = Vec::new();
    for y in (0..m).chain(h - m..h) {
        ring.extend_from_slice(at(y));
    }
    for y in 0..h {
        ring.extend_from_slice(&at(y)[..m]);
        ring.extend_from_slice(&at(y)[w - m..]);
    }
    let mut inner = Vec::with_capacity((3 * h / 4 - h / 4) * (3 * w / 4 - w / 4));
    for y in h / 4..3 * h / 4 {
        inner.extend_from_slice(&at(y)[w / 4..3 * w / 4]);
    }
    (np_median(&mut ring), np_median(&mut inner))
}

/// `print_dmax`: the float32 mean colour, each channel rounded half to even, of the pixels whose
/// Rec. 709 luma (OpenBLAS's sgemv order) is at most its `share` quantile. None for no pixels.
#[multiversion(targets("x86_64+avx2+fma", "aarch64+neon"))]
pub fn print_dmax(px: &[[u8; 3]], share: f64) -> Option<[i64; 3]> {
    if px.is_empty() {
        return None;
    }
    let (wr, wg, wb) = (0.2126f32, 0.7152f32, 0.0722f32);
    let luma: Vec<f32> = px
        .iter()
        .map(|p| {
            let p = p.map(f32::from);
            p[2].mul_add(wb, p[0].mul_add(wr, p[1] * wg))
        })
        .collect();
    let cut = np_quantile(&luma, share)?;
    let mut sum = [0f32; 3];
    let mut k = 0usize;
    for (p, &l) in px.iter().zip(&luma) {
        if l <= cut {
            for c in 0..3 {
                sum[c] += p[c] as f32;
            }
            k += 1;
        }
    }
    Some(sum.map(|s| ((s / k as f32) as f64).round_ties_even() as i64))
}

/// `preview_positive`'s stretch of a linear HxWx3 float32 thumbnail: a channel whose 99th
/// percentile is under 5 % of the strongest borrows the strongest; then the density -log10 of
/// the value clipped at 1e-4, stretched between its 1st and 99th percentiles per channel and
/// quantized as `float_to_uint8`.
pub fn log_stretch_u8(linear: &[f32], dst: &mut [u8]) {
    let ch = |c: usize| {
        linear
            .iter()
            .skip(c)
            .step_by(3)
            .copied()
            .collect::<Vec<f32>>()
    };
    let peaks: Vec<f32> = (0..3)
        .map(|c| np_percentiles(&ch(c), &[99.0]).map_or(f32::NAN, |p| p[0]))
        .collect();
    let strongest = (0..3).fold(0, |b, c| if peaks[c] > peaks[b] { c } else { b });
    let source: [usize; 3] = std::array::from_fn(|c| {
        if peaks[c] < peaks[strongest] * 0.05 {
            strongest
        } else {
            c
        }
    });
    let mut density = vec![0f32; linear.len()];
    crate::par_zip(
        density.as_chunks_mut::<3>().0,
        linear.as_chunks::<3>().0,
        move |d, p| {
            // max drops NaN where np.clip keeps it; a thumbnail decoded from uint8 holds none.
            *d = source.map(|s| -crate::mathf::log10f(p[s].max(1e-4)));
        },
    );
    let bounds: Vec<(f32, f32)> = (0..3)
        .map(|c| {
            let v: Vec<f32> = density.iter().skip(c).step_by(3).copied().collect();
            let p = np_percentiles(&v, &[1.0, 99.0]).unwrap_or(vec![0.0, 0.0]);
            (p[0], (p[1] - p[0]).max(1e-6))
        })
        .collect();
    let mut positive = vec![0f32; linear.len()];
    crate::par_zip(
        positive.as_chunks_mut::<3>().0,
        density.as_chunks::<3>().0,
        move |d, p| {
            *d = std::array::from_fn(|c| ((p[c] - bounds[c].0) / bounds[c].1).clamp(0.0, 1.0));
        },
    );
    crate::convert::to_uint(&positive, dst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_frame_has_no_sharpness() {
        let img = vec![10f32; 25];
        assert_eq!(sharpness(&img, 5, 5, 4.0), 0.0);
        assert_eq!(sharpness(&[1f32; 25], 5, 5, 4.0), 0.0);
    }

    #[test]
    fn grey_rounds_up_from_two_thirds() {
        let mut out = [0u16; 2];
        to_grey(&[1u16, 1, 0, 2, 2, 1], 3, &mut out);
        assert_eq!(out, [1, 2]);
    }
}
