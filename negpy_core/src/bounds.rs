//! The density bounds sampling of negpy/features/exposure/normalization.py.

use crate::stats::np_percentiles_f64;

const LUMA_R: f64 = 0.2126;
const LUMA_G: f64 = 0.7152;
const LUMA_B: f64 = 0.0722;
const EPSILON: f64 = 1e-6;

/// `np.percentile(grid[:, :, ch], qs)` for each channel of an HxWx3 float32 grid, with float64
/// interpolation (an `np.float64` or sequence q).
pub fn channel_percentiles(grid: &[f32], qs: &[f64]) -> [Vec<f64>; 3] {
    std::array::from_fn(|ch| {
        let plane: Vec<f32> = grid.iter().skip(ch).step_by(3).copied().collect();
        np_percentiles_f64(&plane, qs)
    })
}

/// NumPy's linear quantile of float64 values at a virtual index of `(n - 1) * q`, the far half
/// counted back from the upper sample; NaN when any value is NaN. Reorders `values`.
fn quantile_at(values: &mut [f64], virtual_index: f64) -> f64 {
    let n = values.len();
    if n == 0 || values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let lo = (virtual_index.floor().max(0.0) as usize).min(n - 1);
    let t = virtual_index - lo as f64;
    let a = *values.select_nth_unstable_by(lo, f64::total_cmp).1;
    let b = if lo + 1 < n {
        values[lo + 1..]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    } else {
        a
    };
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

fn np_quantile(values: &mut [f64], q: f64) -> f64 {
    let n = values.len();
    quantile_at(values, n.saturating_sub(1) as f64 * q)
}

fn np_percentile(values: &mut [f64], q: f64) -> f64 {
    let n = values.len();
    quantile_at(values, n.saturating_sub(1) as f64 * (q / 100.0))
}

/// `np.median` of float64 values: the middle one, or the mean of the two middle ones.
fn np_median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 || values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let hi = *values.select_nth_unstable_by(n / 2, f64::total_cmp).1;
    if n % 2 == 1 {
        return hi;
    }
    let lo = values[..n / 2]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    (lo + hi) / 2.0
}

/// `_rms_chroma` of one triplet.
fn rms_chroma([r, g, b]: [f64; 3]) -> f64 {
    (((r - g) * (r - g) + (g - b) * (g - b) + (r - b) * (r - b)) / 3.0).sqrt()
}

/// The constants of the same-pixel color pass.
#[derive(Clone, Copy)]
pub struct ColorPass {
    pub chroma_quantile: f64,
    pub chroma_cap: f64,
    pub first_pass_cap: f64,
    pub min_pixels: usize,
    pub band_width: f64,
}

/// `_same_pixel_color_floor_refs`: the dense-end color refs from the luma band's lowest-chroma
/// pixels, or None when that set is too small or too chromatic.
pub fn same_pixel_color_floor_refs(
    grid: &[f32],
    luma_floors: [f64; 3],
    luma_ceils: [f64; 3],
    base: [f64; 3],
    color_clip: f64,
    c: ColorPass,
) -> Option<[f64; 3]> {
    let flat: Vec<[f64; 3]> = grid
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| p.map(|v| v as f64))
        .collect();
    let denom: [f64; 3] = std::array::from_fn(|ch| {
        let d = luma_ceils[ch] - luma_floors[ch];
        if d.abs() < EPSILON {
            if d >= 0.0 { EPSILON } else { -EPSILON }
        } else {
            d
        }
    });
    let luma: Vec<f64> = flat
        .iter()
        .map(|p| {
            let n: [f64; 3] = std::array::from_fn(|ch| (p[ch] - luma_floors[ch]) / denom[ch]);
            LUMA_R * n[0] + LUMA_G * n[1] + LUMA_B * n[2]
        })
        .collect();

    let clip = 0.00001f64.max((50.0 - c.band_width).min(color_clip));
    let (lo, hi) = {
        let mut v = luma.clone();
        (
            np_percentile(&mut v, clip),
            np_percentile(&mut v, clip + c.band_width),
        )
    };
    let d: Vec<[f64; 3]> = flat
        .iter()
        .zip(&luma)
        .filter(|&(_, &l)| l >= lo && l <= hi)
        .map(|(p, _)| std::array::from_fn(|ch| p[ch] - base[ch]))
        .collect();
    if d.len() < c.min_pixels {
        return None;
    }

    // The lowest-chroma quantile of the band, chroma read with `gamma` as each channel's span:
    // the selected rows and their median chroma.
    let select = |gamma: [f64; 3]| -> Option<(Vec<usize>, f64)> {
        let g = gamma.map(|v| if v.abs() < EPSILON { EPSILON } else { v });
        let chroma: Vec<f64> = d
            .iter()
            .map(|p| rms_chroma(std::array::from_fn(|ch| p[ch] / g[ch])))
            .collect();
        let thr = np_quantile(&mut chroma.clone(), c.chroma_quantile);
        let sel: Vec<usize> = (0..chroma.len()).filter(|&i| chroma[i] <= thr).collect();
        if sel.len() < c.min_pixels {
            return None;
        }
        let mut picked: Vec<f64> = sel.iter().map(|&i| chroma[i]).collect();
        Some((sel, np_median(&mut picked)))
    };
    let column_medians = |rows: &[usize]| -> [f64; 3] {
        std::array::from_fn(|ch| np_median(&mut rows.iter().map(|&i| d[i][ch]).collect::<Vec<_>>()))
    };

    let spans: [f64; 3] = std::array::from_fn(|ch| luma_floors[ch] - base[ch]);
    let (first, first_chroma) = select(spans)?;
    // A loose first cap: pass 2 would self-normalize a homogeneous colored cluster to zero
    // chroma and read it as neutral.
    if first_chroma > c.first_pass_cap {
        return None;
    }
    let provisional = column_medians(&first);
    if provisional.iter().any(|v| v.abs() < EPSILON) {
        return None;
    }
    let (second, second_chroma) = select(provisional)?;
    if second_chroma > c.chroma_cap {
        return None;
    }
    let m = column_medians(&second);
    Some(std::array::from_fn(|ch| base[ch] + m[ch]))
}
