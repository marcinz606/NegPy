//! Fade Restoration's measurements on the Estimate grid (`negpy/features/process/fade.py`).

use rayon::prelude::*;

use crate::stats::quantile_f64;

/// Cells per partial sum. A fixed size, so the sums and their merge order never depend on the
/// core count.
const BLOCK: usize = 1 << 14;

/// The numbers `neutral_bands` reads, all from `fade.py`'s CROSSOVER_* constants.
#[derive(Clone, Debug)]
pub struct BandParams {
    /// The inner band edges over the frame's range, rising (CROSSOVER_BAND_EDGES[1:-1]).
    pub inner_edges: Vec<f64>,
    pub weights: [f64; 3],
    pub range_pct: (f64, f64),
    pub min_span: f64,
    pub neutral_spread: f64,
    pub cluster_radius: f64,
    pub steps: usize,
    pub min_weight: f64,
    pub clipped_log: f64,
    pub min_cells: usize,
}

/// Each band's neutral centre (NaN in a band with no weight), its neutral weight, and the total
/// neutral weight of the frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Bands {
    pub centre: Vec<[f64; 3]>,
    pub weight: Vec<f64>,
    pub total: f64,
}

/// Whether a cell holds a reading: no decoded zero in any channel and not full scale in all three.
pub fn is_reading(p: &[f32; 3], clipped_log: f64) -> bool {
    p.iter().all(|&v| f64::from(v) > clipped_log) && p.iter().any(|&v| v < 0.0)
}

/// Per band, (sum of w, sum of w * cell) over the cells, `w(i)` for cell i. Partial sums of
/// BLOCK cells, merged in block order.
fn band_sums(
    cells: &[[f64; 3]],
    band: &[u8],
    n_bands: usize,
    w: impl Fn(usize) -> f64 + Sync,
) -> Vec<[f64; 4]> {
    let partials: Vec<Vec<[f64; 4]>> = (0..cells.len().div_ceil(BLOCK))
        .into_par_iter()
        .map(|b| {
            let mut acc = vec![[0.0f64; 4]; n_bands];
            for i in b * BLOCK..((b + 1) * BLOCK).min(cells.len()) {
                let wi = w(i);
                let a = &mut acc[band[i] as usize];
                let c = cells[i];
                a[0] += wi;
                a[1] += wi * c[0];
                a[2] += wi * c[1];
                a[3] += wi * c[2];
            }
            acc
        })
        .collect();
    let mut out = vec![[0.0f64; 4]; n_bands];
    for part in partials {
        for (o, p) in out.iter_mut().zip(part) {
            for j in 0..4 {
                o[j] += p[j];
            }
        }
    }
    out
}

fn mean_log(c: &[f64; 3], w: &[f64; 3]) -> f64 {
    c[0] * w[0] + c[1] * w[1] + c[2] * w[2]
}

/// `neutral_bands`: soft neutral weight per cell, then mean-shift steps onto each density
/// band's largest near-neutral cluster. None with fewer than `min_cells` readings or no weight.
pub fn neutral_bands(grid: &[[f32; 3]], p: &BandParams) -> Option<Bands> {
    let cells: Vec<[f64; 3]> = grid
        .par_iter()
        .filter(|c| is_reading(c, p.clipped_log))
        .map(|c| c.map(f64::from))
        .collect();
    if cells.len() < p.min_cells.max(1) {
        return None;
    }
    let w3 = p.weights;
    let mut mean: Vec<f64> = cells.par_iter().map(|c| mean_log(c, &w3)).collect();
    let mean_band: Vec<f64> = mean.clone();
    let lo = quantile_f64(&mut mean, p.range_pct.0 / 100.0)?;
    let hi = quantile_f64(&mut mean, p.range_pct.1 / 100.0)?;
    let span = (hi - lo).max(p.min_span);
    let n_bands = p.inner_edges.len() + 1;
    let edges = &p.inner_edges;
    let band: Vec<u8> = mean_band
        .par_iter()
        .map(|&m| {
            let t = ((m - lo) / span).clamp(0.0, 1.0);
            edges.iter().take_while(|&&e| e <= t).count() as u8
        })
        .collect();
    let spread_unit = p.neutral_spread;
    let neutral: Vec<f64> = cells
        .par_iter()
        .map(|c| {
            let s = (c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])) / spread_unit;
            1.0 / (1.0 + s * s)
        })
        .collect();
    let sums = band_sums(&cells, &band, n_bands, |i| neutral[i]);
    let mut weight: Vec<f64> = sums.iter().map(|s| s[0]).collect();
    let total: f64 = weight.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let mut centre: Vec<[f64; 3]> = sums
        .iter()
        .map(|s| {
            if s[0] > 0.0 {
                [s[1] / s[0], s[2] / s[0], s[3] / s[0]]
            } else {
                [f64::NAN; 3]
            }
        })
        .collect();
    let radius = p.cluster_radius;
    for _ in 0..p.steps {
        let live: Vec<bool> = centre.iter().map(|c| !c[0].is_nan()).collect();
        let at = &centre;
        let shift = band_sums(&cells, &band, n_bands, |i| {
            let k = band[i] as usize;
            if !live[k] {
                return 0.0;
            }
            let (c, z) = (cells[i], at[k]);
            let d = (c[0] - z[0])
                .abs()
                .max((c[1] - z[1]).abs())
                .max((c[2] - z[2]).abs())
                / radius;
            neutral[i] / (1.0 + d * d)
        });
        for k in 0..n_bands {
            let s = shift[k];
            if !live[k] || s[0] < p.min_weight {
                continue;
            }
            centre[k] = [s[1] / s[0], s[2] / s[0], s[3] / s[0]];
            weight[k] = s[0];
        }
    }
    Some(Bands {
        centre,
        weight,
        total,
    })
}

/// Each q's percentile of every channel of `cells`, as `np.percentile(cells[:, ch], q)` gives it.
fn channel_percentiles(cells: &[[f32; 3]], qs: &[f64]) -> Option<Vec<[f32; 3]>> {
    let per: Vec<Vec<f32>> = (0..3)
        .map(|ch| {
            let column: Vec<f32> = cells.par_iter().map(|c| c[ch]).collect();
            crate::stats::percentiles(&column, qs)
        })
        .collect::<Option<_>>()?;
    Some(
        (0..qs.len())
            .map(|i| [per[0][i], per[1][i], per[2][i]])
            .collect(),
    )
}

/// `_measured_slide_bounds`' percentiles: each q's percentile of each channel over the cells
/// with no decoded zero, or over every cell when all hold one. None for an empty grid.
pub fn slide_extremes(grid: &[[f32; 3]], clipped_log: f64, qs: &[f64]) -> Option<Vec<[f32; 3]>> {
    let unclipped: Vec<[f32; 3]> = grid
        .par_iter()
        .filter(|c| c.iter().all(|&v| f64::from(v) > clipped_log))
        .copied()
        .collect();
    channel_percentiles(
        if unclipped.is_empty() {
            grid
        } else {
            &unclipped
        },
        qs,
    )
}

/// `measure_white`: each channel's q percentile over the readings, every channel capped at
/// full scale first. None with no reading.
pub fn white(grid: &[[f32; 3]], clipped_log: f64, q: f64) -> Option<[f32; 3]> {
    let cells: Vec<[f32; 3]> = grid
        .par_iter()
        .map(|c| c.map(|v| v.min(0.0)))
        .filter(|c| is_reading(c, clipped_log))
        .collect();
    Some(channel_percentiles(&cells, &[q])?[0])
}

/// `range_points`' percentiles: each q's percentile of the float64 mean of the finite cells.
/// None with fewer than `min_cells` of them.
pub fn mean_percentiles(grid: &[[f32; 3]], qs: &[f64], min_cells: usize) -> Option<Vec<f64>> {
    let mut mean: Vec<f64> = grid
        .par_iter()
        .filter(|c| c.iter().all(|v| v.is_finite()))
        .map(|c| (f64::from(c[0]) + f64::from(c[1]) + f64::from(c[2])) / 3.0)
        .collect();
    if mean.len() < min_cells.max(1) {
        return None;
    }
    mean.par_sort_unstable_by(f64::total_cmp);
    let n = mean.len();
    Some(
        qs.iter()
            .map(|&q| {
                let at = (n - 1) as f64 * (q / 100.0);
                let lo = (at.floor().max(0.0) as usize).min(n - 1);
                let (a, b, t) = (mean[lo], mean[(lo + 1).min(n - 1)], at - lo as f64);
                if t >= 0.5 {
                    b - (b - a) * (1.0 - t)
                } else {
                    a + (b - a) * t
                }
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> BandParams {
        BandParams {
            inner_edges: vec![0.06, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
            weights: [1.0 / 3.0; 3],
            range_pct: (0.5, 99.5),
            min_span: 0.05,
            neutral_spread: 0.25,
            cluster_radius: 0.12,
            steps: 3,
            min_weight: 8.0,
            clipped_log: -5.0,
            min_cells: 64,
        }
    }

    fn ramp(n: usize, cast: [f32; 3]) -> Vec<[f32; 3]> {
        (0..n)
            .map(|i| {
                let v = -2.5 + 2.4 * i as f32 / n as f32;
                [v + cast[0], v + cast[1], v + cast[2]]
            })
            .collect()
    }

    #[test]
    fn a_gray_ramp_has_gray_centres_in_every_band() {
        let bands = neutral_bands(&ramp(4000, [0.0; 3]), &params()).unwrap();
        assert_eq!(bands.centre.len(), 8);
        for c in &bands.centre {
            assert!((c[0] - c[1]).abs() < 1e-9 && (c[1] - c[2]).abs() < 1e-9);
        }
        assert!((bands.total - 4000.0).abs() < 1e-6);
    }

    #[test]
    fn a_cast_moves_the_centres_with_it() {
        let bands = neutral_bands(&ramp(4000, [0.05, 0.0, -0.05]), &params()).unwrap();
        for c in &bands.centre {
            assert!((c[0] - c[2] - 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn too_few_readings_give_none() {
        assert!(neutral_bands(&ramp(10, [0.0; 3]), &params()).is_none());
        let zeros = vec![[0.0f32; 3]; 1000];
        assert!(neutral_bands(&zeros, &params()).is_none());
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let mut cells = ramp(100_000, [0.02, 0.0, -0.03]);
        for (i, c) in cells.iter_mut().enumerate() {
            c[i % 3] += ((i * 7919) % 101) as f32 * 1e-3;
        }
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let many = rayon::ThreadPoolBuilder::new()
            .num_threads(7)
            .build()
            .unwrap();
        let a = one.install(|| neutral_bands(&cells, &params()));
        let b = many.install(|| neutral_bands(&cells, &params()));
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }

    #[test]
    fn the_white_skips_zeros_and_light_past_the_film() {
        let mut cells = ramp(1000, [0.0; 3]);
        cells.extend([[0.2f32, 0.1, 0.3], [-9.0, -1.0, -1.0]]);
        let w = white(&cells, -5.0, 100.0).unwrap();
        assert_eq!(w, cells[999]);
        assert!(white(&[[0.0f32; 3]; 4], -5.0, 50.0).is_none());
    }

    #[test]
    fn slide_extremes_fall_back_to_every_cell() {
        let zeros = vec![[-9.0f32, -1.0, -2.0]; 5];
        assert_eq!(
            slide_extremes(&zeros, -5.0, &[50.0]).unwrap(),
            vec![[-9.0, -1.0, -2.0]]
        );
    }

    #[test]
    fn mean_percentiles_interpolate_between_ranks() {
        let cells: Vec<[f32; 3]> = (0..5)
            .map(|i| [i as f32; 3])
            .chain([[f32::NAN; 3]])
            .collect();
        assert_eq!(
            mean_percentiles(&cells, &[0.0, 50.0, 62.5, 100.0], 1).unwrap(),
            vec![0.0, 2.0, 2.5, 4.0]
        );
        assert!(mean_percentiles(&cells, &[50.0], 64).is_none());
    }
}
