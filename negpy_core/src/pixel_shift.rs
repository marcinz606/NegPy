//! Sony Pixel Shift sets: each shot's samples placed on the output grid, and the movement of
//! one shot against the next per row band.

use rayon::prelude::*;

/// One shot of `merge_set`. `mosaic` is `h` x `w`; the output grid is `scale * (h - 1)` rows of
/// `scale * (w - 1)` sites. Sample `(origin.0 + i, origin.1 + j)` goes to site
/// `(half.0 + scale * i, half.1 + scale * j)`, into the plane of its CFA color: `out` (three
/// interleaved planes) for 0, 1 and 2, `second_green` for 3. `cfa` is the 2x2 pattern, row-major.
#[allow(clippy::too_many_arguments)]
pub fn place_shot(
    mosaic: &[u16],
    (_, w): (usize, usize),
    cfa: [u8; 4],
    scale: usize,
    origin: (usize, usize),
    half: (usize, usize),
    out: &mut [u16],
    second_green: &mut [u16],
) {
    let width = scale * (w - 1);
    out.par_chunks_mut(width * 3)
        .zip(second_green.par_chunks_mut(width))
        .enumerate()
        .filter(|(oy, _)| *oy >= half.0 && (*oy - half.0).is_multiple_of(scale))
        .for_each(move |(oy, (sites, greens))| {
            let y = origin.0 + (oy - half.0) / scale;
            let row = &mosaic[y * w + origin.1..y * w + origin.1 + w - 1];
            for (j, &sample) in row.iter().enumerate() {
                let ox = half.1 + scale * j;
                match cfa[(y % 2) * 2 + (origin.1 + j) % 2] {
                    3 => greens[ox] = sample,
                    color => sites[ox * 3 + usize::from(color)] = sample,
                }
            }
        });
}

/// The last step of `merge_set`: the green plane of `out` becomes the mean of the two greens,
/// a half rounded up.
pub fn average_greens(out: &mut [u16], second_green: &[u16]) {
    const BLOCK: usize = 1 << 14;
    out.par_chunks_mut(BLOCK * 3)
        .zip(second_green.par_chunks(BLOCK))
        .for_each(|(sites, greens)| {
            for (site, &g2) in sites.as_chunks_mut::<3>().0.iter_mut().zip(greens) {
                site[1] = ((u32::from(site[1]) + u32::from(g2) + 1) >> 1) as u16;
            }
        });
}

/// The CFA site (row, col in 0..2) with the largest sum: the color the light shows. The first
/// site wins a tie. An odd last row or column is left out, so every site has as many samples.
pub fn brightest_site(mosaic: &[u16], w: usize) -> (usize, usize) {
    let rows = (mosaic.len() / w) & !1;
    let sums = mosaic[..rows * w]
        .par_chunks(w)
        .enumerate()
        .fold(
            || [0u64; 4],
            |mut acc, (y, row)| {
                for (x, &v) in row[..w & !1].iter().enumerate() {
                    acc[(y % 2) * 2 + x % 2] += u64::from(v);
                }
                acc
            },
        )
        .reduce(
            || [0u64; 4],
            |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
        );
    let mut best = 0;
    for site in 1..4 {
        if sums[site] > sums[best] {
            best = site;
        }
    }
    (best / 2, best % 2)
}

/// How much fine detail one CFA site holds: the sum of the steps between neighboring samples of
/// that site, along rows and along columns, and the sum of the samples. Their ratio compares the
/// shots of one set; a shot blurred while it was exposed has a lower one.
pub fn site_detail(mosaic: &[u16], w: usize, site: (usize, usize)) -> (u64, u64) {
    let h = mosaic.len() / w;
    (site.0..h.saturating_sub(2))
        .into_par_iter()
        .step_by(2)
        .map(|y| {
            let (row, below) = (
                &mosaic[y * w..(y + 1) * w],
                &mosaic[(y + 2) * w..(y + 3) * w],
            );
            let (mut steps, mut total) = (0u64, 0u64);
            for x in (site.1..w.saturating_sub(2)).step_by(2) {
                let v = row[x];
                steps += u64::from(v.abs_diff(row[x + 2])) + u64::from(v.abs_diff(below[x]));
                total += u64::from(v);
            }
            (steps, total)
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// The plane of one CFA site inside a band, as float32.
fn patch(
    mosaic: &[u16],
    w: usize,
    (top, left): (usize, usize),
    (rows, cols): (usize, usize),
    site: (usize, usize),
) -> (Vec<f32>, usize, usize) {
    let (ph, pw) = ((rows - site.0).div_ceil(2), (cols - site.1).div_ceil(2));
    let mut plane = Vec::with_capacity(ph * pw);
    for i in 0..ph {
        let start = (top + site.0 + 2 * i) * w + left + site.1;
        plane.extend((0..pw).map(|j| f32::from(mosaic[start + 2 * j])));
    }
    (plane, ph, pw)
}

/// Step between neighboring 4x4 block means over their mean, along the weaker axis.
fn contrast(plane: &[f32], ph: usize, pw: usize) -> f64 {
    let (rows, cols) = (ph / 4, pw / 4);
    if rows < 2 || cols < 2 {
        return 0.0;
    }
    let mut blocks = vec![0f64; rows * cols];
    for y in 0..rows * 4 {
        for x in 0..cols * 4 {
            blocks[(y / 4) * cols + x / 4] += f64::from(plane[y * pw + x]);
        }
    }
    blocks.iter_mut().for_each(|b| *b /= 16.0);
    let (mut down, mut across) = (0f64, 0f64);
    for y in 0..rows {
        for x in 0..cols {
            let v = blocks[y * cols + x];
            if y + 1 < rows {
                down += (blocks[(y + 1) * cols + x] - v).abs();
            }
            if x + 1 < cols {
                across += (blocks[y * cols + x + 1] - v).abs();
            }
        }
    }
    let step = (down / ((rows - 1) * cols) as f64).min(across / (rows * (cols - 1)) as f64);
    let mean = blocks.iter().sum::<f64>() / blocks.len() as f64;
    step / mean.max(1.0)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// What a patch must show to count as textured: block contrast and phase-correlation response.
#[derive(Clone, Copy)]
pub struct Texture {
    pub contrast: f64,
    pub response: f64,
}

/// The move of `current` against `previous` less the `expected` (dx, dy), per row band of `band.0`
/// rows, on one CFA site, in sensor pixels: `(top, dx, dy)` for each band with a textured patch. A band's move is the
/// median over its patches, one per entry of `lefts`, each `band.1` columns wide.
#[allow(clippy::too_many_arguments)]
pub fn band_moves(
    previous: &[u16],
    current: &[u16],
    (h, w): (usize, usize),
    site: (usize, usize),
    band: (usize, usize),
    lefts: &[usize],
    expected: (f64, f64),
    texture: Texture,
) -> Vec<(usize, f64, f64)> {
    // A step of whole sensor pixels leaves the spectrum, so the peak fit reads only the error: the
    // fit is off for a soft frame at half a sample of this CFA site, which steps two sensor pixels.
    // A step with a half pixel stays in: taken out, the site's aliased detail would turn it into
    // an error.
    let whole = if expected.0.fract() == 0.0 && expected.1.fract() == 0.0 {
        expected
    } else {
        (0.0, 0.0)
    };
    let about = (whole.0 / 2.0, whole.1 / 2.0);
    let rest = (expected.0 - whole.0, expected.1 - whole.1);
    let tops: Vec<usize> = (0..(h + 1).saturating_sub(band.0))
        .step_by(band.0.max(1))
        .collect();
    tops.par_iter()
        .filter_map(|&top| {
            let (mut dx, mut dy) = (Vec::new(), Vec::new());
            for &left in lefts {
                let (first, ph, pw) = patch(previous, w, (top, left), band, site);
                if contrast(&first, ph, pw) < texture.contrast {
                    continue;
                }
                let (second, _, _) = patch(current, w, (top, left), band, site);
                let (x, y, response) =
                    crate::phase::phase_correlate_about(&first, &second, (ph, pw), about);
                if response >= texture.response {
                    dx.push(x);
                    dy.push(y);
                }
            }
            (!dx.is_empty()).then(|| {
                (
                    top,
                    2.0 * median(&mut dx) - rest.0,
                    2.0 * median(&mut dy) - rest.1,
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGGB: [u8; 4] = [0, 1, 3, 2];

    #[test]
    fn four_shots_fill_every_site() {
        let (h, w) = (6usize, 8usize);
        let positions = [(0usize, 1usize), (1, 1), (1, 0), (0, 0)];
        let mut out = vec![0u16; (h - 1) * (w - 1) * 3];
        let mut g2 = vec![0u16; (h - 1) * (w - 1)];
        for (row, col) in positions {
            // A sample holds its output site and color, so a wrong placement shows.
            let mosaic: Vec<u16> = (0..h * w)
                .map(|i| {
                    let (y, x) = (i / w, i % w);
                    let color = match RGGB[(y % 2) * 2 + x % 2] {
                        3 => 1,
                        c => c,
                    };
                    (((y + row) * 16 + x + col) * 4 + usize::from(color)) as u16
                })
                .collect();
            place_shot(
                &mosaic,
                (h, w),
                RGGB,
                1,
                (1 - row, 1 - col),
                (0, 0),
                &mut out,
                &mut g2,
            );
        }
        average_greens(&mut out, &g2);
        for y in 0..h - 1 {
            for x in 0..w - 1 {
                for c in 0..3 {
                    assert_eq!(
                        usize::from(out[(y * (w - 1) + x) * 3 + c]),
                        ((y + 1) * 16 + x + 1) * 4 + c
                    );
                }
            }
        }
    }

    #[test]
    fn greens_round_a_half_up() {
        let mut out = vec![0, 65534, 0];
        average_greens(&mut out, &[65535]);
        assert_eq!(out[1], 65535);
    }

    #[test]
    fn the_lit_site_is_the_brightest() {
        let mosaic: Vec<u16> = (0..64)
            .map(|i| {
                if (i / 8) % 2 == 1 && i % 2 == 0 {
                    900
                } else {
                    100
                }
            })
            .collect();
        assert_eq!(brightest_site(&mosaic, 8), (1, 0));
    }

    #[test]
    fn detail_counts_the_steps_of_one_site() {
        // Site (0, 0) alternates 10 and 30 along a row; the other sites hold a value it must not read.
        let (h, w) = (6usize, 8usize);
        let mosaic: Vec<u16> = (0..h * w)
            .map(|i| match ((i / w) % 2, (i % w) % 2) {
                (0, 0) => 10 + 20 * ((i % w) / 2 % 2) as u16,
                _ => 999,
            })
            .collect();
        // 2 rows x 3 samples, each with a row step of 20 and no column step.
        assert_eq!(site_detail(&mosaic, w, (0, 0)), (120, 100));
    }

    #[test]
    fn a_flat_band_has_no_move() {
        let flat = vec![1000u16; 256 * 256];
        let texture = Texture {
            contrast: 0.005,
            response: 0.3,
        };
        assert!(
            band_moves(
                &flat,
                &flat,
                (256, 256),
                (0, 0),
                (128, 64),
                &[0, 96],
                (1.0, 0.0),
                texture
            )
            .is_empty()
        );
    }
}
