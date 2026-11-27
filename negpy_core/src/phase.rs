//! `cv2.phaseCorrelate` with a `cv2.createHanningWindow` window, through rustfft and realfft.

use rayon::prelude::*;
use rustfft::num_complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use realfft::RealFftPlanner;

/// `cv2.getOptimalDFTSize`: the smallest 2^a 3^b 5^c at least `n`.
pub fn optimal_dft_size(n: usize) -> usize {
    (n.max(1)..)
        .find(|&m| {
            let mut k = m;
            for p in [2, 3, 5] {
                while k % p == 0 {
                    k /= p;
                }
            }
            k == 1
        })
        .unwrap_or(n)
}

/// `cv2.createHanningWindow((w, h), cv2.CV_32F)` for h, w >= 2.
pub fn hanning_window(h: usize, w: usize) -> Vec<f32> {
    let tau = 2.0 * std::f64::consts::PI;
    let (c0, c1) = (tau / (w - 1) as f64, tau / (h - 1) as f64);
    let wc: Vec<f64> = (0..w)
        .map(|j| 0.5 * (1.0 - (c0 * j as f64).cos()))
        .collect();
    let mut out = vec![0f32; h * w];
    out.par_chunks_mut(w).enumerate().for_each(|(i, row)| {
        let wr = 0.5 * (1.0 - (c1 * i as f64).cos());
        for (d, &c) in row.iter_mut().zip(&wc) {
            *d = ((wr * c) as f32).sqrt();
        }
    });
    out
}

type C32 = Complex32;

/// One planner of each kind for every call: they keep the plans they have made.
fn plan(len: usize, inverse: bool) -> Arc<dyn Fft<f32>> {
    static PLANNER: OnceLock<Mutex<FftPlanner<f32>>> = OnceLock::new();
    let mut planner = PLANNER
        .get_or_init(|| Mutex::new(FftPlanner::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if inverse {
        planner.plan_fft_inverse(len)
    } else {
        planner.plan_fft_forward(len)
    }
}

fn real_planner() -> MutexGuard<'static, RealFftPlanner<f32>> {
    static PLANNER: OnceLock<Mutex<RealFftPlanner<f32>>> = OnceLock::new();
    PLANNER
        .get_or_init(|| Mutex::new(RealFftPlanner::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn fft_rows(data: &mut [C32], len: usize, fft: &Arc<dyn Fft<f32>>) {
    data.par_chunks_mut(len).with_min_len(16).for_each_init(
        || vec![C32::default(); fft.get_inplace_scratch_len()],
        |scratch, row| fft.process_with_scratch(row, scratch),
    );
}

/// `src` (rows x cols) transposed, in 32 x 32 tiles.
fn transpose(src: &[C32], rows: usize, cols: usize) -> Vec<C32> {
    const T: usize = 32;
    let mut dst = vec![C32::default(); src.len()];
    dst.par_chunks_mut(rows * T)
        .enumerate()
        .for_each(|(bx, band)| {
            let x0 = bx * T;
            let bw = band.len() / rows;
            for y0 in (0..rows).step_by(T) {
                for x in 0..bw {
                    for y in y0..(y0 + T).min(rows) {
                        band[x * rows + y] = src[y * cols + x0 + x];
                    }
                }
            }
        });
    dst
}

/// The 2-D DFT of the windowed h x w plane zero-padded to m x n, as its n / 2 + 1 columns
/// (a real plane's spectrum holds the rest as conjugates), stored column after column.
fn forward(img: &[f32], win: &[f32], (h, w): (usize, usize), (m, n): (usize, usize)) -> Vec<C32> {
    let k = n / 2 + 1;
    let r2c = real_planner().plan_fft_forward(n);
    let mut spec = vec![C32::default(); m * k];
    spec.par_chunks_mut(k).take(h).enumerate().for_each_init(
        || (r2c.make_input_vec(), r2c.make_scratch_vec()),
        |(input, scratch), (y, out)| {
            let (src, wv) = (&img[y * w..][..w], &win[y * w..][..w]);
            for (d, (&p, &q)) in input.iter_mut().zip(src.iter().zip(wv)) {
                *d = p * q;
            }
            input[w..].fill(0.0);
            // Fails only on buffer lengths, which make_*_vec sets.
            let _ = r2c.process_with_scratch(input, out, scratch);
        },
    );
    let mut cols = transpose(&spec, m, k);
    fft_rows(&mut cols, m, &plan(m, false));
    cols
}

/// The real m x n inverse DFT, unscaled, of n / 2 + 1 spectrum columns stored column after column.
fn inverse(mut cols: Vec<C32>, (m, n): (usize, usize)) -> Vec<f32> {
    let k = n / 2 + 1;
    fft_rows(&mut cols, m, &plan(m, true));
    let mut spec = transpose(&cols, k, m);
    let c2r = real_planner().plan_fft_inverse(n);
    let mut out = vec![0f32; m * n];
    out.par_chunks_mut(n)
        .zip(spec.par_chunks_mut(k))
        .for_each_init(
            || c2r.make_scratch_vec(),
            |scratch, (o, s)| {
                // Rounding leaves the DC and Nyquist bins a trace of imaginary part; theirs is 0.
                s[0].im = 0.0;
                if n % 2 == 0 {
                    s[k - 1].im = 0.0;
                }
                let _ = c2r.process_with_scratch(s, o, scratch);
            },
        );
    out
}

/// `cv2.phaseCorrelate(a, b, window)` of two h x w float32 planes with the Hanning window:
/// the shift of `b` against `a` as (dx, dy), and the peak response. cv2's is half a pixel off
/// along an axis whose padded size is odd.
pub fn phase_correlate(a: &[f32], b: &[f32], h: usize, w: usize) -> (f64, f64, f64) {
    correlate(a, b, (h, w), None)
}

/// `phase_correlate` less an `expected` (dx, dy) shift, which is taken out of the spectrum
/// before the peak is read. The peak fit is exact only near a whole pixel, so a known
/// fractional shift is measured as its remainder.
pub fn phase_correlate_about(
    a: &[f32],
    b: &[f32],
    (h, w): (usize, usize),
    expected: (f64, f64),
) -> (f64, f64, f64) {
    correlate(a, b, (h, w), Some(expected))
}

fn correlate(
    a: &[f32],
    b: &[f32],
    (h, w): (usize, usize),
    expected: Option<(f64, f64)>,
) -> (f64, f64, f64) {
    let win = hanning_window(h, w);
    let (m, n) = (optimal_dft_size(h), optimal_dft_size(w));
    let (mut fa, fb) = rayon::join(
        || forward(a, &win, (h, w), (m, n)),
        || forward(b, &win, (h, w), (m, n)),
    );
    // In f64: at 16-bit scale |v|^2 of the DC bin passes f32::MAX.
    let unit = |p: C32, q: C32| {
        let v = p * q.conj();
        let (re, im) = (v.re as f64, v.im as f64);
        let sq = re * re + im * im;
        let k = sq.sqrt() / (sq + f32::EPSILON as f64);
        (re * k, im * k)
    };
    match expected {
        None => fa
            .par_chunks_mut(m)
            .zip(fb.par_chunks(m))
            .for_each(move |(p, q)| {
                for (p, q) in p.iter_mut().zip(q) {
                    let (re, im) = unit(*p, *q);
                    *p = C32::new(re as f32, im as f32);
                }
            }),
        Some((dx, dy)) => {
            // The ramp of a shift is one turn per axis, so each bin multiplies two table entries.
            let turn = |t: f64| libm::sincos(-std::f64::consts::TAU * t);
            let along_x: Vec<(f64, f64)> = (0..fa.len() / m)
                .map(|x| turn(x as f64 * dx / n as f64))
                .collect();
            let along_y: Vec<(f64, f64)> = (0..m)
                .map(|y| {
                    let v = if 2 * y > m {
                        y as f64 - m as f64
                    } else {
                        y as f64
                    };
                    turn(v * dy / m as f64)
                })
                .collect();
            let (along_x, along_y) = (&along_x, &along_y);
            fa.par_chunks_mut(m)
                .zip(fb.par_chunks(m))
                .enumerate()
                .for_each(move |(x, (p, q))| {
                    let (sin_x, cos_x) = along_x[x];
                    for (y, (p, q)) in p.iter_mut().zip(q).enumerate() {
                        // A Nyquist bin has no sign, so a fractional shift has no phase there.
                        if 2 * x == n || 2 * y == m {
                            *p = C32::default();
                            continue;
                        }
                        let (re, im) = unit(*p, *q);
                        let (sin_y, cos_y) = along_y[y];
                        let (cos, sin) =
                            (cos_x * cos_y - sin_x * sin_y, sin_x * cos_y + cos_x * sin_y);
                        *p = C32::new((re * cos - im * sin) as f32, (re * sin + im * cos) as f32);
                    }
                });
        }
    }
    let corr = inverse(fa, (m, n));

    // fftShift rolls by (n / 2, m / 2). cv2 then measures from (n / 2.0, m / 2.0), half a pixel
    // off on an odd padded size; the roll's own origin is the true zero shift.
    let (xm, ym) = (n / 2, m / 2);
    let mut c = vec![0f64; m * n];
    c.par_chunks_mut(n).enumerate().for_each(|(y, row)| {
        let src = &corr[((y + m - ym) % m) * n..][..n];
        let (head, tail) = row.split_at_mut(xm);
        for (d, &v) in head.iter_mut().zip(&src[n - xm..]) {
            *d = v as f64;
        }
        for (d, &v) in tail.iter_mut().zip(src) {
            *d = v as f64;
        }
    });

    let peak = c
        .iter()
        .enumerate()
        .fold(0, |best, (i, &v)| if v > c[best] { i } else { best });
    let (py, px) = ((peak / n) as i64, (peak % n) as i64);
    let (y0, y1) = (
        (py - 2).max(0) as usize,
        (py + 2).min(m as i64 - 1) as usize,
    );
    let (x0, x1) = (
        (px - 2).max(0) as usize,
        (px + 2).min(n as i64 - 1) as usize,
    );
    let (mut cx, mut cy, mut sum) = (0f64, 0f64, 0f64);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let v = c[y * n + x];
            cx += x as f64 * v;
            cy += y as f64 * v;
            sum += v;
        }
    }
    let response = sum / (m * n) as f64;
    let sum = sum + f64::EPSILON;
    (xm as f64 - cx / sum, ym as f64 - cy / sum, response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_known_shift_reads_as_its_remainder() {
        let (h, w) = (64usize, 96usize);
        // A fixed pseudo-random plane, moved three columns with wrap-around.
        let mut state = 7u32;
        let base: Vec<f32> = (0..h * w)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f32
            })
            .collect();
        let plane = |shift: usize| -> Vec<f32> {
            (0..h * w)
                .map(|i| base[(i / w) * w + (i % w + shift) % w])
                .collect()
        };
        let (a, b) = (plane(0), plane(3));
        let (dx, dy, _) = phase_correlate(&a, &b, h, w);
        let (rx, ry, response) = phase_correlate_about(&a, &b, (h, w), (dx.round(), 0.0));
        assert!((dx.abs() - 3.0).abs() < 0.2 && dy.abs() < 0.2, "{dx} {dy}");
        assert!(
            rx.abs() < 0.2 && ry.abs() < 0.2 && response > 0.3,
            "{rx} {ry} {response}"
        );
    }

    #[test]
    fn optimal_sizes() {
        assert_eq!(optimal_dft_size(1024), 1024);
        assert_eq!(optimal_dft_size(1025), 1080);
        assert_eq!(optimal_dft_size(7), 8);
        assert_eq!(optimal_dft_size(361), 375);
    }

    #[test]
    fn finds_an_integer_shift() {
        let (h, w) = (64, 80);
        let mut seed = 12345u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let field: Vec<f32> = (0..(h + 8) * (w + 8)).map(|_| noise()).collect();
        let at = |x: usize, y: usize| field[y * (w + 8) + x];
        let a: Vec<f32> = (0..h * w).map(|i| at(i % w + 4, i / w + 4)).collect();
        let b: Vec<f32> = (0..h * w).map(|i| at(i % w + 1, i / w + 6)).collect();
        let (dx, dy, _) = phase_correlate(&a, &b, h, w);
        assert!(
            (dx - 3.0).abs() < 0.1 && (dy + 2.0).abs() < 0.1,
            "{dx} {dy}"
        );
    }
}
