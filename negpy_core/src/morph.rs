//! OpenCV's erode, dilate and morphologyEx for rectangular kernels. A min or max has no
//! rounding, so a separable row-then-column pass gives OpenCV's bits.

use multiversion::multiversion;
use rayon::prelude::*;

use crate::filter::{Border, border_interpolate};

pub trait Pixel: Copy + PartialOrd + Send + Sync {
    /// Erode's identity, OpenCV's default constant border for it.
    const HIGH: Self;
    /// Dilate's identity.
    const LOW: Self;
    fn sub(self, other: Self) -> Self;
    fn lo(self, other: Self) -> Self;
    fn hi(self, other: Self) -> Self;
}

impl Pixel for u8 {
    const HIGH: Self = u8::MAX;
    const LOW: Self = u8::MIN;
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
    fn lo(self, other: Self) -> Self {
        self.min(other)
    }
    fn hi(self, other: Self) -> Self {
        self.max(other)
    }
}

impl Pixel for f32 {
    const HIGH: Self = f32::INFINITY;
    const LOW: Self = f32::NEG_INFINITY;
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn lo(self, other: Self) -> Self {
        if other < self { other } else { self }
    }
    fn hi(self, other: Self) -> Self {
        if other > self { other } else { self }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Op {
    Erode,
    Dilate,
    Open,
    Close,
    Gradient,
    TopHat,
    BlackHat,
}

impl Op {
    /// From `cv2.MORPH_*`.
    pub fn from_cv(code: i32) -> Option<Self> {
        Some(match code {
            0 => Op::Erode,
            1 => Op::Dilate,
            2 => Op::Open,
            3 => Op::Close,
            4 => Op::Gradient,
            5 => Op::TopHat,
            6 => Op::BlackHat,
            _ => return None,
        })
    }
}

#[inline(always)]
fn pick<T: Pixel, const ERODE: bool>(a: T, b: T) -> T {
    if ERODE { a.lo(b) } else { a.hi(b) }
}

/// `out[i] = extreme(out[i], line[i])`; AVX2 doubles the lanes of the baseline build.
#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn extreme_into<T: Pixel, const ERODE: bool>(out: &mut [T], line: &[T]) {
    for (o, &v) in out.iter_mut().zip(line) {
        *o = pick::<T, ERODE>(*o, v);
    }
}

/// `out[i]` = the extreme of `out[i]` and every `line[i]`, a register-sized chunk at a time
/// so the accumulator never leaves the registers between taps.
#[multiversion(targets("x86_64+avx2", "aarch64+neon"))]
fn taps_into<T: Pixel, const ERODE: bool>(out: &mut [T], lines: &[&[T]]) {
    const CHUNK: usize = 64;
    let n = out.len();
    let mut x = 0;
    while x + CHUNK <= n {
        let mut acc: [T; CHUNK] = std::array::from_fn(|i| out[x + i]);
        for line in lines {
            let l = &line[x..x + CHUNK];
            for i in 0..CHUNK {
                acc[i] = pick::<T, ERODE>(acc[i], l[i]);
            }
        }
        out[x..x + CHUNK].copy_from_slice(&acc);
        x += CHUNK;
    }
    for line in lines {
        for (o, &v) in out[x..].iter_mut().zip(&line[x..]) {
            *o = pick::<T, ERODE>(*o, v);
        }
    }
}

/// `out[i] = min (or max) of src[i + j * stride]` for `j < len`, from overlapping power-of-two
/// windows: a min or max is idempotent, so the overlap changes nothing and a window of
/// `len` costs log2(len) vector passes.
#[inline(always)]
fn window<T: Pixel, const ERODE: bool>(
    cur: &mut Vec<T>,
    scratch: &mut Vec<T>,
    out: &mut [T],
    len: usize,
    stride: usize,
) {
    let mut win = 1;
    while win * 2 <= len {
        let off = win * stride;
        let valid = cur.len() - off;
        scratch.resize(valid, cur[0]);
        for ((o, &x), &y) in scratch.iter_mut().zip(&cur[..valid]).zip(&cur[off..]) {
            *o = pick::<T, ERODE>(x, y);
        }
        std::mem::swap(cur, scratch);
        win *= 2;
    }
    let off = (len - win) * stride;
    for ((o, &x), &y) in out.iter_mut().zip(cur.iter()).zip(&cur[off..]) {
        *o = pick::<T, ERODE>(x, y);
    }
}

/// One erode (`ERODE`) or dilate pass with a `kw` x `kh` rectangle anchored at `(ax, ay)`.
fn rect_pass<T: Pixel, const ERODE: bool>(
    src: &[T],
    dst: &mut [T],
    (h, w, cn): (usize, usize, usize),
    (kw, kh): (usize, usize),
    (ax, ay): (usize, usize),
    border: Border,
) {
    let n = w * cn;
    let identity = if ERODE { T::HIGH } else { T::LOW };
    let xmap: Vec<Option<usize>> = (0..w + kw - 1)
        .map(|px| border_interpolate(px as isize - ax as isize, w, border))
        .collect();

    // Bands of output rows, each with its own row pass over the kh - 1 rows of halo, so the
    // intermediate rows stay in cache.
    let band = (2 * kh).max(32);
    dst.par_chunks_mut(band * n).enumerate().for_each_init(
        || (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        |(padded, scratch, cur, next): &mut (Vec<T>, Vec<T>, Vec<T>, Vec<T>), (bi, out)| {
            let total = out.len() / n + kh - 1;
            cur.clear();
            cur.resize(total * n, identity);
            for (t, line) in cur.chunks_exact_mut(n).enumerate() {
                let Some(y) = border_interpolate((bi * band + t) as isize - ay as isize, h, border)
                else {
                    continue;
                };
                padded.resize((w + kw - 1) * cn, identity);
                let row = &src[y * n..(y + 1) * n];
                padded[ax * cn..(ax + w) * cn].copy_from_slice(row);
                for px in (0..ax).chain(ax + w..w + kw - 1) {
                    let chunk = &mut padded[px * cn..(px + 1) * cn];
                    match xmap[px] {
                        Some(sx) => chunk.copy_from_slice(&row[sx * cn..(sx + 1) * cn]),
                        None => chunk.fill(identity),
                    }
                }
                window::<T, ERODE>(padded, scratch, line, kw, cn);
            }
            window::<T, ERODE>(cur, next, out, kh, n);
        },
    );
}

/// `cv2.erode`/`cv2.dilate` with a full `kw` x `kh` rectangle and `iterations`: OpenCV turns
/// iterations of a rectangle into one larger rectangle with the anchor scaled.
fn erode_dilate<T: Pixel>(
    src: &[T],
    dst: &mut [T],
    dims: (usize, usize, usize),
    ksize: (usize, usize),
    iterations: usize,
    erode: bool,
    border: Border,
) {
    let (kw, kh) = ksize;
    let (mut ax, mut ay) = (kw / 2, kh / 2);
    let (mut kw, mut kh) = (kw, kh);
    if iterations == 0 || (kw == 1 && kh == 1) {
        dst.copy_from_slice(src);
        return;
    }
    if iterations > 1 {
        ax *= iterations;
        ay *= iterations;
        kw += (iterations - 1) * (kw - 1);
        kh += (iterations - 1) * (kh - 1);
    }
    if erode {
        rect_pass::<T, true>(src, dst, dims, (kw, kh), (ax, ay), border);
    } else {
        rect_pass::<T, false>(src, dst, dims, (kw, kh), (ax, ay), border);
    }
}

/// `cv2.morphologyEx(src, op, ones((kh, kw)), iterations=..., borderType=...)`, the border value
/// OpenCV's default.
pub fn morphology<T: Pixel>(
    src: &[T],
    dst: &mut [T],
    dims: (usize, usize, usize),
    op: Op,
    ksize: (usize, usize),
    iterations: usize,
    border: Border,
) -> Option<()> {
    let (h, w, cn) = dims;
    if h == 0
        || w == 0
        || cn == 0
        || ksize.0 == 0
        || ksize.1 == 0
        || src.len() != h * w * cn
        || dst.len() != src.len()
    {
        return None;
    }
    compose(src, dst, op, |s: &[T], d: &mut [T], erode: bool| {
        erode_dilate(s, d, dims, ksize, iterations, erode, border)
    });
    Some(())
}

/// An operation built from erode (`true`) and dilate passes.
fn compose<T: Pixel>(src: &[T], dst: &mut [T], op: Op, ed: impl Fn(&[T], &mut [T], bool)) {
    match op {
        Op::Erode => return ed(src, dst, true),
        Op::Dilate => return ed(src, dst, false),
        _ => {}
    }
    let mut tmp = vec![src[0]; src.len()];
    match op {
        Op::Erode | Op::Dilate => {}
        Op::Open => {
            ed(src, &mut tmp, true);
            ed(&tmp, dst, false);
        }
        Op::Close => {
            ed(src, &mut tmp, false);
            ed(&tmp, dst, true);
        }
        Op::Gradient => {
            ed(src, &mut tmp, false);
            ed(src, dst, true);
            crate::par_zip(dst, &tmp, move |d, &t| *d = t.sub(*d));
        }
        Op::TopHat => {
            ed(src, &mut tmp, true);
            ed(&tmp, dst, false);
            crate::par_zip(dst, src, move |d, &s| *d = s.sub(*d));
        }
        Op::BlackHat => {
            ed(src, &mut tmp, false);
            ed(&tmp, dst, true);
            crate::par_zip(dst, src, move |d, &s| *d = d.sub(s));
        }
    }
}

/// One erode or dilate pass with an arbitrary `kw` x `kh` kernel mask (nonzero = in), anchor at
/// the center: each kernel row splits into runs, a run is a sliding extreme along the row, and
/// the output takes the extreme over the runs.
fn mask_pass<T: Pixel, const ERODE: bool>(
    src: &[T],
    dst: &mut [T],
    (h, w, cn): (usize, usize, usize),
    mask: &[u8],
    (kw, kh): (usize, usize),
    border: Border,
) {
    let (ax, ay) = (kw / 2, kh / 2);
    let identity = if ERODE { T::HIGH } else { T::LOW };
    let n = w * cn;
    let pw = (w + kw - 1) * cn;
    // (kernel row, run start, run length)
    let mut runs: Vec<(usize, usize, usize)> = Vec::new();
    for j in 0..kh {
        let mut x = 0;
        while x < kw {
            if mask[j * kw + x] != 0 {
                let start = x;
                while x < kw && mask[j * kw + x] != 0 {
                    x += 1;
                }
                runs.push((j, start, x - start));
            } else {
                x += 1;
            }
        }
    }
    if runs.is_empty() {
        dst.fill(identity);
        return;
    }
    let xmap: Vec<Option<usize>> = (0..w + kw - 1)
        .map(|px| border_interpolate(px as isize - ax as isize, w, border))
        .collect();
    let pad_row = |row: &[T], padded: &mut [T]| {
        padded[ax * cn..ax * cn + n].copy_from_slice(row);
        for px in (0..ax).chain(ax + w..w + kw - 1) {
            if let Some(sx) = xmap[px] {
                padded[px * cn..(px + 1) * cn].copy_from_slice(&row[sx * cn..(sx + 1) * cn]);
            }
        }
    };
    let cells: usize = runs.iter().map(|r| r.2).sum();
    if cells <= 64 {
        // Small kernels: one vector min or max per kernel cell, as OpenCV. Interior columns
        // read the source rows directly; the few columns whose taps cross the border go
        // through the border map.
        let taps: Vec<(usize, usize)> = runs
            .iter()
            .flat_map(|&(j, start, len)| (start..start + len).map(move |x| (j, x)))
            .collect();
        let (left, right) = (ax.min(w), (kw - 1 - ax).min(w - ax.min(w)));
        dst.par_chunks_mut(n).enumerate().for_each(|(y, out)| {
            out.fill(identity);
            let rows: Vec<Option<usize>> = (0..kh)
                .map(|j| border_interpolate((y + j) as isize - ay as isize, h, border))
                .collect();
            let inner = left * cn..n - right * cn;
            if inner.start < inner.end {
                let lines: Vec<&[T]> = taps
                    .iter()
                    .filter_map(|&(j, x)| {
                        rows[j].map(|sy| {
                            let at = sy * n + inner.start + x * cn - ax * cn;
                            &src[at..at + inner.len()]
                        })
                    })
                    .collect();
                taps_into::<T, ERODE>(&mut out[inner.clone()], &lines);
            }
            for px in (0..left).chain(w - right..w) {
                for &(j, x) in &taps {
                    let (Some(sy), Some(sx)) = (rows[j], xmap[px + x]) else {
                        continue;
                    };
                    for c in 0..cn {
                        let o = &mut out[px * cn + c];
                        *o = pick::<T, ERODE>(*o, src[sy * n + sx * cn + c]);
                    }
                }
            }
        });
        return;
    }
    // Large kernels: each output row recomputes its runs' sliding extremes from the padded
    // source rows, a log-time window per run and no full-image intermediate.
    dst.par_chunks_mut(n).enumerate().for_each_init(
        || (Vec::new(), Vec::new(), Vec::new(), Vec::new(), usize::MAX),
        |(padded, cur, scratch, slid, padded_row): &mut (Vec<T>, Vec<T>, Vec<T>, Vec<T>, usize),
         (y, out)| {
            out.fill(identity);
            for &(j, start, len) in &runs {
                let Some(sy) = border_interpolate((y + j) as isize - ay as isize, h, border) else {
                    // A constant border row holds the identity, which changes nothing.
                    continue;
                };
                if *padded_row != sy {
                    padded.clear();
                    padded.resize(pw, identity);
                    pad_row(&src[sy * n..(sy + 1) * n], padded);
                    *padded_row = sy;
                }
                cur.clear();
                cur.extend_from_slice(&padded[start * cn..start * cn + n + (len - 1) * cn]);
                slid.clear();
                slid.resize(n, identity);
                window::<T, ERODE>(cur, scratch, slid, len, cn);
                for (o, &v) in out.iter_mut().zip(slid.iter()) {
                    *o = pick::<T, ERODE>(*o, v);
                }
            }
        },
    );
}

/// A structuring element: `h` rows of `w` cells, nonzero = in.
#[derive(Clone, Copy)]
pub struct Kernel<'a> {
    pub mask: &'a [u8],
    pub w: usize,
    pub h: usize,
}

/// `cv2.morphologyEx(src, op, kernel, iterations=..., borderType=...)` for an arbitrary kernel,
/// OpenCV's default anchor and border value. Iterations repeat the pass, as OpenCV does for a
/// kernel that is not a full rectangle.
pub fn morphology_mask<T: Pixel>(
    src: &[T],
    dst: &mut [T],
    dims: (usize, usize, usize),
    op: Op,
    kernel: Kernel,
    iterations: usize,
    border: Border,
) -> Option<()> {
    let (h, w, cn) = dims;
    let (mask, kw, kh) = (kernel.mask, kernel.w, kernel.h);
    let ksize = (kw, kh);
    if h == 0
        || w == 0
        || cn == 0
        || kw == 0
        || kh == 0
        || mask.len() != kw * kh
        || src.len() != h * w * cn
        || dst.len() != src.len()
    {
        return None;
    }
    compose(src, dst, op, |s: &[T], d: &mut [T], erode: bool| {
        let pass = |from: &[T], to: &mut [T]| {
            if erode {
                mask_pass::<T, true>(from, to, dims, mask, ksize, border);
            } else {
                mask_pass::<T, false>(from, to, dims, mask, ksize, border);
            }
        };
        if iterations == 0 {
            d.copy_from_slice(s);
            return;
        }
        pass(s, d);
        if iterations > 1 {
            let mut cur = d.to_vec();
            for _ in 1..iterations {
                pass(&cur, d);
                cur.copy_from_slice(d);
            }
        }
    });
    Some(())
}

/// `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w, h))`: `h` rows of `w` cells.
pub fn ellipse_kernel(w: usize, h: usize) -> Vec<u8> {
    let (r, c) = ((h / 2) as i64, (w / 2) as i64);
    let inv_r2 = if r > 0 { 1.0 / (r * r) as f64 } else { 0.0 };
    let mut out = vec![0u8; w * h];
    for (i, row) in out.chunks_mut(w.max(1)).enumerate() {
        let dy = i as i64 - r;
        if dy.abs() > r {
            continue;
        }
        // OpenCV's saturate_cast<int> rounds half to even.
        let dx = (c as f64 * (((r * r - dy * dy) as f64) * inv_r2).sqrt()).round_ties_even() as i64;
        let (j1, j2) = ((c - dx).max(0) as usize, ((c + dx + 1) as usize).min(w));
        row[j1..j2].fill(1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erode_takes_the_window_minimum() {
        let src: Vec<u8> = vec![5, 9, 1, 7, 3, 8];
        let mut dst = vec![0u8; 6];
        morphology(
            &src,
            &mut dst,
            (1, 6, 1),
            Op::Erode,
            (3, 1),
            1,
            Border::Constant,
        )
        .unwrap();
        assert_eq!(dst, vec![5, 1, 1, 1, 3, 3]);
        morphology(
            &src,
            &mut dst,
            (1, 6, 1),
            Op::Dilate,
            (3, 1),
            1,
            Border::Constant,
        )
        .unwrap();
        assert_eq!(dst, vec![9, 9, 9, 7, 8, 8]);
    }

    #[test]
    fn blackhat_of_a_flat_image_is_zero() {
        let src = vec![0.5f32; 7 * 9];
        let mut dst = vec![1f32; 7 * 9];
        morphology(
            &src,
            &mut dst,
            (7, 9, 1),
            Op::BlackHat,
            (3, 3),
            2,
            Border::Constant,
        )
        .unwrap();
        assert!(dst.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ellipse_matches_opencv() {
        assert_eq!(ellipse_kernel(1, 1), vec![1]);
        #[rustfmt::skip]
        let want = vec![
            0, 0, 1, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
        ];
        assert_eq!(ellipse_kernel(5, 5), want);
    }
}
