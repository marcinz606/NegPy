//! OpenCV's warps of a float32 image: `remap`, `warpAffine` and `warpPerspective`, bilinear or
//! bicubic, with replicated edges or a zero border. Each source coordinate is fixed point at
//! 1/32 px, rounded as OpenCV rounds it; the weights are its table's float32 products, summed
//! in its order.

use multiversion::multiversion;
use rayon::prelude::*;

/// An HxWxC float32 image.
#[derive(Clone, Copy)]
pub struct Image<'a> {
    pub data: &'a [f32],
    pub h: usize,
    pub w: usize,
    pub cn: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Interp {
    /// `remap` only: the source pixel nearest the map point.
    Nearest,
    Linear,
    Cubic,
}

/// What a sample outside the source reads: the nearest edge pixel, or zero.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Border {
    Replicate,
    Constant,
}

const BITS: i32 = 5;
const MASK: i32 = (1 << BITS) - 1;
const STEP: f32 = 1.0 / (1 << BITS) as f32;

/// OpenCV's bicubic weight table: for each pair of 1/32 px fractions (y major), the products
/// of `interpolateCubic` (A = -0.75) of the y and the x fraction, row major.
type CubicTable = [[f32; 16]; 1 << (2 * BITS)];

fn cubic_table() -> &'static CubicTable {
    static TAB: std::sync::OnceLock<Box<CubicTable>> = std::sync::OnceLock::new();
    TAB.get_or_init(|| {
        const A: f32 = -0.75;
        let c: [[f32; 4]; 1 << BITS] = std::array::from_fn(|i| {
            let x = i as f32 * STEP;
            let c0 = ((A * (x + 1.0) - 5.0 * A) * (x + 1.0) + 8.0 * A) * (x + 1.0) - 4.0 * A;
            let c1 = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
            let c2 = ((A + 2.0) * (1.0 - x) - (A + 3.0)) * (1.0 - x) * (1.0 - x) + 1.0;
            [c0, c1, c2, 1.0 - c0 - c1 - c2]
        });
        let mut tab = Box::new([[0f32; 16]; 1 << (2 * BITS)]);
        for (f, w) in tab.iter_mut().enumerate() {
            let (cy, cx) = (c[f >> BITS], c[f & MASK as usize]);
            *w = std::array::from_fn(|i| cy[i / 4] * cx[i % 4]);
        }
        tab
    })
}

/// The source index of `v` on an axis of `n` under `border`; None reads zero.
#[inline(always)]
fn edge(v: i32, n: usize, border: Border) -> Option<usize> {
    match border {
        Border::Replicate => Some(v.clamp(0, n as i32 - 1) as usize),
        Border::Constant => (v >= 0 && (v as usize) < n).then_some(v as usize),
    }
}

/// The bilinear pixel at fixed-point source coordinates (x, y) into `out`. `CN` is the channel
/// count, or 0 for one known only at run time.
#[inline(always)]
fn linear<const CN: usize>(src: &Image, border: Border, x: i32, y: i32, out: &mut [f32]) {
    let (fx, fy) = ((x & MASK) as f32 * STEP, (y & MASK) as f32 * STEP);
    let (ix, iy) = (x >> BITS, y >> BITS);
    let w0 = (1.0 - fy) * (1.0 - fx);
    let w1 = (1.0 - fy) * fx;
    let w2 = fy * (1.0 - fx);
    let w3 = fy * fx;
    let cn = if CN == 0 { src.cn } else { CN };
    let row = src.w * cn;
    let s = src.data;
    if (ix as u32) < src.w as u32 - 1 && (iy as u32) < src.h as u32 - 1 {
        let a = iy as usize * row + ix as usize * cn;
        let (p, q) = (&s[a..][..2 * cn], &s[a + row..][..2 * cn]);
        for (k, o) in out[..cn].iter_mut().enumerate() {
            *o = p[k] * w0 + p[cn + k] * w1 + q[k] * w2 + q[cn + k] * w3;
        }
        return;
    }
    let (wi, hi) = (src.w as i32, src.h as i32);
    if border == Border::Constant && (ix >= wi || ix + 1 < 0 || iy >= hi || iy + 1 < 0) {
        out[..cn].fill(0.0);
        return;
    }
    let (x0, x1) = (edge(ix, src.w, border), edge(ix + 1, src.w, border));
    let (y0, y1) = (edge(iy, src.h, border), edge(iy + 1, src.h, border));
    let at = |yy: Option<usize>, xx: Option<usize>, k: usize| match (yy, xx) {
        (Some(yy), Some(xx)) => s[yy * row + xx * cn + k],
        _ => 0.0,
    };
    for (k, o) in out[..cn].iter_mut().enumerate() {
        *o = at(y0, x0, k) * w0 + at(y0, x1, k) * w1 + at(y1, x0, k) * w2 + at(y1, x1, k) * w3;
    }
}

/// The bicubic pixel at fixed-point source coordinates (x, y): rows of four summed, then added
/// row by row; at the border every term is added in turn, as OpenCV does.
#[inline(always)]
fn cubic<const CN: usize>(
    src: &Image,
    border: Border,
    tab: &CubicTable,
    x: i32,
    y: i32,
    out: &mut [f32],
) {
    let w = &tab[(((y & MASK) << BITS) | (x & MASK)) as usize];
    let (sx, sy) = ((x >> BITS) - 1, (y >> BITS) - 1);
    let cn = if CN == 0 { src.cn } else { CN };
    let row = src.w * cn;
    let s = src.data;
    if (sx as u32) < (src.w as u32).saturating_sub(3)
        && (sy as u32) < (src.h as u32).saturating_sub(3)
    {
        let a = sy as usize * row + sx as usize * cn;
        let p: [&[f32]; 4] = std::array::from_fn(|i| &s[a + i * row..][..4 * cn]);
        for (k, o) in out[..cn].iter_mut().enumerate() {
            let r = |i: usize| {
                let (p, w) = (p[i], &w[i * 4..][..4]);
                p[k] * w[0] + p[cn + k] * w[1] + p[2 * cn + k] * w[2] + p[3 * cn + k] * w[3]
            };
            let mut sum = r(0);
            sum += r(1);
            sum += r(2);
            sum += r(3);
            *o = sum;
        }
        return;
    }
    let (wi, hi) = (src.w as i32, src.h as i32);
    if border == Border::Constant && (sx >= wi || sx + 4 <= 0 || sy >= hi || sy + 4 <= 0) {
        out[..cn].fill(0.0);
        return;
    }
    let xs: [Option<usize>; 4] = std::array::from_fn(|i| edge(sx + i as i32, src.w, border));
    let ys: [Option<usize>; 4] = std::array::from_fn(|i| edge(sy + i as i32, src.h, border));
    for (k, o) in out[..cn].iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (i, yy) in ys.iter().enumerate() {
            let Some(yy) = yy else { continue };
            for (j, xx) in xs.iter().enumerate() {
                if let Some(xx) = xx {
                    sum += s[yy * row + xx * cn + k] * w[i * 4 + j];
                }
            }
        }
        *o = sum;
    }
}

/// The source pixel at whole-pixel coordinates (x, y) in 1/32 px units.
#[inline(always)]
fn nearest<const CN: usize>(src: &Image, border: Border, x: i32, y: i32, out: &mut [f32]) {
    let cn = if CN == 0 { src.cn } else { CN };
    let (ix, iy) = (x >> BITS, y >> BITS);
    if (ix as u32) < src.w as u32 && (iy as u32) < src.h as u32 {
        out[..cn].copy_from_slice(&src.data[(iy as usize * src.w + ix as usize) * cn..][..cn]);
        return;
    }
    match (
        edge(x >> BITS, src.w, border),
        edge(y >> BITS, src.h, border),
    ) {
        (Some(xx), Some(yy)) => {
            out[..cn].copy_from_slice(&src.data[(yy * src.w + xx) * cn..][..cn]);
        }
        _ => out[..cn].fill(0.0),
    }
}

fn sample_row<const CN: usize>(
    src: &Image,
    (interp, border): (Interp, Border),
    tab: &CubicTable,
    (xs, ys): (&[i32], &[i32]),
    row: &mut [f32],
) {
    let cn = if CN == 0 { src.cn } else { CN };
    let px = row.chunks_exact_mut(cn).zip(xs).zip(ys);
    match interp {
        Interp::Linear => px.for_each(|((o, &x), &y)| linear::<CN>(src, border, x, y, o)),
        Interp::Cubic => px.for_each(|((o, &x), &y)| cubic::<CN>(src, border, tab, x, y, o)),
        Interp::Nearest => px.for_each(|((o, &x), &y)| nearest::<CN>(src, border, x, y, o)),
    }
}

static EMPTY: CubicTable = [[0.0; 16]; 1 << (2 * BITS)];

/// Fills a `dw`-wide output, row by row in parallel, from `coords(y, xs, ys)`, which writes the
/// fixed-point source coordinates of output row y.
fn warp(
    src: &Image,
    mode: (Interp, Border),
    dw: usize,
    out: &mut [f32],
    coords: impl Fn(usize, &mut [i32], &mut [i32]) + Sync,
) {
    let cn = src.cn;
    let tab = if mode.0 == Interp::Cubic {
        cubic_table()
    } else {
        &EMPTY
    };
    out.par_chunks_mut(dw * cn).enumerate().for_each_init(
        || (vec![0i32; dw], vec![0i32; dw]),
        |(xs, ys), (y, row)| {
            coords(y, xs, ys);
            let c = (&xs[..], &ys[..]);
            match cn {
                1 => sample_row::<1>(src, mode, tab, c, row),
                3 => sample_row::<3>(src, mode, tab, c, row),
                4 => sample_row::<4>(src, mode, tab, c, row),
                _ => sample_row::<0>(src, mode, tab, c, row),
            }
        },
    );
}

/// OpenCV's `cvRound` of a float64 into an int, ties to even, for |v| < 2^30: added to 1.5 * 2^52,
/// the integer fills the low mantissa bits. `round_ties_even` is a libm call on the x86-64
/// baseline, and the saturating `as` cast keeps LLVM from vectorizing.
#[inline(always)]
fn round_i32(v: f64) -> i32 {
    (v + 6755399441055744.0).to_bits() as u32 as i32
}

/// Coordinates far outside any image, held where `round_i32` stays exact.
const FAR: f64 = (1u32 << 30) as f64;

/// `cv2.remap(src, map_x, map_y, interpolation, borderMode)`; the maps are float32, one entry
/// per output pixel, `dw` wide.
pub fn remap(
    src: &Image,
    mode: (Interp, Border),
    (map_x, map_y): (&[f32], &[f32]),
    dw: usize,
    out: &mut [f32],
) {
    // Nearest rounds the map point to a whole pixel.
    let scale = if mode.0 == Interp::Nearest {
        1.0
    } else {
        (1 << BITS) as f32
    };
    let shift = if mode.0 == Interp::Nearest { BITS } else { 0 };
    warp(src, mode, dw, out, move |y, xs, ys| {
        remap_coords(
            &map_x[y * dw..][..dw],
            &map_y[y * dw..][..dw],
            (scale, shift),
            xs,
            ys,
        )
    });
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn remap_coords(
    mx: &[f32],
    my: &[f32],
    (scale, shift): (f32, i32),
    xs: &mut [i32],
    ys: &mut [i32],
) {
    let far = FAR / (1 << shift) as f64;
    for (((x, y), &a), &b) in xs.iter_mut().zip(ys.iter_mut()).zip(mx).zip(my) {
        *x = round_i32(((a * scale) as f64).clamp(-far, far)) << shift;
        *y = round_i32(((b * scale) as f64).clamp(-far, far)) << shift;
    }
}

/// The radial lens correction of a same-size image (`apply_radial_distortion`): the remap
/// source of output pixel p is c + (p - c) s f with f = 1 + k1 |(p - c) s|^2 inv_hd2, in float32
/// as NumPy builds the maps.
pub fn remap_radial(
    src: &Image,
    (cx, cy): (f32, f32),
    s: f32,
    k1: f32,
    inv_hd2: f32,
    out: &mut [f32],
) {
    let dw = src.w;
    warp(
        src,
        (Interp::Linear, Border::Replicate),
        dw,
        out,
        move |y, xs, ys| radial_coords(y as f32, (cx, cy), s, k1, inv_hd2, xs, ys),
    );
}

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn radial_coords(
    y: f32,
    (cx, cy): (f32, f32),
    s: f32,
    k1: f32,
    inv_hd2: f32,
    xs: &mut [i32],
    ys: &mut [i32],
) {
    let scale = (1 << BITS) as f32;
    let py = (y - cy) * s;
    for (i, (x, yv)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
        let px = (i as f32 - cx) * s;
        let f = 1.0 + k1 * (px * px + py * py) * inv_hd2;
        *x = round_i32((((cx + px * f) * scale) as f64).clamp(-FAR, FAR));
        *yv = round_i32((((cy + py * f) * scale) as f64).clamp(-FAR, FAR));
    }
}

/// `cv2.warpAffine(src, m, (dw, dh), interpolation, borderMode)`, bilinear or bicubic; `m` maps
/// source to output unless `inverse_map` (OpenCV's WARP_INVERSE_MAP).
pub fn warp_affine(
    src: &Image,
    mode: (Interp, Border),
    m: [f64; 6],
    inverse_map: bool,
    dw: usize,
    out: &mut [f32],
) {
    let mut m = m;
    if !inverse_map {
        let d = m[0] * m[4] - m[1] * m[3];
        let d = if d != 0.0 { 1.0 / d } else { 0.0 };
        let (a11, a22) = (m[4] * d, m[0] * d);
        m[0] = a11;
        m[1] *= -d;
        m[3] *= -d;
        m[4] = a22;
        let b1 = -m[0] * m[2] - m[1] * m[5];
        let b2 = -m[3] * m[2] - m[4] * m[5];
        m[2] = b1;
        m[5] = b2;
    }
    const AB_SCALE: f64 = 1024.0;
    const ROUND: i32 = 1 << (10 - BITS - 1);
    let adelta: Vec<i32> = (0..dw)
        .map(|x| round_i32((m[0] * x as f64 * AB_SCALE).clamp(-FAR, FAR)))
        .collect();
    let bdelta: Vec<i32> = (0..dw)
        .map(|x| round_i32((m[3] * x as f64 * AB_SCALE).clamp(-FAR, FAR)))
        .collect();
    warp(src, mode, dw, out, move |y, xs, ys| {
        let x0 = round_i32(((m[1] * y as f64 + m[2]) * AB_SCALE).clamp(-FAR, FAR)) + ROUND;
        let y0 = round_i32(((m[4] * y as f64 + m[5]) * AB_SCALE).clamp(-FAR, FAR)) + ROUND;
        for (i, (x, yv)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
            *x = x0.wrapping_add(adelta[i]) >> (10 - BITS);
            *yv = y0.wrapping_add(bdelta[i]) >> (10 - BITS);
        }
    });
}

/// OpenCV's closed-form 3x3 inverse (`cv::invert`, DECOMP_LU, n <= 3); None when singular.
fn invert3(s: [f64; 9]) -> Option<[f64; 9]> {
    let e = |i: usize, j: usize| s[i * 3 + j];
    let d = e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
        - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
        + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0));
    if d == 0.0 {
        return None;
    }
    let d = 1.0 / d;
    Some([
        (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1)) * d,
        (e(0, 2) * e(2, 1) - e(0, 1) * e(2, 2)) * d,
        (e(0, 1) * e(1, 2) - e(0, 2) * e(1, 1)) * d,
        (e(1, 2) * e(2, 0) - e(1, 0) * e(2, 2)) * d,
        (e(0, 0) * e(2, 2) - e(0, 2) * e(2, 0)) * d,
        (e(0, 2) * e(1, 0) - e(0, 0) * e(1, 2)) * d,
        (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0)) * d,
        (e(0, 1) * e(2, 0) - e(0, 0) * e(2, 1)) * d,
        (e(0, 0) * e(1, 1) - e(0, 1) * e(1, 0)) * d,
    ])
}

/// `cv2.warpPerspective(src, m, (dw, dh), interpolation, borderMode)`, bilinear or bicubic; `m`
/// maps source to output unless `inverse_map`. A singular `m` inverts to zeros, as `cv::invert`
/// leaves it.
pub fn warp_perspective(
    src: &Image,
    mode: (Interp, Border),
    m: [f64; 9],
    inverse_map: bool,
    (dw, dh): (usize, usize),
    out: &mut [f32],
) {
    let m = if inverse_map {
        m
    } else {
        invert3(m).unwrap_or([0.0; 9])
    };
    // OpenCV starts the projective sums again at each block of bw0 columns.
    let bh0 = dh.min(16);
    let bw0 = (1024 / bh0).min(dw);
    warp(src, mode, dw, out, move |y, xs, ys| {
        let yf = y as f64;
        for (b, (xs, ys)) in xs.chunks_mut(bw0).zip(ys.chunks_mut(bw0)).enumerate() {
            let xb = (b * bw0) as f64;
            let x0 = m[0] * xb + m[1] * yf + m[2];
            let y0 = m[3] * xb + m[4] * yf + m[5];
            let w0 = m[6] * xb + m[7] * yf + m[8];
            persp_coords([x0, y0, w0], [m[0], m[3], m[6]], xs, ys);
        }
    });
}

/// One block's coordinates from the projective sums `start` at its first column and their
/// steps per column.
#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn persp_coords(start: [f64; 3], step: [f64; 3], xs: &mut [i32], ys: &mut [i32]) {
    for (i, (x, y)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
        let x1 = i as f64;
        let w = start[2] + step[2] * x1;
        let w = if w != 0.0 {
            (1 << BITS) as f64 / w
        } else {
            0.0
        };
        *x = round_i32(((start[0] + step[0] * x1) * w).clamp(-FAR, FAR));
        *y = round_i32(((start[1] + step[1] * x1) * w).clamp(-FAR, FAR));
    }
}

/// `cv2.getRotationMatrix2D(center, angle, scale)`: the 2x3 matrix that turns about `center`
/// by `angle` degrees counter-clockwise. OpenCV takes the center as float32.
pub fn rotation_matrix((cx, cy): (f32, f32), angle: f64, scale: f64) -> [f64; 6] {
    let angle = angle * (std::f64::consts::PI / 180.0);
    let alpha = angle.cos() * scale;
    let beta = angle.sin() * scale;
    let (cx, cy) = (cx as f64, cy as f64);
    [
        alpha,
        beta,
        (1.0 - alpha) * cx - beta * cy,
        -beta,
        alpha,
        beta * cx + (1.0 - alpha) * cy,
    ]
}

/// `cv2.getPerspectiveTransform(src, dst)`: the 3x3 matrix mapping four float32 points onto
/// four, solved with c22 = 1 by OpenCV's LU with partial pivoting. None for a degenerate quad.
pub fn perspective_transform(src: &[[f32; 2]; 4], dst: &[[f32; 2]; 4]) -> Option<[f64; 9]> {
    let mut a = [[0f64; 8]; 8];
    let mut b = [0f64; 8];
    for i in 0..4 {
        let ([x, y], [u, v]) = (src[i], dst[i]);
        a[i][0] = x as f64;
        a[i + 4][3] = x as f64;
        a[i][1] = y as f64;
        a[i + 4][4] = y as f64;
        a[i][2] = 1.0;
        a[i + 4][5] = 1.0;
        // float32 products, as OpenCV multiplies two Point2f fields.
        a[i][6] = (-x * u) as f64;
        a[i][7] = (-y * u) as f64;
        a[i + 4][6] = (-x * v) as f64;
        a[i + 4][7] = (-y * v) as f64;
        b[i] = u as f64;
        b[i + 4] = v as f64;
    }
    let (a0, b0) = (a, b);
    for i in 0..8 {
        let mut k = i;
        for j in i + 1..8 {
            if a[j][i].abs() > a[k][i].abs() {
                k = j;
            }
        }
        if a[k][i].abs() < f64::EPSILON * 100.0 {
            return None;
        }
        if k != i {
            a.swap(i, k);
            b.swap(i, k);
        }
        let d = -1.0 / a[i][i];
        for j in i + 1..8 {
            let alpha = a[j][i] * d;
            let pivot = a[i];
            for (v, p) in a[j][i + 1..].iter_mut().zip(&pivot[i + 1..]) {
                *v += alpha * p;
            }
            b[j] += alpha * b[i];
        }
    }
    for i in (0..8).rev() {
        let mut s = b[i];
        for k in i + 1..8 {
            s -= a[i][k] * b[k];
        }
        b[i] = s / a[i][i];
    }
    let residual = (0..8)
        .map(|r| {
            let e = (0..8).map(|c| a0[r][c] * b[c]).sum::<f64>() - b0[r];
            e * e
        })
        .sum::<f64>()
        .sqrt();
    (residual < 1e-8).then(|| [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(h: usize, w: usize, cn: usize) -> Vec<f32> {
        (0..h * w * cn)
            .map(|i| (i % 7) as f32 * 0.25 - 0.5)
            .collect()
    }

    #[test]
    fn identity_affine_copies() {
        let data = ramp(9, 13, 3);
        let src = Image {
            data: &data,
            h: 9,
            w: 13,
            cn: 3,
        };
        let mut out = vec![0.0; data.len()];
        let mode = (Interp::Linear, Border::Replicate);
        warp_affine(
            &src,
            mode,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            false,
            13,
            &mut out,
        );
        assert_eq!(out, data);
        warp_perspective(
            &src,
            mode,
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            false,
            (13, 9),
            &mut out,
        );
        assert_eq!(out, data);
    }

    #[test]
    fn remap_half_pixel_averages_and_replicates() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let src = Image {
            data: &data,
            h: 1,
            w: 4,
            cn: 1,
        };
        let mut out = vec![0.0; 3];
        let (mx, my) = ([0.5, -3.0, 9.0], [0.0; 3]);
        remap(
            &src,
            (Interp::Linear, Border::Replicate),
            (&mx, &my),
            3,
            &mut out,
        );
        assert_eq!(out, [0.5, 0.0, 3.0]);
        remap(
            &src,
            (Interp::Linear, Border::Constant),
            (&mx, &my),
            3,
            &mut out,
        );
        assert_eq!(out, [0.5, 0.0, 0.0]);
        remap(
            &src,
            (Interp::Nearest, Border::Replicate),
            (&[1.4, 2.5, 3.5], &my),
            3,
            &mut out,
        );
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn cubic_weights_sum_to_one() {
        for w in cubic_table().iter() {
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert_eq!(cubic_table()[0][5], 1.0);
    }

    #[test]
    fn invert3_matches_identity_product() {
        let m = [2.0, 0.5, 1.0, 0.25, 3.0, -1.0, 0.001, 0.002, 1.0];
        let inv = invert3(m).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let p: f64 = (0..3).map(|k| m[i * 3 + k] * inv[k * 3 + j]).sum();
                assert!((p - (i == j) as i32 as f64).abs() < 1e-12);
            }
        }
        assert!(invert3([0.0; 9]).is_none());
    }
}
