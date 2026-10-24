//! OpenCV 4.13's SIFT detector and float descriptor (`sift.dispatch.cpp`, `sift.simd.hpp`,
//! Apache-2.0) with its default parameters. OpenCV rounds each float descriptor element to an
//! integer in 0..=255, so the descriptors here are bytes of the same values.

use rayon::prelude::*;

use crate::filter::{self, Border};
use crate::resize;

const LAYERS: usize = 3;
const CONTRAST: f32 = 0.04;
const EDGE: f32 = 10.0;
const SIGMA: f32 = 1.6;
const INIT_SIGMA: f32 = 0.5;
const IMG_BORDER: usize = 5;
const MAX_INTERP_STEPS: usize = 5;
const ORI_BINS: usize = 36;
const ORI_SIG_FCTR: f32 = 1.5;
const ORI_RADIUS: f32 = 4.5;
const ORI_PEAK_RATIO: f32 = 0.8;
const DESCR_WIDTH: usize = 4;
const DESCR_BINS: usize = 8;
const DESCR_SCL_FCTR: f32 = 3.0;
const DESCR_MAG_THR: f32 = 0.2;
const INT_DESCR_FCTR: f32 = 512.0;
pub const DESCR_LEN: usize = DESCR_WIDTH * DESCR_WIDTH * DESCR_BINS;
/// The first octave is the input upscaled two times.
const FIRST_OCTAVE: i32 = -1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    /// OpenCV's packed octave: octave in bits 0..8, layer in 8..16, sub-layer in 16..24.
    pub octave: i32,
}

struct Plane {
    data: Vec<f32>,
    h: usize,
    w: usize,
}

impl Plane {
    #[inline(always)]
    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.w..(r + 1) * self.w]
    }

    #[inline(always)]
    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.w + c]
    }
}

#[inline(always)]
fn cv_round(v: f32) -> i32 {
    v.round_ties_even() as i32
}

/// OpenCV's `fastAtan2` in degrees, 0..360.
#[inline(always)]
fn fast_atan2(y: f32, x: f32) -> f32 {
    const R2D: f32 = (180.0 / std::f64::consts::PI) as f32;
    let (p1, p3, p5, p7) = (
        0.999_787_87_f32 * R2D,
        -0.325_808_4_f32 * R2D,
        0.155_578_66_f32 * R2D,
        -0.044_326_555_f32 * R2D,
    );
    let (ax, ay) = (x.abs(), y.abs());
    let poly = |c: f32| {
        let c2 = c * c;
        (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c
    };
    let mut a = if ax >= ay {
        poly(ay / (ax + f64::EPSILON as f32))
    } else {
        90.0 - poly(ax / (ay + f64::EPSILON as f32))
    };
    if x < 0.0 {
        a = 180.0 - a;
    }
    if y < 0.0 {
        a = 360.0 - a;
    }
    a
}

fn blur(src: &Plane, sigma: f64) -> Plane {
    let mut data = vec![0f32; src.data.len()];
    filter::gaussian_blur(
        &src.data,
        &mut data,
        (src.h, src.w, 1),
        (0, 0),
        (sigma, sigma),
        Border::Reflect101,
    )
    .expect("an odd Gaussian kernel");
    Plane {
        data,
        h: src.h,
        w: src.w,
    }
}

/// `createInitialImage` as `cv2.SIFT_create` runs it, with `enable_precise_upscale` off: a
/// bilinear `resize` to twice the size, then the blur that brings the assumed input blur up to
/// SIGMA.
fn initial_image(gray: &[u8], h: usize, w: usize) -> Plane {
    let src: Vec<f32> = gray.iter().map(|&v| v as f32).collect();
    let (dh, dw) = (2 * h, 2 * w);
    let mut dbl = vec![0f32; dh * dw];
    resize::resize_linear(&src, (h, w, 1), (dh, dw), false, &mut dbl).expect("a non-empty image");
    let sig_diff = (SIGMA * SIGMA - INIT_SIGMA * INIT_SIGMA * 4.0)
        .max(0.01)
        .sqrt();
    blur(
        &Plane {
            data: dbl,
            h: dh,
            w: dw,
        },
        sig_diff as f64,
    )
}

fn gaussian_pyramid(base: Plane, octaves: usize) -> Vec<Plane> {
    let k = 2f64.powf(1.0 / LAYERS as f64);
    let sig: Vec<f64> = (0..LAYERS + 3)
        .map(|i| {
            if i == 0 {
                return SIGMA as f64;
            }
            let prev = k.powi(i as i32 - 1) * SIGMA as f64;
            let total = prev * k;
            (total * total - prev * prev).sqrt()
        })
        .collect();
    let mut pyr: Vec<Plane> = Vec::with_capacity(octaves * (LAYERS + 3));
    pyr.push(base);
    for o in 0..octaves {
        for (i, &s) in sig.iter().enumerate() {
            if o == 0 && i == 0 {
                continue;
            }
            let next = if i == 0 {
                let src = &pyr[(o - 1) * (LAYERS + 3) + LAYERS];
                let (dh, dw) = (src.h / 2, src.w / 2);
                let mut data = vec![0f32; dh * dw];
                resize::resize_nearest(&src.data, (src.h, src.w, 1), (dh, dw), &mut data);
                Plane { data, h: dh, w: dw }
            } else {
                blur(&pyr[o * (LAYERS + 3) + i - 1], s)
            };
            pyr.push(next);
        }
    }
    pyr
}

fn dog_pyramid(gpyr: &[Plane], octaves: usize) -> Vec<Plane> {
    (0..octaves * (LAYERS + 2))
        .map(|a| {
            let (o, i) = (a / (LAYERS + 2), a % (LAYERS + 2));
            let (lo, hi) = (&gpyr[o * (LAYERS + 3) + i], &gpyr[o * (LAYERS + 3) + i + 1]);
            let mut data = hi.data.clone();
            crate::par_zip(&mut data, &lo.data, move |d, &s| *d -= s);
            Plane {
                data,
                h: lo.h,
                w: lo.w,
            }
        })
        .collect()
}

/// OpenCV's 3x3 `Matx::solve` (DECOMP_LU): Cramer's rule; zeros for a singular matrix.
fn solve3(a: [[f32; 3]; 3], b: [f32; 3]) -> [f32; 3] {
    let d = a[0][0] as f64 * (a[1][1] as f64 * a[2][2] as f64 - a[1][2] as f64 * a[2][1] as f64)
        - a[0][1] as f64 * (a[1][0] as f64 * a[2][2] as f64 - a[1][2] as f64 * a[2][0] as f64)
        + a[0][2] as f64 * (a[1][0] as f64 * a[2][1] as f64 - a[1][1] as f64 * a[2][0] as f64);
    let d = d as f32;
    if d == 0.0 {
        return [0.0; 3];
    }
    let d = 1.0 / d;
    [
        d * (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
            + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2])),
        d * (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
            - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0])),
        d * (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
            - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
            + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])),
    ]
}

/// `adjustLocalExtrema`: the sub-pixel, sub-layer extremum and its contrast and edge tests.
fn adjust_local_extrema(
    dog: &[Plane],
    octv: usize,
    layer: &mut usize,
    r: &mut usize,
    c: &mut usize,
) -> Option<Keypoint> {
    const IMG_SCALE: f32 = 1.0 / 255.0;
    const DERIV: f32 = IMG_SCALE * 0.5;
    const SECOND: f32 = IMG_SCALE;
    const CROSS: f32 = IMG_SCALE * 0.25;
    let (mut xi, mut xr, mut xc) = (0f32, 0f32, 0f32);
    let mut step = 0;
    while step < MAX_INTERP_STEPS {
        let idx = octv * (LAYERS + 2) + *layer;
        let (img, prev, next) = (&dog[idx], &dog[idx - 1], &dog[idx + 1]);
        let (r0, c0) = (*r, *c);
        let d_d = [
            (img.at(r0, c0 + 1) - img.at(r0, c0 - 1)) * DERIV,
            (img.at(r0 + 1, c0) - img.at(r0 - 1, c0)) * DERIV,
            (next.at(r0, c0) - prev.at(r0, c0)) * DERIV,
        ];
        let v2 = img.at(r0, c0) * 2.0;
        let dxx = (img.at(r0, c0 + 1) + img.at(r0, c0 - 1) - v2) * SECOND;
        let dyy = (img.at(r0 + 1, c0) + img.at(r0 - 1, c0) - v2) * SECOND;
        let dss = (next.at(r0, c0) + prev.at(r0, c0) - v2) * SECOND;
        let dxy = (img.at(r0 + 1, c0 + 1) - img.at(r0 + 1, c0 - 1) - img.at(r0 - 1, c0 + 1)
            + img.at(r0 - 1, c0 - 1))
            * CROSS;
        let dxs = (next.at(r0, c0 + 1) - next.at(r0, c0 - 1) - prev.at(r0, c0 + 1)
            + prev.at(r0, c0 - 1))
            * CROSS;
        let dys = (next.at(r0 + 1, c0) - next.at(r0 - 1, c0) - prev.at(r0 + 1, c0)
            + prev.at(r0 - 1, c0))
            * CROSS;
        let x = solve3([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]], d_d);
        xi = -x[2];
        xr = -x[1];
        xc = -x[0];
        if xi.abs() < 0.5 && xr.abs() < 0.5 && xc.abs() < 0.5 {
            break;
        }
        let far = (i32::MAX / 3) as f32;
        if xi.abs() > far || xr.abs() > far || xc.abs() > far {
            return None;
        }
        let nc = c0 as i64 + cv_round(xc) as i64;
        let nr = r0 as i64 + cv_round(xr) as i64;
        let nl = *layer as i64 + cv_round(xi) as i64;
        if nl < 1
            || nl > LAYERS as i64
            || nc < IMG_BORDER as i64
            || nc >= (img.w - IMG_BORDER) as i64
            || nr < IMG_BORDER as i64
            || nr >= (img.h - IMG_BORDER) as i64
        {
            return None;
        }
        (*c, *r, *layer) = (nc as usize, nr as usize, nl as usize);
        step += 1;
    }
    if step >= MAX_INTERP_STEPS {
        return None;
    }
    let idx = octv * (LAYERS + 2) + *layer;
    let (img, prev, next) = (&dog[idx], &dog[idx - 1], &dog[idx + 1]);
    let (r0, c0) = (*r, *c);
    let d_d = [
        (img.at(r0, c0 + 1) - img.at(r0, c0 - 1)) * DERIV,
        (img.at(r0 + 1, c0) - img.at(r0 - 1, c0)) * DERIV,
        (next.at(r0, c0) - prev.at(r0, c0)) * DERIV,
    ];
    let t = d_d[0] * xc + d_d[1] * xr + d_d[2] * xi;
    let contr = img.at(r0, c0) * IMG_SCALE + t * 0.5;
    if contr.abs() * (LAYERS as f32) < CONTRAST {
        return None;
    }
    let v2 = img.at(r0, c0) * 2.0;
    let dxx = (img.at(r0, c0 + 1) + img.at(r0, c0 - 1) - v2) * SECOND;
    let dyy = (img.at(r0 + 1, c0) + img.at(r0 - 1, c0) - v2) * SECOND;
    let dxy = (img.at(r0 + 1, c0 + 1) - img.at(r0 + 1, c0 - 1) - img.at(r0 - 1, c0 + 1)
        + img.at(r0 - 1, c0 - 1))
        * CROSS;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    if det <= 0.0 || tr * tr * EDGE >= (EDGE + 1.0) * (EDGE + 1.0) * det {
        return None;
    }
    let scale = (1u32 << octv) as f32;
    Some(Keypoint {
        x: (c0 as f32 + xc) * scale,
        y: (r0 as f32 + xr) * scale,
        size: SIGMA * libm::powf(2.0, (*layer as f32 + xi) / LAYERS as f32) * scale * 2.0,
        angle: 0.0,
        response: contr.abs(),
        octave: octv as i32
            + ((*layer as i32) << 8)
            + ((((xi as f64 + 0.5) * 255.0).round_ties_even() as i32) << 16),
    })
}

/// `calcOrientationHist`: the smoothed gradient orientation histogram around (r, c); returns its
/// maximum.
fn orientation_hist(
    img: &Plane,
    (r, c): (usize, usize),
    radius: i32,
    sigma: f32,
    hist: &mut [f32; ORI_BINS],
) -> f32 {
    let n = ORI_BINS;
    let expf_scale = -1.0 / (2.0 * sigma * sigma);
    let mut temp = [0f32; ORI_BINS + 4];
    for i in -radius..=radius {
        let y = r as i32 + i;
        if y <= 0 || y >= img.h as i32 - 1 {
            continue;
        }
        for j in -radius..=radius {
            let x = c as i32 + j;
            if x <= 0 || x >= img.w as i32 - 1 {
                continue;
            }
            let (y, x) = (y as usize, x as usize);
            let dx = img.at(y, x + 1) - img.at(y, x - 1);
            let dy = img.at(y - 1, x) - img.at(y + 1, x);
            let w = libm::expf((i * i + j * j) as f32 * expf_scale);
            let ori = fast_atan2(dy, dx);
            let mag = (dx * dx + dy * dy).sqrt();
            let mut bin = cv_round((n as f32 / 360.0) * ori);
            if bin >= n as i32 {
                bin -= n as i32;
            }
            if bin < 0 {
                bin += n as i32;
            }
            temp[bin as usize + 2] += w * mag;
        }
    }
    temp[1] = temp[n + 1];
    temp[0] = temp[n];
    temp[n + 2] = temp[2];
    temp[n + 3] = temp[3];
    let mut maxval = f32::MIN;
    for (i, h) in hist.iter_mut().enumerate() {
        let t = |k: usize| temp[i + k];
        *h = (t(0) + t(4)) * (1.0 / 16.0) + (t(1) + t(3)) * (4.0 / 16.0) + t(2) * (6.0 / 16.0);
        maxval = maxval.max(*h);
    }
    maxval
}

/// The keypoints of one DoG row: local extrema in the 3x3x3 neighborhood, refined, then one
/// keypoint per dominant orientation.
fn row_extrema(gpyr: &[Plane], dog: &[Plane], o: usize, i: usize, r: usize) -> Vec<Keypoint> {
    let threshold = (0.5 * CONTRAST as f64 / LAYERS as f64 * 255.0).floor() as f32;
    let idx = o * (LAYERS + 2) + i;
    let (img, prev, next) = (&dog[idx], &dog[idx - 1], &dog[idx + 1]);
    let w = img.w;
    let (cur, up, down) = (img.row(r), img.row(r - 1), img.row(r + 1));
    let nb = |p: &Plane, c: usize| {
        let (a, b, d) = (p.row(r - 1), p.row(r), p.row(r + 1));
        [
            a[c - 1],
            a[c],
            a[c + 1],
            b[c - 1],
            b[c + 1],
            d[c - 1],
            d[c],
            d[c + 1],
        ]
    };
    let mut out = Vec::new();
    for c in IMG_BORDER..w - IMG_BORDER {
        let val = cur[c];
        if val.abs() <= threshold {
            continue;
        }
        let ring = [
            up[c - 1],
            up[c],
            up[c + 1],
            cur[c - 1],
            cur[c + 1],
            down[c - 1],
            down[c],
            down[c + 1],
        ];
        let is_ext = if val > 0.0 {
            let mx = |v: [f32; 8]| v.iter().fold(f32::MIN, |m, &x| m.max(x));
            val >= mx(ring)
                && val >= mx(nb(prev, c))
                && val >= mx(nb(next, c))
                && val >= prev.at(r, c).max(next.at(r, c))
        } else {
            let mn = |v: [f32; 8]| v.iter().fold(f32::MAX, |m, &x| m.min(x));
            val <= mn(ring)
                && val <= mn(nb(prev, c))
                && val <= mn(nb(next, c))
                && val <= prev.at(r, c).min(next.at(r, c))
        };
        if !is_ext {
            continue;
        }
        let (mut r1, mut c1, mut layer) = (r, c, i);
        let Some(kpt) = adjust_local_extrema(dog, o, &mut layer, &mut r1, &mut c1) else {
            continue;
        };
        let scl_octv = kpt.size * 0.5 / (1u32 << o) as f32;
        let mut hist = [0f32; ORI_BINS];
        let omax = orientation_hist(
            &gpyr[o * (LAYERS + 3) + layer],
            (r1, c1),
            cv_round(ORI_RADIUS * scl_octv),
            ORI_SIG_FCTR * scl_octv,
            &mut hist,
        );
        let mag_thr = omax * ORI_PEAK_RATIO;
        let n = ORI_BINS;
        for j in 0..n {
            let l = if j > 0 { j - 1 } else { n - 1 };
            let r2 = if j < n - 1 { j + 1 } else { 0 };
            if hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr {
                let mut bin =
                    j as f32 + 0.5 * (hist[l] - hist[r2]) / (hist[l] - 2.0 * hist[j] + hist[r2]);
                bin = if bin < 0.0 {
                    n as f32 + bin
                } else if bin >= n as f32 {
                    bin - n as f32
                } else {
                    bin
                };
                let mut angle = 360.0 - (360.0 / n as f32) * bin;
                if (angle - 360.0).abs() < f32::EPSILON {
                    angle = 0.0;
                }
                out.push(Keypoint { angle, ..kpt });
            }
        }
    }
    out
}

/// `calcSIFTDescriptor`: the 4x4x8 trilinear gradient histogram, clipped at 0.2 of its norm and
/// scaled to bytes.
fn descriptor(img: &Plane, (px, py): (f32, f32), ori: f32, scl: f32) -> [u8; DESCR_LEN] {
    let (d, n) = (DESCR_WIDTH as i32, DESCR_BINS as i32);
    let (pxi, pyi) = (cv_round(px), cv_round(py));
    let rad = ori * (std::f64::consts::PI / 180.0) as f32;
    let bins_per_rad = n as f32 / 360.0;
    let exp_scale = -1.0 / (d as f32 * d as f32 * 0.5);
    let hist_width = DESCR_SCL_FCTR * scl;
    let radius = cv_round(hist_width * std::f32::consts::SQRT_2 * (d + 1) as f32 * 0.5);
    let diag = ((img.w as f64).powi(2) + (img.h as f64).powi(2)).sqrt() as i32;
    let radius = radius.min(diag);
    let cos_t = libm::cosf(rad) / hist_width;
    let sin_t = libm::sinf(rad) / hist_width;
    let (dp, np) = ((d + 2) as usize, (n + 2) as usize);
    let mut hist = vec![0f32; dp * dp * np];
    let (rows, cols) = (img.h as i32, img.w as i32);
    for i in -radius..=radius {
        for j in -radius..=radius {
            let c_rot = j as f32 * cos_t - i as f32 * sin_t;
            let r_rot = j as f32 * sin_t + i as f32 * cos_t;
            let rbin = r_rot + (d / 2) as f32 - 0.5;
            let cbin = c_rot + (d / 2) as f32 - 0.5;
            let (r, c) = (pyi + i, pxi + j);
            if !(rbin > -1.0
                && rbin < d as f32
                && cbin > -1.0
                && cbin < d as f32
                && r > 0
                && r < rows - 1
                && c > 0
                && c < cols - 1)
            {
                continue;
            }
            let (ru, cu) = (r as usize, c as usize);
            let dx = img.at(ru, cu + 1) - img.at(ru, cu - 1);
            let dy = img.at(ru - 1, cu) - img.at(ru + 1, cu);
            let w = libm::expf((c_rot * c_rot + r_rot * r_rot) * exp_scale);
            let o = fast_atan2(dy, dx);
            let mag = (dx * dx + dy * dy).sqrt() * w;
            let obin = (o - ori) * bins_per_rad;
            let (r0, c0, mut o0) = (
                rbin.floor() as i32,
                cbin.floor() as i32,
                obin.floor() as i32,
            );
            let (rbin, cbin, obin) = (rbin - r0 as f32, cbin - c0 as f32, obin - o0 as f32);
            if o0 < 0 {
                o0 += n;
            }
            if o0 >= n {
                o0 -= n;
            }
            let v_r1 = mag * rbin;
            let v_r0 = mag - v_r1;
            let v_rc11 = v_r1 * cbin;
            let v_rc10 = v_r1 - v_rc11;
            let v_rc01 = v_r0 * cbin;
            let v_rc00 = v_r0 - v_rc01;
            let v_rco111 = v_rc11 * obin;
            let v_rco110 = v_rc11 - v_rco111;
            let v_rco101 = v_rc10 * obin;
            let v_rco100 = v_rc10 - v_rco101;
            let v_rco011 = v_rc01 * obin;
            let v_rco010 = v_rc01 - v_rco011;
            let v_rco001 = v_rc00 * obin;
            let v_rco000 = v_rc00 - v_rco001;
            let idx = (((r0 + 1) as usize * dp) + (c0 + 1) as usize) * np + o0 as usize;
            hist[idx] += v_rco000;
            hist[idx + 1] += v_rco001;
            hist[idx + np] += v_rco010;
            hist[idx + np + 1] += v_rco011;
            hist[idx + dp * np] += v_rco100;
            hist[idx + dp * np + 1] += v_rco101;
            hist[idx + (dp + 1) * np] += v_rco110;
            hist[idx + (dp + 1) * np + 1] += v_rco111;
        }
    }
    let (du, nu) = (d as usize, n as usize);
    let mut raw = [0f32; DESCR_LEN];
    for i in 0..du {
        for j in 0..du {
            let idx = ((i + 1) * dp + (j + 1)) * np;
            hist[idx] += hist[idx + nu];
            hist[idx + 1] += hist[idx + nu + 1];
            raw[(i * du + j) * nu..(i * du + j + 1) * nu].copy_from_slice(&hist[idx..idx + nu]);
        }
    }
    let nrm2: f32 = raw.iter().map(|v| v * v).sum();
    let thr = nrm2.sqrt() * DESCR_MAG_THR;
    let mut nrm2 = 0f32;
    for v in raw.iter_mut() {
        *v = v.min(thr);
        nrm2 += *v * *v;
    }
    let nrm2 = INT_DESCR_FCTR / nrm2.sqrt().max(f32::EPSILON);
    raw.map(|v| (v * nrm2).round_ties_even().clamp(0.0, 255.0) as u8)
}

/// `KeypointGreater`, the order `removeDuplicatedSorted` sorts by.
fn keypoint_order(a: &Keypoint, b: &Keypoint) -> std::cmp::Ordering {
    a.x.total_cmp(&b.x)
        .then(a.y.total_cmp(&b.y))
        .then(b.size.total_cmp(&a.size))
        .then(a.angle.total_cmp(&b.angle))
        .then(b.response.total_cmp(&a.response))
        .then(b.octave.cmp(&a.octave))
}

/// `SIFT_create(nfeatures).detectAndCompute(gray, None)` on an 8-bit image: the keypoints and
/// their descriptors, row for row. `nfeatures` 0 keeps every keypoint.
pub fn detect_and_compute(
    gray: &[u8],
    h: usize,
    w: usize,
    nfeatures: usize,
) -> (Vec<Keypoint>, Vec<[u8; DESCR_LEN]>) {
    if h == 0 || w == 0 || gray.len() != h * w {
        return (Vec::new(), Vec::new());
    }
    let base = initial_image(gray, h, w);
    let octaves = ((base.w.min(base.h) as f64).ln() / 2f64.ln() - 2.0).round_ties_even() as i32
        - FIRST_OCTAVE;
    if octaves < 1 {
        return (Vec::new(), Vec::new());
    }
    let octaves = octaves as usize;
    let gpyr = gaussian_pyramid(base, octaves);
    let dog = dog_pyramid(&gpyr, octaves);
    let mut kpts: Vec<Keypoint> = Vec::new();
    for o in 0..octaves {
        for i in 1..=LAYERS {
            let rows = dog[o * (LAYERS + 2) + i].h;
            if rows <= 2 * IMG_BORDER || dog[o * (LAYERS + 2) + i].w <= 2 * IMG_BORDER {
                continue;
            }
            let found: Vec<Keypoint> = (IMG_BORDER..rows - IMG_BORDER)
                .into_par_iter()
                .flat_map_iter(|r| row_extrema(&gpyr, &dog, o, i, r))
                .collect();
            kpts.extend(found);
        }
    }
    kpts.sort_by(keypoint_order);
    kpts.dedup_by(|b, a| a.x == b.x && a.y == b.y && a.size == b.size && a.angle == b.angle);
    if nfeatures > 0 && kpts.len() > nfeatures {
        // retainBest: the nfeatures strongest, and every keypoint tied with the last of them.
        let mut by_response = kpts.clone();
        by_response.sort_by(|a, b| b.response.total_cmp(&a.response));
        let cut = by_response[nfeatures - 1].response;
        kpts.retain(|k| k.response >= cut);
    }
    let scale = 1.0 / (1u32 << -FIRST_OCTAVE) as f32;
    for k in kpts.iter_mut() {
        k.octave = (k.octave & !255) | ((k.octave + FIRST_OCTAVE) & 255);
        k.x *= scale;
        k.y *= scale;
        k.size *= scale;
    }
    let desc = kpts
        .par_iter()
        .map(|k| {
            let mut octave = k.octave & 255;
            let layer = ((k.octave >> 8) & 255) as usize;
            if octave >= 128 {
                octave |= -128;
            }
            let s = if octave >= 0 {
                1.0 / (1u32 << octave) as f32
            } else {
                (1u32 << -octave) as f32
            };
            let img = &gpyr[(octave - FIRST_OCTAVE) as usize * (LAYERS + 3) + layer];
            let mut angle = 360.0 - k.angle;
            if (angle - 360.0).abs() < f32::EPSILON {
                angle = 0.0;
            }
            descriptor(img, (k.x * s, k.y * s), angle, k.size * s * 0.5)
        })
        .collect();
    (kpts, desc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs(h: usize, w: usize) -> Vec<u8> {
        (0..h * w)
            .map(|p| {
                let (y, x) = ((p / w) as f32, (p % w) as f32);
                let v = (x * 0.21).sin() * (y * 0.13).cos() + (x * 0.05 + y * 0.07).sin();
                (127.0 + 60.0 * v) as u8
            })
            .collect()
    }

    #[test]
    fn fast_atan2_matches_atan2() {
        for k in 0..360 {
            let t = (k as f32 + 0.5).to_radians();
            let (y, x) = (t.sin(), t.cos());
            let want = y.atan2(x).to_degrees().rem_euclid(360.0);
            assert!((fast_atan2(y, x) - want).abs() < 0.02, "{k}");
        }
    }

    #[test]
    fn detects_keypoints_with_unit_norm_descriptors() {
        let (h, w) = (160, 200);
        let (kp, desc) = detect_and_compute(&blobs(h, w), h, w, 0);
        assert!(kp.len() > 20);
        assert_eq!(kp.len(), desc.len());
        for k in &kp {
            assert!(k.x >= 0.0 && k.x < w as f32 && k.y >= 0.0 && k.y < h as f32);
        }
        for d in &desc {
            let n: f32 = d.iter().map(|&v| (v as f32).powi(2)).sum::<f32>().sqrt();
            assert!((n - 512.0).abs() < 8.0, "{n}");
        }
    }

    #[test]
    fn nfeatures_keeps_the_strongest() {
        let (h, w) = (160, 200);
        let (all, _) = detect_and_compute(&blobs(h, w), h, w, 0);
        let (best, desc) = detect_and_compute(&blobs(h, w), h, w, 10);
        assert!(best.len() >= 10 && best.len() < all.len());
        assert_eq!(desc.len(), best.len());
    }
}
