//! `cv2.createLineSegmentDetector().detect(u8)` with its defaults (LSD_REFINE_STD, scale 0.8),
//! ported from OpenCV 4.13's `modules/imgproc/src/lsd.cpp` (Apache-2.0). The Gaussian and the
//! INTER_LINEAR_EXACT downscale copy OpenCV's fixed-point 8-bit code.

use rayon::prelude::*;

use crate::filter::{self, Border};

const SCALE: f64 = 0.8;
const SIGMA_SCALE: f64 = 0.6;
const QUANT: f64 = 2.0;
const ANG_TH: f64 = 22.5;
const DENSITY_TH: f64 = 0.7;
const N_BINS: usize = 1024;
const NOTDEF: f64 = -1024.0;
const DEG_TO_RADS: f64 = std::f64::consts::PI / 180.0;

/// OpenCV's `fastAtan2`, in degrees on [0, 360): a polynomial, about 0.3 degrees off.
fn fast_atan2(y: f32, x: f32) -> f32 {
    const K: f32 = (180.0 / std::f64::consts::PI) as f32;
    let (p1, p3, p5, p7) = (
        0.9997878412794807_f64 as f32 * K,
        -0.3258083974640975_f64 as f32 * K,
        0.1555786518463281_f64 as f32 * K,
        -0.04432655554792128_f64 as f32 * K,
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

/// `getGaussianKernelFixedPoint_ED` with 8 fraction bits: the kernel rounded with error
/// diffusion, its centre taking what makes the sum exactly 256.
fn fixed_kernel(n: usize, sigma: f64) -> Vec<u32> {
    let k = filter::gaussian_kernel_f64(n, sigma);
    let mut out = vec![0u32; n];
    let (mut err, mut sum) = (0f64, 0i64);
    for i in 0..n / 2 {
        let adj = k[i] * 256.0 + err;
        let v = adj.round_ties_even() as i64;
        err = adj - v as f64;
        out[i] = v as u32;
        out[n - 1 - i] = v as u32;
        sum += v;
    }
    out[n / 2] = (256 - 2 * sum) as u32;
    out
}

/// `cv2.GaussianBlur(u8, (n, n), sigma)` (reflect-101): exact integer rows in 8.8, columns in
/// 16.16, rounded half up.
fn gaussian_blur_u8(src: &[u8], (h, w): (usize, usize), n: usize, sigma: f64) -> Vec<u8> {
    let k = fixed_kernel(n, sigma);
    let r = (n / 2) as isize;
    let idx =
        |p: isize, len: usize| filter::border_interpolate(p, len, Border::Reflect101).unwrap_or(0);
    let xs: Vec<Vec<usize>> = (0..w as isize)
        .map(|x| (-r..=r).map(|d| idx(x + d, w)).collect())
        .collect();
    let mut rows = vec![0u32; h * w];
    rows.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let line = &src[y * w..(y + 1) * w];
        for (o, taps) in out.iter_mut().zip(&xs) {
            *o = taps.iter().zip(&k).map(|(&t, &c)| line[t] as u32 * c).sum();
        }
    });
    let mut dst = vec![0u8; h * w];
    dst.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let taps: Vec<&[u32]> = (-r..=r)
            .map(|d| &rows[idx(y as isize + d, h) * w..][..w])
            .collect();
        for (x, o) in out.iter_mut().enumerate() {
            let s: u32 = taps.iter().zip(&k).map(|(t, &c)| t[x] * c).sum();
            *o = ((s + (1 << 15)) >> 16).min(255) as u8;
        }
    });
    dst
}

/// `interpolationLinear::getCoeffs` for every destination index: (offset, c0, c1) in 8.8, with
/// c0 = 256 and c1 = 0 past either end.
fn linear_exact_axis(src: usize, dst: usize, inv_scale: f64) -> Vec<(usize, u32, u32)> {
    let scale = 1.0 / inv_scale;
    (0..dst)
        .map(|d| {
            let f = scale * (d as f64 + 0.5) - 0.5;
            let i = f.floor() as i64;
            if i < 0 || src <= 1 {
                (0, 256, 0)
            } else if (i as usize) < src - 1 {
                let c1 = ((f - i as f64) * 256.0).round_ties_even() as u32;
                (i as usize, 256 - c1, c1)
            } else {
                (src - 1, 256, 0)
            }
        })
        .collect()
}

/// `cv2.resize(u8, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR_EXACT)`.
fn resize_exact_u8(src: &[u8], (h, w): (usize, usize)) -> (Vec<u8>, usize, usize) {
    let dw = (w as f64 * SCALE).round_ties_even() as usize;
    let dh = (h as f64 * SCALE).round_ties_even() as usize;
    let xs = linear_exact_axis(w, dw, SCALE);
    let ys = linear_exact_axis(h, dh, SCALE);
    let hline = |sy: usize, out: &mut [u32]| {
        let row = &src[sy * w..(sy + 1) * w];
        for (o, &(i, c0, c1)) in out.iter_mut().zip(&xs) {
            let b = if c1 == 0 { 0 } else { row[i + 1] as u32 * c1 };
            *o = row[i] as u32 * c0 + b;
        }
    };
    let mut dst = vec![0u8; dh * dw];
    dst.par_chunks_mut(dw)
        .zip(&ys)
        .for_each(|(out, &(i, c0, c1))| {
            let (mut a, mut b) = (vec![0u32; dw], vec![0u32; dw]);
            hline(i, &mut a);
            if c1 != 0 {
                hline(i + 1, &mut b);
            }
            for ((o, &a), &b) in out.iter_mut().zip(&a).zip(&b) {
                *o = ((a * c0 + b * c1 + (1 << 15)) >> 16).min(255) as u8;
            }
        });
    (dst, dh, dw)
}

struct Point {
    x: usize,
    y: usize,
    angle: f64,
    modgrad: f64,
}

struct Rect {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    width: f64,
}

struct Detector {
    w: usize,
    h: usize,
    angles: Vec<f64>,
    modgrad: Vec<f64>,
    /// `(cosf, sinf)` of each defined angle, as the region sums take them.
    unit: Vec<(f32, f32)>,
    /// Taken by a region, or with no defined angle: no region can take it.
    used: Vec<bool>,
}

fn dist(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)).sqrt()
}

fn angle_diff_signed(a: f64, b: f64) -> f64 {
    let mut d = a - b;
    while d <= -std::f64::consts::PI {
        d += 2.0 * std::f64::consts::PI;
    }
    while d > std::f64::consts::PI {
        d -= 2.0 * std::f64::consts::PI;
    }
    d
}

impl Detector {
    /// `ll_angle`: gradient angle and norm per pixel, and the pixels pseudo-ordered by norm.
    fn new(img: &[u8], h: usize, w: usize, threshold: f64) -> (Self, Vec<u32>) {
        let mut angles = vec![NOTDEF; h * w];
        let mut modgrad = vec![0f64; h * w];
        angles
            .par_chunks_mut(w)
            .zip(modgrad.par_chunks_mut(w))
            .enumerate()
            .take(h.saturating_sub(1))
            .for_each(|(y, (ang, md))| {
                let (r0, r1) = (&img[y * w..(y + 1) * w], &img[(y + 1) * w..(y + 2) * w]);
                for x in 0..w - 1 {
                    let da = r1[x + 1] as i32 - r0[x] as i32;
                    let bc = r0[x + 1] as i32 - r1[x] as i32;
                    let (gx, gy) = (da + bc, da - bc);
                    let norm = ((gx * gx + gy * gy) as f64 / 4.0).sqrt();
                    md[x] = norm;
                    if norm > threshold {
                        ang[x] = fast_atan2(gx as f32, -gy as f32) as f64 * DEG_TO_RADS;
                    }
                }
            });
        let max_grad = angles
            .iter()
            .zip(&modgrad)
            .filter(|(a, _)| **a != NOTDEF)
            .map(|(_, m)| *m)
            .fold(-1f64, f64::max);
        let coef = if max_grad > 0.0 {
            (N_BINS - 1) as f64 / max_grad
        } else {
            0.0
        };
        // A stable sort by descending bin is a bucket sort that keeps raster order in a bin.
        let bins: Vec<u16> = (0..h.saturating_sub(1) * w)
            .map(|i| {
                if i % w == w - 1 {
                    u16::MAX
                } else {
                    (modgrad[i] * coef) as u16
                }
            })
            .collect();
        let mut start = vec![0usize; N_BINS + 1];
        for &b in bins.iter().filter(|&&b| b != u16::MAX) {
            start[N_BINS - 1 - b as usize + 1] += 1;
        }
        for i in 0..N_BINS {
            start[i + 1] += start[i];
        }
        let mut order = vec![0u32; start[N_BINS]];
        for (i, &b) in bins.iter().enumerate().filter(|&(_, &b)| b != u16::MAX) {
            let s = &mut start[N_BINS - 1 - b as usize];
            order[*s] = i as u32;
            *s += 1;
        }
        let mut unit = vec![(0f32, 0f32); h * w];
        let mut used = vec![false; h * w];
        unit.par_chunks_mut(w)
            .zip(used.par_chunks_mut(w))
            .zip(angles.par_chunks(w))
            .for_each(|((u, d), a)| {
                for ((u, d), &a) in u.iter_mut().zip(d).zip(a) {
                    if a == NOTDEF {
                        *d = true;
                    } else {
                        *u = (libm::cosf(a as f32), libm::sinf(a as f32));
                    }
                }
            });
        let det = Detector {
            w,
            h,
            angles,
            modgrad,
            unit,
            used,
        };
        (det, order)
    }

    fn is_aligned(&self, x: usize, y: usize, theta: f64, prec: f64) -> bool {
        let a = self.angles[y * self.w + x];
        if a == NOTDEF {
            return false;
        }
        let mut n = (theta - a).abs();
        if n > 1.5 * std::f64::consts::PI {
            n = (n - 2.0 * std::f64::consts::PI).abs();
        }
        n <= prec
    }

    /// Returns the region's mean angle.
    fn region_grow(&mut self, sx: usize, sy: usize, reg: &mut Vec<Point>, prec: f64) -> f64 {
        reg.clear();
        let i = sy * self.w + sx;
        let mut reg_angle = self.angles[i];
        reg.push(Point {
            x: sx,
            y: sy,
            angle: reg_angle,
            modgrad: self.modgrad[i],
        });
        let mut sumdx = reg_angle.cos() as f32;
        let mut sumdy = reg_angle.sin() as f32;
        self.used[i] = true;
        let mut k = 0;
        while k < reg.len() {
            let (px, py) = (reg[k].x, reg[k].y);
            for yy in py.saturating_sub(1)..=(py + 1).min(self.h - 1) {
                for xx in px.saturating_sub(1)..=(px + 1).min(self.w - 1) {
                    let j = yy * self.w + xx;
                    if !self.used[j] && self.is_aligned(xx, yy, reg_angle, prec) {
                        let angle = self.angles[j];
                        self.used[j] = true;
                        reg.push(Point {
                            x: xx,
                            y: yy,
                            angle,
                            modgrad: self.modgrad[j],
                        });
                        let (c, s) = self.unit[j];
                        sumdx += c;
                        sumdy += s;
                        reg_angle = fast_atan2(sumdy, sumdx) as f64 * DEG_TO_RADS;
                    }
                }
            }
            k += 1;
        }
        reg_angle
    }

    fn region2rect(reg: &[Point], reg_angle: f64, prec: f64) -> Rect {
        let (mut x, mut y, mut sum) = (0.0, 0.0, 0.0);
        for p in reg {
            x += p.x as f64 * p.modgrad;
            y += p.y as f64 * p.modgrad;
            sum += p.modgrad;
        }
        x /= sum;
        y /= sum;
        let theta = Self::get_theta(reg, x, y, reg_angle, prec);
        let (dx, dy) = (theta.cos(), theta.sin());
        let (mut l_min, mut l_max, mut w_min, mut w_max) = (0f64, 0f64, 0f64, 0f64);
        for p in reg {
            let (rx, ry) = (p.x as f64 - x, p.y as f64 - y);
            let l = rx * dx + ry * dy;
            let w = -rx * dy + ry * dx;
            if l > l_max {
                l_max = l;
            } else if l < l_min {
                l_min = l;
            }
            if w > w_max {
                w_max = w;
            } else if w < w_min {
                w_min = w;
            }
        }
        Rect {
            x1: x + l_min * dx,
            y1: y + l_min * dy,
            x2: x + l_max * dx,
            y2: y + l_max * dy,
            width: (w_max - w_min).max(1.0),
        }
    }

    fn get_theta(reg: &[Point], x: f64, y: f64, reg_angle: f64, prec: f64) -> f64 {
        let (mut ixx, mut iyy, mut ixy) = (0.0, 0.0, 0.0);
        for p in reg {
            let (dx, dy) = (p.x as f64 - x, p.y as f64 - y);
            ixx += dy * dy * p.modgrad;
            iyy += dx * dx * p.modgrad;
            ixy -= dx * dy * p.modgrad;
        }
        let lambda = 0.5 * (ixx + iyy - ((ixx - iyy) * (ixx - iyy) + 4.0 * ixy * ixy).sqrt());
        let deg = if ixx.abs() > iyy.abs() {
            fast_atan2((lambda - ixx) as f32, ixy as f32)
        } else {
            fast_atan2(ixy as f32, (lambda - iyy) as f32)
        };
        let theta = deg as f64 * DEG_TO_RADS;
        if angle_diff_signed(theta, reg_angle).abs() > prec {
            theta + std::f64::consts::PI
        } else {
            theta
        }
    }

    fn density(reg: &[Point], r: &Rect) -> f64 {
        reg.len() as f64 / (dist(r.x1, r.y1, r.x2, r.y2) * r.width)
    }

    fn release(&mut self, p: &Point) {
        self.used[p.y * self.w + p.x] = false;
    }

    fn refine(&mut self, reg: &mut Vec<Point>, prec: f64, rec: &mut Rect) -> bool {
        if Self::density(reg, rec) >= DENSITY_TH {
            return true;
        }
        let (xc, yc, ang_c) = (reg[0].x as f64, reg[0].y as f64, reg[0].angle);
        let (seed_x, seed_y) = (reg[0].x, reg[0].y);
        let (mut sum, mut s_sum, mut n) = (0.0, 0.0, 0usize);
        for p in reg.iter() {
            self.release(p);
            if dist(xc, yc, p.x as f64, p.y as f64) < rec.width {
                let d = angle_diff_signed(p.angle, ang_c);
                sum += d;
                s_sum += d * d;
                n += 1;
            }
        }
        let mean = sum / n as f64;
        let tau = 2.0 * ((s_sum - 2.0 * mean * sum) / n as f64 + mean * mean).sqrt();
        let reg_angle = self.region_grow(seed_x, seed_y, reg, tau);
        if reg.len() < 2 {
            return false;
        }
        *rec = Self::region2rect(reg, reg_angle, prec);
        let density = Self::density(reg, rec);
        if density >= DENSITY_TH {
            return true;
        }
        self.reduce_region_radius(reg, reg_angle, prec, rec, density)
    }

    fn reduce_region_radius(
        &mut self,
        reg: &mut Vec<Point>,
        reg_angle: f64,
        prec: f64,
        rec: &mut Rect,
        mut density: f64,
    ) -> bool {
        let (xc, yc) = (reg[0].x as f64, reg[0].y as f64);
        let d2 = |x: f64, y: f64| (x - xc) * (x - xc) + (y - yc) * (y - yc);
        let mut rad = d2(rec.x1, rec.y1).max(d2(rec.x2, rec.y2));
        while density < DENSITY_TH {
            rad *= 0.75 * 0.75;
            let mut i = 0;
            while i < reg.len() {
                if d2(reg[i].x as f64, reg[i].y as f64) > rad {
                    let p = reg.swap_remove(i);
                    self.release(&p);
                } else {
                    i += 1;
                }
            }
            if reg.len() < 2 {
                return false;
            }
            *rec = Self::region2rect(reg, reg_angle, prec);
            density = Self::density(reg, rec);
        }
        true
    }
}

/// Segments (x1, y1, x2, y2) of an HxW 8-bit image, in pixels of that image.
pub fn line_segments(img: &[u8], (h, w): (usize, usize)) -> Vec<[f32; 4]> {
    let sigma = SIGMA_SCALE / SCALE;
    let half = (sigma * (2.0 * 3.0 * 10f64.ln()).sqrt()).ceil() as usize;
    let blurred = gaussian_blur_u8(img, (h, w), 1 + 2 * half, sigma);
    let (scaled, sh, sw) = resize_exact_u8(&blurred, (h, w));
    if sh < 2 || sw < 2 {
        return Vec::new();
    }
    let prec = std::f64::consts::PI * ANG_TH / 180.0;
    let p = ANG_TH / 180.0;
    let rho = QUANT / prec.sin();
    let (mut det, order) = Detector::new(&scaled, sh, sw, rho);
    let log_nt = 5.0 * ((sw as f64).log10() + (sh as f64).log10()) / 2.0 + 11f64.log10();
    let min_reg_size = (-log_nt / p.log10()) as usize;
    let mut reg = Vec::new();
    let mut lines = Vec::new();
    for &i in &order {
        let i = i as usize;
        if det.used[i] {
            continue;
        }
        let (x, y) = (i % sw, i / sw);
        let reg_angle = det.region_grow(x, y, &mut reg, prec);
        if reg.len() < min_reg_size {
            continue;
        }
        let mut rec = Detector::region2rect(&reg, reg_angle, prec);
        if !det.refine(&mut reg, prec, &mut rec) {
            continue;
        }
        let s = |v: f64| ((v + 0.5) / SCALE) as f32;
        lines.push([s(rec.x1), s(rec.y1), s(rec.x2), s(rec.y2)]);
    }
    lines
}
