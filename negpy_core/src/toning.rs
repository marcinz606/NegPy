//! `_apply_chemical_toning_jit` of negpy/features/toning/logic.py: silver-ledger toning on
//! linear reflectance, in float64 as Numba types it.

// The steps index several per-pixel arrays of one run in step.
#![allow(clippy::needless_range_loop)]

use multiversion::multiversion;
use rayon::prelude::*;

use crate::mathf::{exp10_f64, log10_f64, pow_f64};

/// One bath: strength, reference density, exponent, per-channel covering-power gain.
#[derive(Clone, Copy)]
pub struct Bath {
    pub strength: f64,
    pub d_ref: f64,
    pub power: f64,
    pub gain: [f64; 3],
}

#[derive(Clone, Copy)]
pub struct Toning {
    pub selenium: Bath,
    pub sepia: Bath,
    pub gold: Bath,
    pub gold_sepia_gain: [f64; 3],
    pub blue: Bath,
    pub copper: Bath,
    pub vanadium: Bath,
    /// Gold converts all densities alike (a lith print) instead of highlights first.
    pub gold_flat: bool,
}

impl Bath {
    /// Conversion per pixel of `d0`, capped at 1: strength (D / d_ref)^power when `rising`,
    /// else strength (1 - D / d_ref)^power. All zero while the bath is off.
    fn conversion(&self, d0: &[f64], out: &mut [f64], rising: bool) {
        if self.strength <= 0.0 {
            out.fill(0.0);
            return;
        }
        for (c, &d) in out.iter_mut().zip(d0) {
            let frac = (d / self.d_ref).min(1.0);
            let x = if rising { frac } else { 1.0 - frac };
            *c = (self.strength * pow_f64(x, self.power)).min(1.0);
        }
    }
}

pub fn apply_chemical_toning(img: &[f32], dst: &mut [f32], t: &Toning) {
    dst.par_chunks_mut(3 * RUN)
        .zip(img.par_chunks(3 * RUN))
        .for_each(|(d, s)| tone_run(s, d, t));
}

/// Pixels per pass: each bath's conversion runs as a flat loop over a run, so it vectorizes.
const RUN: usize = 256;

#[multiversion(targets(
    "x86_64+avx512f+avx512bw+avx512vl+avx512dq",
    "x86_64+avx2+fma",
    "aarch64+neon"
))]
fn tone_run(src: &[f32], dst: &mut [f32], t: &Toning) {
    let px = src.as_chunks::<3>().0;
    let n = px.len();
    let mut d3 = [[0f64; RUN]; 3];
    for (ch, d) in d3.iter_mut().enumerate() {
        for (d, p) in d.iter_mut().zip(px) {
            *d = -log10_f64((p[ch] as f64).clamp(1e-6, 1.0));
        }
    }
    let mut d0 = [0f64; RUN];
    for (i, m) in d0[..n].iter_mut().enumerate() {
        *m = (d3[0][i] + d3[1][i] + d3[2][i]) / 3.0;
    }
    let d0 = &d0[..n];
    let mut c = [[0f64; RUN]; 6];
    t.selenium.conversion(d0, &mut c[0][..n], true);
    t.sepia.conversion(d0, &mut c[1][..n], false);
    if t.gold_flat {
        c[2][..n].fill(t.gold.strength.clamp(0.0, 1.0));
    } else {
        t.gold.conversion(d0, &mut c[2][..n], false);
    }
    t.blue.conversion(d0, &mut c[3][..n], true);
    t.copper.conversion(d0, &mut c[4][..n], true);
    t.vanadium.conversion(d0, &mut c[5][..n], false);
    // The ledger leaves each channel's covering-power factor k.
    let mut k = [[0f64; RUN]; 3];
    for i in 0..n {
        let [c_sel, c_sep, c_au, c_blue, c_cu, c_van] = std::array::from_fn(|j| c[j][i]);
        // One silver reservoir: each bath claims a share of what is left; gold also plates
        // the sulfide.
        let mut a = 1.0;
        let f_sel = a * c_sel;
        a -= f_sel;
        let mut f_sep = a * c_sep;
        a -= f_sep;
        let f_au = a * c_au;
        a -= f_au;
        let f_ausp = f_sep * c_au;
        f_sep -= f_ausp;
        let f_blue = a * c_blue;
        a -= f_blue;
        let f_cu = a * c_cu;
        a -= f_cu;
        let f_van = a * c_van;
        a -= f_van;
        for ch in 0..3 {
            k[ch][i] = a
                + f_sel * t.selenium.gain[ch]
                + f_sep * t.sepia.gain[ch]
                + f_au * t.gold.gain[ch]
                + f_ausp * t.sepia.gain[ch] * t.gold_sepia_gain[ch]
                + f_blue * t.blue.gain[ch]
                + f_cu * t.copper.gain[ch]
                + f_van * t.vanadium.gain[ch];
        }
    }
    let out = dst.as_chunks_mut::<3>().0;
    for ch in 0..3 {
        for i in 0..n {
            out[i][ch] = exp10_f64(-(d3[ch][i] * k[ch][i])).clamp(0.0, 1.0) as f32;
        }
    }
}
