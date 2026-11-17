use rayon::prelude::*;

/// `apply_luminance_vignette`: a luminance gain of `exp2(strength * factor)` per channel, the factor a
/// cosine ramp over a radial distance from the crop centre. A scalar gain on every channel
/// moves luma, not color, so the edge change is a luminance vignette, not a tint. cos and exp2
/// come from libm in float64, rounded once.
#[allow(clippy::too_many_arguments)]
pub fn apply_luminance_vignette(
    src: &[f32],
    h: usize,
    w: usize,
    cn: usize,
    strength: f64,
    feather: f64,
    roundness: f64,
    dst: &mut [f32],
) {
    let (cy, cx) = ((h as f64 - 1.0) * 0.5, (w as f64 - 1.0) * 0.5);
    let max_dist = (cx * cx + cy * cy).sqrt().max(1.0) as f32;
    let (cxf, cyf) = (cx as f32, cy as f32);
    let (rx, ry) = (cx.max(1.0) as f32, cy.max(1.0) as f32);
    let (circle, ellipse) = ((1.0 - roundness) as f32, roundness as f32);
    let inner = (1.0 - feather) as f32;
    let span = feather.max(1e-6) as f32;
    let pi = std::f64::consts::PI as f32;
    let strength_f = strength as f32;
    dst.par_chunks_mut(w * cn)
        .zip(src.par_chunks(w * cn))
        .enumerate()
        .for_each(move |(y, (d, s))| {
            let dy = y as f32 - cyf;
            for x in 0..w {
                let dx = x as f32 - cxf;
                let d_circle = (dx * dx + dy * dy).sqrt() / max_dist;
                let d_ellipse = ((dx / rx) * (dx / rx) + (dy / ry) * (dy / ry)).sqrt();
                let dist = d_circle * circle + d_ellipse * ellipse;
                let t = ((dist - inner) / span).clamp(0.0, 1.0);
                let factor = 0.5f32 * (1.0f32 - libm::cos((t * pi) as f64) as f32);
                let gain = libm::exp2((strength_f * factor) as f64) as f32;
                for c in 0..cn {
                    let i = x * cn + c;
                    d[i] = (s[i] * gain).clamp(0.0, 1.0);
                }
            }
        });
}
