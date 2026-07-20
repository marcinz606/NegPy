// Spectral crosstalk moved to the normalization stage (capture-side,
// negative-density domain) — see normalization.wgsl.
struct LabUniforms {
    sharpen: f32,
    chroma_denoise: f32,
    saturation: f32,
    vibrance: f32,
    glow_amount: f32,
    halation_strength: f32,
    scale_factor: f32,
    sharpen_radius_px: f32,
    sharpen_masking: f32,
    sharpen_method: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LabUniforms;
// USM: (blurred L*, original L*) from lab_sharpen_h/v.wgsl.
// RL:  (deconvolved Y, observed Y) from the rl_*.wgsl chain. 1x1 dummy when sharpen=0.
@group(0) @binding(3) var sharpen_tex: texture_2d<f32>;

// Sharpen constants — mirrored from features/lab/logic.py.
const SHARPEN_GATE_LO = 1.5;
const SHARPEN_GATE_HI = 2.0;
const SHARPEN_OVERSHOOT_LIGHT = 1.0;
const SHARPEN_OVERSHOOT_DARK = 2.0;
const SHARPEN_MASK_T_HI = 10.0;
const RL_EPS = 1e-6;

const LUMA_COEFFS = vec3<f32>(0.2126, 0.7152, 0.0722);

// CIELAB L* from linear luminance Y (D65, Yn=1) — mirrors _lab_l_from_y in logic.py.
fn lab_l_from_y(y_in: f32) -> f32 {
    var y = max(y_in, 0.0);
    if (y > 0.008856) { y = pow(y, 1.0 / 3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    return (116.0 * y) - 16.0;
}

// 64-tap Fibonacci spiral — uniform area coverage, smooth Gaussian approximation.
// Points lie in the unit disk; scale by the desired pixel radius when sampling.
const FIBONACCI_64 = array<vec2<f32>, 64>(
    vec2<f32>(0.088388, 0.000000),
    vec2<f32>(-0.112886, 0.103413),
    vec2<f32>(0.017279, -0.196886),
    vec2<f32>(0.142286, 0.185586),
    vec2<f32>(-0.261112, -0.046187),
    vec2<f32>(0.247348, -0.157342),
    vec2<f32>(-0.082733, 0.307763),
    vec2<f32>(-0.157781, -0.303797),
    vec2<f32>(0.342321, 0.125015),
    vec2<f32>(-0.356128, 0.147004),
    vec2<f32>(0.171677, -0.366864),
    vec2<f32>(0.126865, 0.404466),
    vec2<f32>(-0.382373, -0.221593),
    vec2<f32>(0.448567, -0.098616),
    vec2<f32>(-0.273753, 0.389386),
    vec2<f32>(-0.063243, -0.488045),
    vec2<f32>(0.388252, 0.327220),
    vec2<f32>(-0.522466, 0.021606),
    vec2<f32>(0.381099, -0.379244),
    vec2<f32>(-0.025497, 0.551396),
    vec2<f32>(-0.362617, -0.434536),
    vec2<f32>(0.574425, 0.077288),
    vec2<f32>(-0.486709, 0.338640),
    vec2<f32>(0.132997, -0.591185),
    vec2<f32>(0.307615, 0.536829),
    vec2<f32>(-0.601358, -0.191850),
    vec2<f32>(0.584143, -0.269889),
    vec2<f32>(-0.253065, 0.604686),
    vec2<f32>(-0.225855, -0.627935),
    vec2<f32>(0.600976, 0.315856),
    vec2<f32>(-0.667533, 0.175960),
    vec2<f32>(0.379431, -0.590102),
    vec2<f32>(0.120699, 0.702313),
    vec2<f32>(-0.572008, -0.442995),
    vec2<f32>(0.731702, -0.060620),
    vec2<f32>(-0.505760, 0.546712),
    vec2<f32>(0.003684, -0.755181),
    vec2<f32>(0.514305, 0.566946),
    vec2<f32>(-0.772295, -0.071576),
    vec2<f32>(0.625787, -0.474950),
    vec2<f32>(-0.142381, 0.782650),
    vec2<f32>(-0.428884, -0.681539),
    vec2<f32>(0.785920, 0.215388),
    vec2<f32>(-0.733486, 0.376413),
    vec2<f32>(0.289862, -0.781852),
    vec2<f32>(0.317911, 0.780942),
    vec2<f32>(-0.770264, -0.365042),
    vec2<f32>(0.823263, -0.253821),
    vec2<f32>(-0.440157, 0.751049),
    vec2<f32>(-0.184643, -0.859851),
    vec2<f32>(0.724177, 0.514422),
    vec2<f32>(-0.890157, 0.110939),
    vec2<f32>(0.587054, -0.689695),
    vec2<f32>(0.033320, 0.913689),
    vec2<f32>(-0.647727, -0.657276),
    vec2<f32>(0.930014, 0.047552),
    vec2<f32>(-0.724323, 0.598472),
    vec2<f32>(0.130975, -0.938767),
    vec2<f32>(0.542205, 0.787449),
    vec2<f32>(-0.939649, -0.216211),
    vec2<f32>(0.845937, -0.479274),
    vec2<f32>(-0.302492, 0.932436),
    vec2<f32>(-0.410097, -0.899101),
    vec2<f32>(0.916976, 0.389028)
);
// Sum of exp(-2*r²) over all 64 Fibonacci samples — used to normalize the
// accumulator the same way a Gaussian convolution kernel is normalized (sum=1).
const BLOOM_GAUSS_SUM = 27.668145;

// Working-space TRC (Adobe RGB: pure 563/256 gamma). Lab is the encoded->linear
// transition: input samples are decoded; the highlight/sharpen perceptual domain uses the same TRC.
fn oetf_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(x, vec3<f32>(0.45470693));
}

fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return pow(e, vec3<f32>(2.19921875));
}

fn load_lin(coords: vec2<i32>) -> vec3<f32> {
    return oetf_decode(textureLoad(input_tex, coords, 0).rgb);
}

fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    // Linear working RGB -> CIELAB (D65). Input is scene-linear (no TRC decode).
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);

    // Adobe RGB (1998) -> XYZ, D65 (working-space primaries; matches CPU rgb_to_lab_working).
    var x = r * 0.5767309 + g * 0.1855540 + b * 0.1881852;
    var y = r * 0.2973769 + g * 0.6273491 + b * 0.0752741;
    var z = r * 0.0270343 + g * 0.0706872 + b * 0.9911085;

    x = x / 0.95047;
    y = y / 1.00000;
    z = z / 1.08883;

    if (x > 0.008856) { x = pow(x, 1.0/3.0); } else { x = (7.787 * x) + (16.0 / 116.0); }
    if (y > 0.008856) { y = pow(y, 1.0/3.0); } else { y = (7.787 * y) + (16.0 / 116.0); }
    if (z > 0.008856) { z = pow(z, 1.0/3.0); } else { z = (7.787 * z) + (16.0 / 116.0); }

    let l = (116.0 * y) - 16.0;
    let a = 500.0 * (x - y);
    let b_lab = 200.0 * (y - z);

    return vec3<f32>(l, a, b_lab);
}

fn lab_to_rgb(lab: vec3<f32>) -> vec3<f32> {
    var y = (lab.x + 16.0) / 116.0;
    var x = lab.y / 500.0 + y;
    var z = y - lab.z / 200.0;

    if (pow(x, 3.0) > 0.008856) { x = pow(x, 3.0); } else { x = (x - 16.0 / 116.0) / 7.787; }
    if (pow(y, 3.0) > 0.008856) { y = pow(y, 3.0); } else { y = (y - 16.0 / 116.0) / 7.787; }
    if (pow(z, 3.0) > 0.008856) { z = pow(z, 3.0); } else { z = (z - 16.0 / 116.0) / 7.787; }

    x = x * 0.95047;
    y = y * 1.00000;
    z = z * 1.08883;

    // XYZ -> Adobe RGB (1998), D65. Returns scene-linear (no encode).
    let r = x * 2.0413690 + y * -0.5649464 + z * -0.3446944;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0134474 + y * -0.1183897 + z * 1.0154096;

    return max(vec3<f32>(r, g, b), vec3<f32>(0.0));
}

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    let v = max(c.r, max(c.g, c.b));
    let m = min(c.r, min(c.g, c.b));
    let d = v - m;
    var h: f32;
    var s: f32;
    if (d == 0.0) { h = 0.0; }
    else if (v == c.r) { h = (c.g - c.b) / d; }
    else if (v == c.g) { h = (c.b - c.r) / d + 2.0; }
    else { h = (c.r - c.g) / d + 4.0; }
    h = fract(h / 6.0);
    if (v == 0.0) { s = 0.0; } else { s = d / v; }
    return vec3<f32>(h, s, v);
}

fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let h = c.x * 6.0;
    let s = c.y;
    let v = c.z;
    let i = floor(h);
    let f = h - i;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    let cond = i32(i) % 6;
    if (cond == 0) { return vec3<f32>(v, t, p); }
    else if (cond == 1) { return vec3<f32>(q, v, p); }
    else if (cond == 2) { return vec3<f32>(p, v, t); }
    else if (cond == 3) { return vec3<f32>(p, q, v); }
    else if (cond == 4) { return vec3<f32>(t, p, v); }
    else { return vec3<f32>(v, p, q); }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = load_lin(coords);

    // 1. Chroma Denoise
    // Variable-radius blur of the CIELAB a*/b* channels. The Fibonacci-disk taps
    // (weighted exp(-r^2 * 2), normalised by BLOOM_GAUSS_SUM) approximate a Gaussian
    // of sigma = radius / 2, so radius = 2 * chroma_denoise * scale_factor matches the
    // CPU path's GaussianBlur sigma of (chroma_denoise * scale_factor). This previously
    // used a fixed 5x5 kernel that ignored the slider, so the strength never changed on
    // GPU — inadequate for the heavy chroma noise of inverted C41 negatives.
    if (params.chroma_denoise > 0.0) {
        let radius = 2.0 * params.chroma_denoise * params.scale_factor;
        if (radius >= 0.5) {
            let current_lab = rgb_to_lab(color);
            var blur_ab = vec2<f32>(0.0);
            for (var tap = 0; tap < 64; tap++) {
                let offset = FIBONACCI_64[tap];
                let s_coord = clamp(coords + vec2<i32>(offset * radius), vec2<i32>(0), vec2<i32>(dims) - 1);
                let s_lab = rgb_to_lab(load_lin(s_coord));
                let r = length(offset);
                let w = exp(-r * r * 2.0);
                blur_ab += s_lab.yz * w;
            }
            blur_ab = blur_ab / BLOOM_GAUSS_SUM;
            color = lab_to_rgb(vec3<f32>(current_lab.x, blur_ab.x, blur_ab.y));
        }
    }

    // 3. Vibrance
    if (params.vibrance != 1.0) {
        var lab = rgb_to_lab(color);
        let chroma = length(lab.yz);
        let muted_mask = clamp(1.0 - (chroma / 60.0), 0.0, 1.0);
        let boost = (params.vibrance - 1.0) * muted_mask;
        lab.y = lab.y * (1.0 + boost);
        lab.z = lab.z * (1.0 + boost);
        color = lab_to_rgb(lab);
    }

    // 4. Global Saturation (CIELAB chroma scaling — preserves L*)
    if (params.saturation != 1.0) {
        var lab = rgb_to_lab(color);
        lab.y = lab.y * params.saturation;
        lab.z = lab.z * params.saturation;
        color = lab_to_rgb(lab);
    }

    // 5. Sharpening — sharpen_tex holds precomputed state from the h/v (USM) or
    // rl_*.wgsl (deconvolution) passes over input_tex, which the earlier
    // chroma/vibrance/saturation stages never touch in L*/Y (they only rewrite
    // a*/b*), so the state matches what `color` would produce.
    if (params.sharpen > 0.0 && params.sharpen_method < 0.5) {
        // Unsharp mask — mirrors apply_output_sharpening in logic.py.
        let blur_l = textureLoad(sharpen_tex, coords, 0).x;
        let current_lab = rgb_to_lab(color);
        let diff = current_lab.x - blur_l;
        let gate = smoothstep(SHARPEN_GATE_LO, SHARPEN_GATE_HI, abs(diff));
        var gain = params.sharpen * 2.5 * gate;

        // Local 3x3 stats over the original L* (sharpen_tex.y), clamped coords —
        // matches the CPU's BORDER_REPLICATE erode/dilate and boxed gradient.
        var l_min = 1e9;
        var l_max = -1e9;
        var grad_box = 0.0;
        for (var j = -1; j <= 1; j++) {
            for (var i = -1; i <= 1; i++) {
                let p = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1);
                let lv = textureLoad(sharpen_tex, p, 0).y;
                l_min = min(l_min, lv);
                l_max = max(l_max, lv);
                if (params.sharpen_masking > 0.0) {
                    let px = clamp(p + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let mx = clamp(p - vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let py = clamp(p + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let my = clamp(p - vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let gx = (textureLoad(sharpen_tex, px, 0).y - textureLoad(sharpen_tex, mx, 0).y) * 0.5;
                    let gy = (textureLoad(sharpen_tex, py, 0).y - textureLoad(sharpen_tex, my, 0).y) * 0.5;
                    grad_box += sqrt(gx * gx + gy * gy);
                }
            }
        }

        // Edge mask: sharpening ramps in with local gradient; higher masking
        // raises the bar, protecting flat areas (sky, skin, grain).
        if (params.sharpen_masking > 0.0) {
            let t = SHARPEN_MASK_T_HI * params.sharpen_masking;
            gain = gain * smoothstep(0.5 * t, t, (grad_box / 9.0) * params.scale_factor);
        }

        // Overshoot clamp to the local range kills halos; tighter above than
        // below because L*-domain USM exaggerates light halos.
        var l_new = current_lab.x + diff * gain;
        l_new = clamp(l_new, l_min - SHARPEN_OVERSHOOT_DARK, l_max + SHARPEN_OVERSHOOT_LIGHT);
        l_new = clamp(l_new, 0.0, 100.0);
        color = lab_to_rgb(vec3<f32>(l_new, current_lab.y, current_lab.z));
    } else if (params.sharpen > 0.0) {
        // Richardson-Lucy — mirrors apply_rl_sharpening in logic.py. sharpen_tex
        // is (deconvolved Y, observed Y); apply the luminance ratio to RGB
        // (chroma-preserving), gated by the shared edge mask over L*(obs).
        let d = textureLoad(sharpen_tex, coords, 0);
        var gain = params.sharpen;
        if (params.sharpen_masking > 0.0) {
            var grad_box = 0.0;
            for (var j = -1; j <= 1; j++) {
                for (var i = -1; i <= 1; i++) {
                    let p = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let px = clamp(p + vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let mx = clamp(p - vec2<i32>(1, 0), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let py = clamp(p + vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let my = clamp(p - vec2<i32>(0, 1), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let gx = (lab_l_from_y(textureLoad(sharpen_tex, px, 0).y) - lab_l_from_y(textureLoad(sharpen_tex, mx, 0).y)) * 0.5;
                    let gy = (lab_l_from_y(textureLoad(sharpen_tex, py, 0).y) - lab_l_from_y(textureLoad(sharpen_tex, my, 0).y)) * 0.5;
                    grad_box += sqrt(gx * gx + gy * gy);
                }
            }
            let t = SHARPEN_MASK_T_HI * params.sharpen_masking;
            gain = gain * smoothstep(0.5 * t, t, (grad_box / 9.0) * params.scale_factor);
        }
        let ratio = d.x / max(d.y, RL_EPS);
        color = color * max(1.0 + (ratio - 1.0) * gain, 0.0);
    }

    // 6. Glow and Halation
    // Radii match the CPU defaults (base_r at scale_factor=1): glow=15px, halation=25px.
    // Accumulate highlight-weighted Gaussian samples then divide by the fixed kernel
    // weight sum (BLOOM_GAUSS_SUM) — mirrors how a normalised Gaussian convolution
    // kernel divides by its total weight, so intensity decays naturally with distance
    // rather than being renormalised back up to full highlight brightness.
    if (params.glow_amount > 0.0 || params.halation_strength > 0.0) {
        let HIGHLIGHT_THRESHOLD = 0.5;
        // Halation mask in LINEAR reflectance: fixed by scene exposure, so the
        // footprint doesn't move with grade/density (mirrors CPU logic.py).
        let HALATION_THRESHOLD_LINEAR = 0.65;
        let GLOW_RADIUS = 15.0;
        let HAL_RADIUS = 25.0;

        var glow_accum = vec3<f32>(0.0);
        var hal_accum = vec3<f32>(0.0);

        for (var tap = 0; tap < 64; tap++) {
            let offset = FIBONACCI_64[tap];

            if (params.glow_amount > 0.0) {
                let g_off = offset * GLOW_RADIUS;
                let g_coord = clamp(coords + vec2<i32>(g_off), vec2<i32>(0), vec2<i32>(dims) - 1);
                let g_samp = load_lin(g_coord);
                // Glow is a print-side lens effect: mask stays in display domain.
                let g_luma = dot(oetf_encode(g_samp), LUMA_COEFFS);
                let g_hl = max(0.0, (g_luma - HIGHLIGHT_THRESHOLD) / (1.0 - HIGHLIGHT_THRESHOLD));
                let g_r = length(offset);  // normalised radius in [0,1]
                let g_w = exp(-g_r * g_r * 2.0);
                glow_accum += g_samp * (g_hl * g_w);
            }

            if (params.halation_strength > 0.0) {
                let h_off = offset * HAL_RADIUS;
                let h_coord = clamp(coords + vec2<i32>(h_off), vec2<i32>(0), vec2<i32>(dims) - 1);
                let h_samp = load_lin(h_coord);
                let h_luma = dot(h_samp, LUMA_COEFFS);
                let h_hl = max(0.0, (h_luma - HALATION_THRESHOLD_LINEAR) / (1.0 - HALATION_THRESHOLD_LINEAR));
                let h_r = length(offset);
                let h_w = exp(-h_r * h_r * 2.0);
                hal_accum += vec3<f32>(h_samp.r, h_samp.r * 0.3, h_samp.r * 0.05) * (h_hl * h_w);
            }
        }

        // Scattered light is added exposure — additive in linear, clamped at store.
        if (params.glow_amount > 0.0) {
            color = color + (glow_accum / BLOOM_GAUSS_SUM) * params.glow_amount;
        }

        if (params.halation_strength > 0.0) {
            color = color + (hal_accum / BLOOM_GAUSS_SUM) * params.halation_strength;
        }
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}