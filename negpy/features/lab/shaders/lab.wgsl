struct LabUniforms {
    crosstalk_row0: vec4<f32>,
    crosstalk_row1: vec4<f32>,
    crosstalk_row2: vec4<f32>,
    strength: f32,
    sharpen: f32,
    chroma_denoise: f32,
    saturation: f32,
    vibrance: f32,
    glow_amount: f32,
    halation_strength: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: LabUniforms;

const gauss_kernel = array<f32, 25>(
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765
);

const LUMA_COEFFS = vec3<f32>(0.2126, 0.7152, 0.0722);

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

// Working-space TRC (ProPhoto ROMM: gamma 1.8 + linear toe). Lab is the encoded->linear
// transition: input samples are decoded; the highlight/sharpen perceptual domain uses the same TRC.
fn oetf_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(pow(x, vec3<f32>(0.55555556)), x * 16.0, x < vec3<f32>(0.001953125));
}

fn oetf_decode(c: vec3<f32>) -> vec3<f32> {
    let e = max(c, vec3<f32>(0.0));
    return select(pow(e, vec3<f32>(1.8)), e / 16.0, e < vec3<f32>(0.03125));
}

fn load_lin(coords: vec2<i32>) -> vec3<f32> {
    return oetf_decode(textureLoad(input_tex, coords, 0).rgb);
}

fn rgb_to_lab(rgb: vec3<f32>) -> vec3<f32> {
    // Linear Adobe RGB -> CIELAB (D65). Input is scene-linear (no sRGB decode).
    let r = max(rgb.r, 0.0);
    let g = max(rgb.g, 0.0);
    let b = max(rgb.b, 0.0);

    // ProPhoto RGB (ROMM) -> XYZ, D50 (working-space primaries; matches CPU rgb_to_lab_working).
    var x = r * 0.7976749 + g * 0.1351917 + b * 0.0313534;
    var y = r * 0.2880402 + g * 0.7118741 + b * 0.0000857;
    var z = r * 0.0000000 + g * 0.0000000 + b * 0.8252100;

    x = x / 0.96422;
    y = y / 1.00000;
    z = z / 0.82521;

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

    x = x * 0.96422;
    y = y * 1.00000;
    z = z * 0.82521;

    // XYZ -> ProPhoto RGB (ROMM), D50. Returns scene-linear (no encode).
    let r = x * 1.3459433 + y * -0.2556075 + z * -0.0511118;
    let g = x * -0.5445989 + y * 1.5081673 + z * 0.0205351;
    let b = x * 0.0000000 + y * 0.0000000 + z * 1.2118128;

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
    if (params.chroma_denoise > 0.0) {
        let current_lab = rgb_to_lab(color);
        var blur_ab = vec2<f32>(0.0);
        for (var j = -2; j <= 2; j++) {
            for (var i = -2; i <= 2; i++) {
                let sample_coords = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1);
                let sample_lab = rgb_to_lab(load_lin(sample_coords));
                let weight = gauss_kernel[(j + 2) * 5 + (i + 2)];
                blur_ab += sample_lab.yz * weight;
            }
        }
        color = lab_to_rgb(vec3<f32>(current_lab.x, blur_ab.x, blur_ab.y));
    }

    // 2. Spectral Crosstalk
    if (params.strength > 0.0) {
        let epsilon = 1e-6;
        let dens = -log(max(color, vec3<f32>(epsilon))) / 2.302585;
        let m0 = params.crosstalk_row0.xyz;
        let m1 = params.crosstalk_row1.xyz;
        let m2 = params.crosstalk_row2.xyz;
        let mixed_dens = vec3<f32>(dot(dens, m0), dot(dens, m1), dot(dens, m2));
        color = pow(vec3<f32>(10.0), -mixed_dens);
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

    // 5. Sharpening
    if (params.sharpen > 0.0) {
        var blur_luma = 0.0;
        for (var j = -2; j <= 2; j++) {
            for (var i = -2; i <= 2; i++) {
                let sample_coords = clamp(coords + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1);
                let sample_color = load_lin(sample_coords);
                let weight = gauss_kernel[(j + 2) * 5 + (i + 2)];
                blur_luma += dot(oetf_encode(sample_color), LUMA_COEFFS) * weight;
            }
        }
        // Derive the USM ratio from input_tex (matching the blur source). Using
        // post-saturation `color` here against a pre-lab `blur_luma` would
        // synthesise a phantom edge wherever the lab stages shifted perceptual
        // luma — most visibly on saturated reds, where CIELAB sat preserves L*
        // but cuts G/B and drops the perceptual luma far enough below
        // neighbouring blur to drive the ratio negative and crush the pixel.
        let input_color = load_lin(coords);
        let input_luma = dot(oetf_encode(input_color), LUMA_COEFFS);
        let amount = params.sharpen * 2.5;
        let sharpened_luma = clamp(input_luma + (input_luma - blur_luma) * amount, 0.0, 1.0);
        let ratio = sharpened_luma / max(input_luma, 1e-6);
        color = oetf_decode(oetf_encode(color) * ratio);
    }

    // 6. Glow and Halation
    // Radii match the CPU defaults (base_r at scale_factor=1): glow=15px, halation=25px.
    // Accumulate highlight-weighted Gaussian samples then divide by the fixed kernel
    // weight sum (BLOOM_GAUSS_SUM) — mirrors how a normalised Gaussian convolution
    // kernel divides by its total weight, so intensity decays naturally with distance
    // rather than being renormalised back up to full highlight brightness.
    if (params.glow_amount > 0.0 || params.halation_strength > 0.0) {
        let HIGHLIGHT_THRESHOLD = 0.5;
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
                // Highlight mask in display domain (keeps 0.5 threshold); bloom is linear.
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
                let h_luma = dot(oetf_encode(h_samp), LUMA_COEFFS);
                let h_hl = max(0.0, (h_luma - HIGHLIGHT_THRESHOLD) / (1.0 - HIGHLIGHT_THRESHOLD));
                let h_r = length(offset);
                let h_w = exp(-h_r * h_r * 2.0);
                hal_accum += vec3<f32>(h_samp.r, h_samp.r * 0.3, h_samp.r * 0.05) * (h_hl * h_w);
            }
        }

        if (params.glow_amount > 0.0) {
            let glow_color = (glow_accum / BLOOM_GAUSS_SUM) * params.glow_amount;
            color = 1.0 - (1.0 - color) * (1.0 - glow_color);
        }

        if (params.halation_strength > 0.0) {
            let hal_color = (hal_accum / BLOOM_GAUSS_SUM) * params.halation_strength;
            color = 1.0 - (1.0 - color) * (1.0 - hal_color);
        }
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}