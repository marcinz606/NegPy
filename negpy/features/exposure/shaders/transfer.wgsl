// Transparency transfer curve — GPU mirror of features/exposure/transfer.py.
// Replaces the print curve (exposure.wgsl) when E-6 runs with Normalize off.
// Every term vanishes at its neutral value so the default render is an exact
// pass-through of the capture, matching the CPU path bit-for-bit closely enough
// for test_transparency_transfer.py's parity bound.

struct TransferUniforms {
    exposure_offset: f32,
    contrast: f32,
    density_range: f32,
    zone_k: f32,
    pivot: f32,
    toe_knee: f32,
    sh_knee: f32,
    baseline_gain: f32,
    // Per-channel knee heights, knee widths and WB density offsets (w lane unused).
    toe: vec4<f32>,
    shoulder: vec4<f32>,
    toe_width: vec4<f32>,
    shoulder_width: vec4<f32>,
    cmy: vec4<f32>,
    // Zone Density: (shadow ΔD, highlight ΔD, shadow centre, highlight centre).
    zone: vec4<f32>,
    // x = width of the black taper, in density; yzw unused.
    zone_taper: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: TransferUniforms;

// width * log(1 + exp(x / width)), overflow-safe — mirrors transfer.py::_softplus.
fn softplus(x: f32, width: f32) -> f32 {
    let t = x / width;
    return width * (log(1.0 + exp(-abs(t))) + max(t, 0.0));
}

// Scene-linear -> display-linear: Narkowicz's closed-form fit to the ACES RRT + sRGB
// ODT. Mirrors transfer.py::display_rendering — a published filmic curve with a real
// toe and shoulder, so highlights roll off to display white instead of stopping at
// wherever the sensor's white level fell.
fn display_rendering(v: f32) -> f32 {
    let x = max(v, 0.0);
    let num = x * (2.51 * x + 0.03);
    let den = x * (2.43 * x + 0.59) + 0.14;
    return clamp(num / max(den, 1e-8), 0.0, 1.0);
}

// Working-space OETF (Adobe RGB: pure 563/256 gamma). The GPU exposure stage emits
// display-encoded values — every stage behind it expects that — so the transfer curve
// has to encode here too. The CPU mirror encodes once at the end of the engine instead.
fn oetf_encode(t: f32) -> f32 {
    let x = max(t, 0.0);
    return pow(x, 0.45470693);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let norm = textureLoad(input_tex, coords, 0).rgb;

    var res: vec3<f32>;
    for (var ch = 0; ch < 3; ch++) {
        var d = norm[ch] * params.density_range;

        d = d - params.exposure_offset + params.cmy[ch] * params.density_range;
        d = params.pivot + (d - params.pivot) * params.contrast;

        // Zone Density: mid-sparing offsets on the print path's own weights. Positive
        // adds density, so it darkens. After contrast, before the knees — as on the print.
        if (params.zone.x != 0.0 || params.zone.y != 0.0) {
            // Fade the shadow lift out at the bottom of the window. A print bounds a
            // shadow burn at paper black; this curve has no paper, so without the taper a
            // lift walks the black point up with it and the frame stops having blacks.
            let t = clamp((params.density_range - d) / params.zone_taper.x, 0.0, 1.0);
            let taper = t * t * (3.0 - 2.0 * t);
            let w_sh = taper / (1.0 + exp(-params.zone_k * (d - params.zone.z)));
            let w_hi = 1.0 - 1.0 / (1.0 + exp(-params.zone_k * (d - params.zone.w)));
            d = d + params.zone.x * w_sh + params.zone.y * w_hi;
        }

        // Shadows sit at high density, highlights at low, so the toe compresses
        // above its knee and the shoulder below its own.
        let t = params.toe[ch];
        if (t != 0.0) {
            d = d - t * softplus(d - params.toe_knee, params.toe_width[ch]);
        }
        let s = params.shoulder[ch];
        if (s != 0.0) {
            d = d + s * softplus(params.sh_knee - d, params.shoulder_width[ch]);
        }

        // Baseline + display rendering last: the controls above shape the scene.
        res[ch] = oetf_encode(display_rendering(pow(10.0, -d) * params.baseline_gain));
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}
