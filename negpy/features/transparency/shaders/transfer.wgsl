// Transparency transfer curve — GPU mirror of features/transparency/logic.py.
// Replaces the print curve (exposure.wgsl) for every slide.
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
    // Shadows/Highlights WB: xyz = shadow CMY density offset, w = split centre.
    shadow_cmy: vec4<f32>,
    // Shadows/Highlights WB: xyz = highlight CMY density offset, w = split sharpness.
    highlight_cmy: vec4<f32>,
    // x = width of the black taper, in density. y = positive_source (nonzero skips
    // display_rendering below). zw unused.
    zone_taper: vec4<f32>,
    // Cast Removal affine on density: per-channel gain and offset (w lane unused).
    cast_gain: vec4<f32>,
    cast_offset: vec4<f32>,
    // Dye Separation: xyz = per-channel k (global + trim; no paper matrix to compose
    // the trims into instead). w = Separation Damping (0 = off).
    separation: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: TransferUniforms;

// width * log(1 + exp(x / width)), overflow-safe — mirrors logic.py::_softplus.
fn softplus(x: f32, width: f32) -> f32 {
    let t = x / width;
    return width * (log(1.0 + exp(-abs(t))) + max(t, 0.0));
}

// Scene-linear -> display-linear: Narkowicz's closed-form fit to the ACES RRT + sRGB
// ODT. Mirrors logic.py::display_rendering — a published filmic curve with a real
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

// One pixel's effective dye-separation k; mirrors separation_damping_gain in
// exposure/logic.py. 0.35 mirrors separation_damping_ref_spread in models.py and
// the copy in exposure.wgsl -- change all three. Copied here because WGSL has no
// includes.
fn separation_damping_gain(k: f32, damping: f32, chroma: f32) -> f32 {
    if (k <= 0.0) {
        return 0.0;
    }
    let h = (0.35 - chroma) / (0.35 + chroma);
    return min(pow(k, (1.0 - damping) + damping * h), 3.0);
}

// One channel's capture density through every control above Dye Separation. Mirrors
// apply_transfer_curve's shape().
fn shape(d_in: f32, ch: i32) -> f32 {
    var d = d_in;

    // Cast Removal: the channel's neutral refs onto green's. First, so every
    // control below shapes the corrected signal.
    d = d * params.cast_gain[ch] + params.cast_offset[ch];

    d = d - params.exposure_offset + params.cmy[ch] * params.density_range;
    d = params.pivot + (d - params.pivot) * params.contrast;

    // Shadows/Highlights WB: regional CMY, mirroring exposure.wgsl's own blend.
    if (params.shadow_cmy[ch] != 0.0 || params.highlight_cmy[ch] != 0.0) {
        let w_sh = 1.0 / (1.0 + exp(-params.highlight_cmy.w * (d - params.shadow_cmy.w)));
        let w_hi = 1.0 - w_sh;
        d = d + params.shadow_cmy[ch] * w_sh + params.highlight_cmy[ch] * w_hi;
    }

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

    return d;
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
    var dens: vec3<f32>;
    let capture = norm * params.density_range;
    for (var ch = 0; ch < 3; ch++) {
        dens[ch] = shape(capture[ch], ch);
    }

    // Dye Separation: each channel scales its own deviation from a reference density by
    // its own k: the mean of the channels, each softly capped on the capture and then
    // shaped. Mirrors logic.py (see apply_transfer_curve); 2.5, 1.5 and 0.15 mirror
    // SEPARATION_CAP_LEVEL/SPREAD/SOFTNESS.
    // Separation Damping makes each channel's k chroma-dependent per pixel, from the
    // same chroma but each channel's own k (see separation_damping_gain).
    if (any(params.separation.xyz != vec3<f32>(1.0))) {
        let d_lo = min(capture.x, min(capture.y, capture.z));
        let cap = 2.5 + softplus(d_lo + (1.5 - 2.5), 0.15);
        let capped = vec3<f32>(
            shape(cap - softplus(cap - capture.x, 0.15), 0),
            shape(cap - softplus(cap - capture.y, 0.15), 1),
            shape(cap - softplus(cap - capture.z, 0.15), 2),
        );
        let d_ref = (capped.x + capped.y + capped.z) / 3.0;
        let e = dens - vec3<f32>(d_ref);
        if (params.separation.w > 0.0) {
            let c = capped - vec3<f32>(d_ref);
            let chroma = sqrt(((c.x - c.y) * (c.x - c.y) + (c.y - c.z) * (c.y - c.z) + (c.x - c.z) * (c.x - c.z)) / 3.0);
            let k_eff = vec3<f32>(
                separation_damping_gain(params.separation.x, params.separation.w, chroma),
                separation_damping_gain(params.separation.y, params.separation.w, chroma),
                separation_damping_gain(params.separation.z, params.separation.w, chroma),
            );
            dens = vec3<f32>(d_ref) + k_eff * e;
        } else {
            dens = vec3<f32>(d_ref) + params.separation.xyz * e;
        }
    }

    for (var ch = 0; ch < 3; ch++) {
        // Baseline + display rendering last: the controls above shape the scene. A
        // positive source skips both (baseline_gain arrives as 1.0), matching
        // logic.py::apply_transfer_curve.
        let scene = pow(10.0, -dens[ch]) * params.baseline_gain;
        if (params.zone_taper.y != 0.0) {
            res[ch] = oetf_encode(clamp(scene, 0.0, 1.0));
        } else {
            res[ch] = oetf_encode(display_rendering(scene));
        }
    }

    textureStore(output_tex, coords, vec4<f32>(res, 1.0));
}
