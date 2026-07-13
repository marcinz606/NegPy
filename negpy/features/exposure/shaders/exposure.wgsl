struct ExposureUniforms {
    pivots: vec4<f32>,
    slopes: vec4<f32>,
    curvatures: vec4<f32>,
    cmy_offsets: vec4<f32>,
    shadow_cmy: vec4<f32>,
    highlight_cmy: vec4<f32>,
    // Per-channel knee widths (per_channel_widths, pre-clamped CPU-side): six
    // lanes in the ex-scalar toe/shoulder/midtone_gamma slots (those values ride
    // the vec4 w-lanes) and the former flare pad, keeping the 256B layout.
    toe_width_r: f32,
    toe_width_g: f32,
    toe_width_b: f32,
    shoulder_width_r: f32,
    // Zone Density ΔD shadow offset in the ex-d_min slot (the curve reads d_min_rgb).
    shadow_density: f32,
    d_max: f32,
    a_toe_base: f32,
    a_sh_base: f32,
    // Free slot (ex-width_ref; toeshoulder_width_ref is the 2.5 literal below).
    pad0: f32,
    toe_height: f32,
    sh_height: f32,
    zone_center: f32,
    shoulder_width_g: f32,
    // Free slot (ex-surround_gamma).
    pad1: f32,
    mode: u32,
    v_star: f32,
    shoulder_width_b: f32,
    gamma_width: f32,
    use_dye: u32,
    // Black point compensation flag (0/1); was the 16B pad before d_min_rgb.
    bpc: f32,
    // Per-channel paper-white floor (base+fog incl. tint) in xyz; w carries the
    // Zone Density ΔD highlight offset (the block is full at 256B).
    d_min_rgb: vec4<f32>,
    // Row-normalized dye coupling rows (D_rgb = M * D_dye above base).
    dye_r: vec4<f32>,
    dye_g: vec4<f32>,
    dye_b: vec4<f32>,
    // Dodge/burn: xyz = per-channel normalized-space size of one EV stop
    // (local_ev_scale), w = enable flag (0 -> ev_tex is a dummy, skip it).
    ev_scale: vec4<f32>,
    // Split Grade per-channel zone contrast gains (split_grade_deltas), w free.
    // These rows push the block past 256B: exposure spans two UBO slots.
    split_sh: vec4<f32>,
    split_hi: vec4<f32>,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ExposureUniforms;
// Per-pixel dodge/burn EV map, rasterised on the CPU (shared with the CPU path).
@group(0) @binding(3) var ev_tex: texture_2d<f32>;

fn fast_sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        let z = exp(x);
        return z / (1.0 + z);
    }
}

// Numerically stable softplus: log(1 + exp(x)). Antiderivative of the sigmoid.
fn softplus(x: f32) -> f32 {
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

// Working-space OETF (ProPhoto ROMM: gamma 1.8 + linear toe); feeds the encoded
// perceptual region (clahe, retouch) before lab decodes back to linear.
fn oetf_encode(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    return select(pow(x, 0.55555556), x * 16.0, x < 0.001953125);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coords = vec2<i32>(i32(gid.x), i32(gid.y));
    var color = textureLoad(input_tex, coords, 0);

    // B&W: panchromatic luminance BEFORE the curve (single-density response).
    if (params.mode == 1u) {
        let luma = dot(color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        color = vec4<f32>(luma, luma, luma, color.a);
    }

    let eps = 1e-6;
    // Asymmetric H&D print curve (toe-linear-shoulder); mirrors the CPU
    // _apply_print_curve_kernel. toe -> shadow (paper-black) bound, shoulder ->
    // highlight (paper-white) bound. a_toe_base/a_sh_base carry shadow/highlight
    // sharpness; width sets gentleness, slider sets roll-off height.
    let toe_w3 = vec3<f32>(params.toe_width_r, params.toe_width_g, params.toe_width_b);
    let sh_w3 = vec3<f32>(params.shoulder_width_r, params.shoulder_width_g, params.shoulder_width_b);
    // 2.5 mirrors toeshoulder_width_ref in models.py — change together.
    let a_hl = params.a_sh_base * 2.5 / max(sh_w3, vec3<f32>(eps));
    let a_sh_base = params.a_toe_base * 2.5 / max(toe_w3, vec3<f32>(eps));
    // Per-channel toe/shoulder, pre-scaled CPU-side (per_channel_toe_shoulder).
    // The uniform block is full at 256B, so the vec4 w-lanes carry them.
    let toe3 = vec3<f32>(params.pivots.w, params.slopes.w, params.curvatures.w);
    let sh3 = vec3<f32>(params.cmy_offsets.w, params.shadow_cmy.w, params.highlight_cmy.w);
    // Per-channel Snap rides the dye-row w-lanes; scalar midtone_gamma is layout-only.
    let mg3 = vec3<f32>(params.dye_r.w, params.dye_g.w, params.dye_b.w);
    let toe_neg = toe3 < vec3<f32>(0.0);
    // Negative toe: tighten shadow roll-off (sharper knee) rather than extending
    // d_max_eff beyond paper black (perceptually near-zero effect above d_max).
    let a_sh = select(a_sh_base, a_sh_base * (1.0 - toe3 * 4.0), toe_neg);
    let d_min_rgb = params.d_min_rgb.xyz;
    let d_min_eff = max(d_min_rgb + sh3 * params.sh_height, vec3<f32>(0.0));
    let d_max_base = select(vec3<f32>(params.d_max) - toe3 * params.toe_height, vec3<f32>(params.d_max), toe_neg);
    let d_max_eff = max(d_max_base, d_min_eff + vec3<f32>(0.1));

    // Dodge/burn print-exposure offset (EV stops), same domain as cmy_offsets.
    var ev = 0.0;
    if (params.ev_scale.w != 0.0) {
        ev = textureLoad(ev_tex, coords, 0).r;
    }

    var dens: vec3<f32>;

    for (var ch = 0; ch < 3; ch++) {
        let val = color[ch] + params.cmy_offsets[ch] + ev * params.ev_scale[ch];
        // Quadratic per-channel core (curvature 0 -> the original straight line).
        var v = params.slopes[ch] * (val - params.pivots[ch]) + params.curvatures[ch] * val * val;

        // Variable-gamma paper S-curve: extra local gamma at the midtone centre
        // (v_star), easing to zero toward toe/shoulder. Mirrors the CPU kernel.
        if (mg3[ch] != 0.0) {
            v = v + mg3[ch] * params.gamma_width * tanh((v - params.v_star) / params.gamma_width);
        }

        // Regional CMY: shadow weight rises with density, highlight falls.
        let w_sh = fast_sigmoid(3.0 * (v - params.zone_center));
        let w_hi = 1.0 - w_sh;
        v = v + params.shadow_cmy[ch] * w_sh + params.highlight_cmy[ch] * w_hi;

        // Split Grade: local contrast rotation about the zone centers, mid-
        // sparing. Own block before Zone Density (sequential stays monotone).
        let w_gsh = fast_sigmoid(4.0 * (v - (params.zone_center + 0.75)));
        let w_ghi = 1.0 - fast_sigmoid(4.0 * (v - (params.zone_center - 0.40)));
        v = v + params.split_sh[ch] * w_gsh * (v - (params.zone_center + 0.75)) + params.split_hi[ch] * w_ghi * (v - (params.zone_center - 0.40));

        // Zone Density (ΔD), mid-sparing weights; +0.75 / -0.40 / 4.0 mirror
        // the zone_density_* constants in models.py — change together.
        let w_zsh = fast_sigmoid(4.0 * (v - (params.zone_center + 0.75)));
        let w_zhi = 1.0 - fast_sigmoid(4.0 * (v - (params.zone_center - 0.40)));
        v = v + params.shadow_density * w_zsh + params.d_min_rgb.w * w_zhi;

        // Shoulder: smooth lower bound at paper white (highlights).
        let v1 = d_min_eff[ch] + softplus(a_hl[ch] * (v - d_min_eff[ch])) / a_hl[ch];
        // Toe: smooth upper bound at paper black (shadows).
        dens[ch] = d_max_eff[ch] - softplus(a_sh[ch] * (d_max_eff[ch] - v1)) / a_sh[ch];
    }

    // Dye unwanted absorptions: mix the densities above paper base.
    if (params.use_dye != 0u) {
        let e = dens - d_min_rgb;
        dens = d_min_rgb + vec3<f32>(
            dot(params.dye_r.xyz, e),
            dot(params.dye_g.xyz, e),
            dot(params.dye_b.xyz, e),
        );
    }

    var transmittance = pow(vec3<f32>(10.0), -dens);
    // BPC: physical paper black -> display 0; mirrors the CPU kernel prologue
    // (negative toe raises the clip point). oetf_encode clamps the tail to 0.
    if (params.bpc != 0.0) {
        let db = vec3<f32>(params.d_max) + select(vec3<f32>(0.0), toe3 * params.toe_height, toe_neg);
        let tb = pow(vec3<f32>(10.0), -db);
        transmittance = (transmittance - tb) / (vec3<f32>(1.0) - tb);
    }

    // B&W: re-collapse after the curve — per-channel trims must not tint a
    // B&W print. Mirrors the CPU post-curve collapse in exposure/processor.py.
    if (params.mode == 1u) {
        let l = dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
        transmittance = vec3<f32>(l, l, l);
    }

    let res = vec3<f32>(
        oetf_encode(transmittance.x),
        oetf_encode(transmittance.y),
        oetf_encode(transmittance.z),
    );

    textureStore(output_tex, coords, vec4<f32>(clamp(res, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
