struct ExposureUniforms {
    pivots: vec4<f32>,
    slopes: vec4<f32>,
    cmy_offsets: vec4<f32>,
    shadow_cmy: vec4<f32>,
    highlight_cmy: vec4<f32>,
    toe: f32,
    shoulder: f32,
    toe_width: f32,
    shoulder_width: f32,
    d_min: f32,
    d_max: f32,
    a_toe_base: f32,
    a_sh_base: f32,
    width_ref: f32,
    toe_height: f32,
    sh_height: f32,
    zone_center: f32,
    flare: f32,
    surround_gamma: f32,
    mode: u32,
    v_star: f32,
    midtone_gamma: f32,
    gamma_width: f32,
    pad_a: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: ExposureUniforms;

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
    let a_hl = params.a_sh_base * params.width_ref / max(params.shoulder_width, eps);
    let a_sh_base = params.a_toe_base * params.width_ref / max(params.toe_width, eps);
    // Negative toe: tighten shadow roll-off (sharper knee) rather than extending
    // d_max_eff beyond paper black (perceptually near-zero effect above d_max).
    let a_sh = select(a_sh_base * (1.0 - params.toe * 4.0), a_sh_base, params.toe >= 0.0);
    var d_min_eff = params.d_min + params.shoulder * params.sh_height;
    if (d_min_eff < 0.0) { d_min_eff = 0.0; }
    var d_max_eff = select(params.d_max, params.d_max - params.toe * params.toe_height, params.toe >= 0.0);
    if (d_max_eff < d_min_eff + 0.1) { d_max_eff = d_min_eff + 0.1; }
    let flare_white = pow(10.0, -params.d_min);

    var res: vec3<f32>;

    for (var ch = 0; ch < 3; ch++) {
        let val = color[ch] + params.cmy_offsets[ch];
        var v = params.slopes[ch] * (val - params.pivots[ch]);

        // Variable-gamma paper S-curve: extra local gamma at the midtone centre
        // (v_star), easing to zero toward toe/shoulder. Mirrors the CPU kernel.
        if (params.midtone_gamma != 0.0) {
            v = v + params.midtone_gamma * params.gamma_width * tanh((v - params.v_star) / params.gamma_width);
        }

        // Regional CMY: shadow weight rises with density, highlight falls.
        let w_sh = fast_sigmoid(3.0 * (v - params.zone_center));
        let w_hi = 1.0 - w_sh;
        v = v + params.shadow_cmy[ch] * w_sh + params.highlight_cmy[ch] * w_hi;

        // Shoulder: smooth lower bound at paper white (highlights).
        let v1 = d_min_eff + softplus(a_hl * (v - d_min_eff)) / a_hl;
        // Toe: smooth upper bound at paper black (shadows).
        var density = d_max_eff - softplus(a_sh * (d_max_eff - v1)) / a_sh;

        if (params.surround_gamma != 1.0) {
            density = params.d_min + params.surround_gamma * (density - params.d_min);
        }

        var transmittance = pow(10.0, -density);
        if (params.flare != 0.0) {
            transmittance = (transmittance + params.flare * flare_white) / (1.0 + params.flare);
        }

        res[ch] = oetf_encode(transmittance);
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(res, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
