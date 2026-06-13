struct ExposureUniforms {
    pivots: vec4<f32>,
    slopes: vec4<f32>,
    cmy_offsets: vec4<f32>,
    shadow_cmy: vec4<f32>,
    highlight_cmy: vec4<f32>,
    toe: f32,
    toe_width: f32,
    shoulder: f32,
    shoulder_width: f32,
    d_max: f32,
    d_min: f32,
    mode: u32,
    toe_onset: f32,
    asymptote: f32,
    shoulder_beta: f32,
    nu: f32,
    flare: f32,
    surround_gamma: f32,
    pad_c: f32,
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

// sRGB OETF (linear -> display encoding); matches the Lab stage's sRGB decode.
fn srgb_oetf(t: f32) -> f32 {
    if (t <= 0.0031308) {
        return 12.92 * t;
    }
    return 1.055 * pow(t, 1.0 / 2.4) - 0.055;
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

    var res: vec3<f32>;

    for (var ch = 0; ch < 3; ch++) {
        let val = color[ch] + params.cmy_offsets[ch];
        let diff = val - params.pivots[ch];
        let epsilon = 1e-6;

        // Shoulder: integrated local gamma (1 - shoulder*M_s) on the input
        // axis, anchored at the pivot (x(0) = 0) so midtones are invariant.
        // Toe: density-domain shadow lever — raises or crushes everything
        // darker than the onset (toe_onset = 1.2 D, ~0.28 sRGB). Anchored at D = 0
        // with its tangent removed so highlights are invariant at any width —
        // the shadow zone above the pivot is too narrow for a useful
        // input-axis toe.
        // toe/shoulder arrive pre-scaled by toe_shoulder_strength.
        let a_t = params.toe_width / max(1.0 - params.pivots[ch], epsilon);
        let c_t = 0.5 * (1.0 - params.pivots[ch]);
        let a_s = params.shoulder_width / max(params.pivots[ch], epsilon);
        let c_s = -0.5 * params.pivots[ch];

        let zt = a_t * (diff - c_t);
        let zs = -a_s * (diff - c_s);
        let toe_mask = fast_sigmoid(zt);
        let shoulder_mask = fast_sigmoid(zs);

        let sig_s = -softplus(zs) / a_s;
        let sig_s0 = -softplus(-a_s * (0.0 - c_s)) / a_s;

        let x_adj = diff - params.shoulder * (sig_s - sig_s0);
        let arg = x_adj + params.shadow_cmy[ch] * toe_mask + params.highlight_cmy[ch] * shoulder_mask;

        // Richards curve toward the projected (virtual) asymptote: nu shortens
        // the toe (whites snap to paper white) and lengthens the top approach,
        // like real paper. Paper black is enforced by the soft clamp below.
        var density = params.d_min + (params.asymptote - params.d_min) * pow(fast_sigmoid(params.slopes[ch] * arg), params.nu);

        if (params.toe != 0.0) {
            let b_t = params.toe_width * 2.0;
            let d_onset = params.toe_onset;
            let sp_d = softplus(b_t * (density - d_onset)) / b_t;
            let sp_0 = softplus(b_t * (0.0 - d_onset)) / b_t;
            let sig_0 = fast_sigmoid(b_t * (0.0 - d_onset));
            density = density - params.toe * (sp_d - sp_0 - sig_0 * density);
        }

        // Surround system gamma (Bartleson-Breneman): fixed contrast expansion
        // about paper white, before the Dmax clamp so physical black is capped.
        if (params.surround_gamma != 1.0) {
            density = params.d_min + params.surround_gamma * (density - params.d_min);
        }

        // Abrupt smooth saturation shoulder at paper Dmax.
        density = density - softplus(params.shoulder_beta * (density - params.d_max)) / params.shoulder_beta;

        var transmittance = pow(10.0, -density);

        // Veiling-glare / print-flare floor in linear reflectance, normalized so
        // paper white is invariant. Lifts the deepest blacks and softens the toe.
        if (params.flare != 0.0) {
            let white = pow(10.0, -params.d_min);
            transmittance = (transmittance + params.flare * white) / (1.0 + params.flare);
        }

        res[ch] = srgb_oetf(max(transmittance, 0.0));
    }

    textureStore(output_tex, coords, vec4<f32>(clamp(res, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
